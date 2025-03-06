//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <DemandLoading/EXRReader.h>

#include "Util/Exception.h"
#include "Util/Stopwatch.h"

#include <cuda_runtime.h>

#include <ImfChannelList.h>
#include <ImfTiledInputFile.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

using namespace Imf;
using namespace Imath;

namespace demandLoading {

CUarray_format pixelTypeToArrayFormat( PixelType type )
{
    switch( type )
    {
        case UINT:
            return CU_AD_FORMAT_UNSIGNED_INT32;
        case HALF:
            return CU_AD_FORMAT_HALF;
        case FLOAT:
            return CU_AD_FORMAT_FLOAT;
        default:
            DEMAND_ASSERT_MSG( false, "Invalid EXR pixel type" );
            return CU_AD_FORMAT_FLOAT;
    }
}

// Open the image and read header info, including dimensions and format.  Returns false on error.
bool EXRReader::open( TextureInfo* info )
{
    {
        Stopwatch stopwatch;
        std::unique_lock<std::mutex> lock( m_mutex );

        // Check to see if the image is already open
        if( m_info.width == 0 && m_info.height == 0 )
        {
            // Open input file.
            DEMAND_ASSERT( !m_inputFile );
            m_inputFile.reset( new TiledInputFile( m_filename.c_str() ) );

            // Get the width and height from the data window of the finest mipLevel.
            const Box2i dw = m_inputFile->dataWindowForLevel( 0, 0 );
            m_info.width   = dw.max.x - dw.min.x + 1;
            m_info.height  = dw.max.y - dw.min.y + 1;

            // Note that non-power-of-two EXR files often have one fewer miplevel than one would expect
            // (they don't round up from 1+log2(max(width/height))).
            DEMAND_ASSERT( m_inputFile->numLevels() != 0 );
            m_info.numMipLevels = m_inputFile->numLevels();

            m_tileWidth  = m_inputFile->tileXSize();
            m_tileHeight = m_inputFile->tileYSize();

            // Get channel info from the header.  Missing channels will be filled with zeros
            // by the FrameBuffer/Slice logic below.

            const ChannelList& channels = m_inputFile->header().channels();

            const Channel* R = channels.findChannel( "R" );
            const Channel* G = channels.findChannel( "G" );
            const Channel* B = channels.findChannel( "B" );
            const Channel* A = channels.findChannel( "A" );

            DEMAND_ASSERT_MSG( R, "First channel is missing in EXR file" );
            m_pixelType   = R->type;
            m_info.format = pixelTypeToArrayFormat( m_pixelType );

            // CUDA textures don't support float3, so we round up to four channels.
            m_info.numChannels = A ? 4 : ( B ? 4 : ( G ? 2 : 1 ) );
        }

        m_totalReadTime += stopwatch.elapsed();
    }

    if( info != nullptr )
        *info = m_info;
    return true;
}

// Do the setup work for a FrameBuffer, putting in the slices as needed based on the channelDesc
void EXRReader::setupFrameBuffer( Imf::FrameBuffer& frameBuffer, char* base, size_t xStride, size_t yStride )
{
    const unsigned int channelSize = getBytesPerChannel( m_info.format );
    frameBuffer.insert( "R", Slice( m_pixelType, base, xStride, yStride ) );
    if( m_info.numChannels > 1 )
    {
        frameBuffer.insert( "G", Slice( m_pixelType, &base[1 * channelSize], xStride, yStride ) );
    }
    if( m_info.numChannels > 2 )
    {
        // CUDA textures don't support float3, so we round up to four channels.
        frameBuffer.insert( "B", Slice( m_pixelType, &base[2 * channelSize], xStride, yStride ) );
        frameBuffer.insert( "A", Slice( m_pixelType, &base[3 * channelSize], xStride, yStride ) );
    }
}

// Close the image.
void EXRReader::close()
{
    m_inputFile.reset();
}

void EXRReader::readActualTile( char* dest, unsigned int rowPitch, unsigned int mipLevel, unsigned int tileX, unsigned int tileY )
{
    DEMAND_ASSERT( m_inputFile );

    // Get the data window for the tile, which reflects whether it's a partial tile.
    const Box2i dw = m_inputFile->dataWindowForTile( tileX, tileY, mipLevel );

    // Compute base pointer and strides for frame buffer
    const unsigned int bytesPerPixel = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const size_t       xStride       = bytesPerPixel;
    const size_t       yStride       = rowPitch;
    char*              base          = dest - ( ( dw.min.x + dw.min.y * rowPitch / bytesPerPixel ) * bytesPerPixel );

    // Create frame buffer for the tile and read the tile data.
    FrameBuffer frameBuffer;
    setupFrameBuffer( frameBuffer, base, xStride, yStride );
    m_inputFile->setFrameBuffer( frameBuffer );
    m_inputFile->readTile( tileX, tileY, mipLevel );
}

bool EXRReader::readTile( char* dest, unsigned int mipLevel, unsigned int tileX, unsigned int tileY, unsigned int tileWidth, unsigned int tileHeight )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Stats tracking
    Stopwatch stopwatch;

    // We require that the requested tile size is an integer multiple of the EXR tile size.
    const unsigned int actualTileWidth  = m_inputFile->tileXSize();
    const unsigned int actualTileHeight = m_inputFile->tileYSize();
    if( !( actualTileWidth <= tileWidth && tileWidth % actualTileWidth == 0 )
        || !( actualTileHeight <= tileHeight && tileHeight % actualTileHeight == 0 ) )
    {
        std::stringstream str;
        str << "Unsupported EXR tile size (" << actualTileWidth << "x" << actualTileHeight << ").  Expected "
            << tileWidth << "x" << tileHeight << " (or a whole fraction thereof) for this pixel format";
        throw Exception( str.str().c_str() );
    }

    const unsigned int actualTileX    = tileX * tileWidth / actualTileWidth;
    const unsigned int actualTileY    = tileY * tileHeight / actualTileHeight;
    const unsigned int numTilesX      = tileWidth / actualTileWidth;
    const unsigned int numTilesY      = tileHeight / actualTileHeight;
    const unsigned int bytesPerPixel  = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const unsigned int rowPitch       = tileWidth * bytesPerPixel;
    const size_t       actualTileSize = actualTileWidth * actualTileHeight * bytesPerPixel;

    for( unsigned int j = 0; j < numTilesY; ++j )
    {
        for( unsigned int i = 0; i < numTilesX; ++i )
        {
            char* start = dest + j * numTilesX * actualTileSize + i * actualTileWidth * bytesPerPixel;
            readActualTile( start, rowPitch, mipLevel, actualTileX + i, actualTileY + j );

            // Stats tracking
            m_numBytesRead += actualTileSize;
        }
    }

    // Stats tracking
    m_numTilesRead += 1;
    m_totalReadTime += stopwatch.elapsed();

    return true;
}

bool EXRReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    DEMAND_ASSERT( m_inputFile );

    // Stats tracking
    Stopwatch stopwatch;

    // Get miplevel data window offset and dimensions.
    const Box2i dw    = m_inputFile->dataWindowForLevel( mipLevel, mipLevel );
    const int   width = dw.max.x - dw.min.x + 1;
    DEMAND_ASSERT( width == static_cast<int>( expectedWidth ) );
    DEMAND_ASSERT( ( dw.max.y - dw.min.y + 1 ) == static_cast<int>( expectedHeight ) );

    // Compute base pointer and strides for frame buffer
    const unsigned int bytesPerPixel = getBytesPerChannel( m_info.format ) * m_info.numChannels;
    const size_t       xStride       = bytesPerPixel;
    const size_t       yStride       = width * xStride;
    char*              base          = dest - ( ( dw.min.x + dw.min.y * width ) * bytesPerPixel );

    // Create frame buffer and read the tiles for the specified mipLevel.
    FrameBuffer frameBuffer;
    setupFrameBuffer( frameBuffer, base, xStride, yStride );
    m_inputFile->setFrameBuffer( frameBuffer );
    m_inputFile->readTiles( 0, m_inputFile->numXTiles( mipLevel ) - 1, 0, m_inputFile->numYTiles( mipLevel ) - 1, mipLevel, mipLevel );

    // Stats tracking
    {
        const unsigned int actualTileWidth  = m_inputFile->tileXSize();
        const unsigned int actualTileHeight = m_inputFile->tileYSize();
        const size_t       actualTileSize   = actualTileWidth * actualTileHeight * bytesPerPixel;
        const int          numXTiles        = m_inputFile->numXTiles( mipLevel );
        const int          numYTiles        = m_inputFile->numYTiles( mipLevel );

        m_numTilesRead += numXTiles * numYTiles;
        m_numBytesRead += numXTiles * numYTiles * actualTileSize;
        m_totalReadTime += stopwatch.elapsed();
    }

    return true;
}

}  // namespace demandLoading
