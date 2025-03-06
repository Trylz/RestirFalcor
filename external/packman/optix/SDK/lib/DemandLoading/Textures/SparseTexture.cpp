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

#include <DemandLoading/ImageReader.h>

#include "Textures/SparseTexture.h"
#include "Util/Exception.h"

#include <algorithm>
#include <cmath>

namespace demandLoading {

void SparseTexture::init( const TextureDescriptor& descriptor, const TextureInfo& info )
{
    // Redundant initialization can occur because requests from multiple streams are not yet
    // deduplicated.
    if( m_isInitialized )
        return;

    m_isInitialized = true;
    m_info          = info;
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Work around an invalid read (reported by valgrind) in cuMipmappedArrayCreate when the number
    // of miplevels is less than the start of the mip tail.  See bug 3139148.
    // Note that the texture descriptor clamps the maximum miplevel appropriately, and we'll never
    // map tiles (or the mip tail) beyond the actual maximum miplevel.
    const unsigned int nominalNumMipLevels = calculateNumMipLevels( m_info.width, m_info.height );
    DEMAND_ASSERT( info.numMipLevels <= nominalNumMipLevels );

    // Create CUDA array
    CUDA_ARRAY3D_DESCRIPTOR ad{};
    ad.Width       = info.width;
    ad.Height      = info.height;
    ad.Format      = info.format;
    ad.NumChannels = info.numChannels;
    ad.Flags       = CUDA_ARRAY3D_SPARSE;
    DEMAND_CUDA_CHECK( cuMipmappedArrayCreate( &m_array, &ad, nominalNumMipLevels ) );

    // Get sparse texture properties
    DEMAND_CUDA_CHECK( cuMipmappedArrayGetSparseProperties( &m_properties, m_array ) );

    // Precompute array of mip level dimensions (for use in getTileDimensions).
    for( unsigned int mipLevel = 0; mipLevel < info.numMipLevels; ++mipLevel )
    {
        m_mipLevelDims.push_back( queryMipLevelDims( mipLevel ) );
    }

    // Create CUDA texture descriptor
    CUDA_TEXTURE_DESC td{};
    td.addressMode[0]      = descriptor.addressMode[0];
    td.addressMode[1]      = descriptor.addressMode[1];
    td.filterMode          = descriptor.filterMode;
    td.flags               = CU_TRSF_NORMALIZED_COORDINATES | descriptor.flags;
    td.maxAnisotropy       = descriptor.maxAnisotropy;
    td.mipmapFilterMode    = descriptor.mipmapFilterMode;
    td.maxMipmapLevelClamp = float( info.numMipLevels - 1 );
    td.minMipmapLevelClamp = 0.f;

    // Create texture object.
    CUDA_RESOURCE_DESC rd{};
    rd.resType                    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    rd.res.mipmap.hMipmappedArray = m_array;
    DEMAND_CUDA_CHECK( cuTexObjectCreate( &m_texture, &rd, &td, 0 ) );
};


uint2 SparseTexture::queryMipLevelDims( unsigned int mipLevel ) const
{
    // Get CUDA array for the specified level from the mipmapped array.
    DEMAND_ASSERT( mipLevel < m_info.numMipLevels );
    CUarray mipLevelArray;
    DEMAND_CUDA_CHECK( cuMipmappedArrayGetLevel( &mipLevelArray, m_array, mipLevel ) );

    // Get the array descriptor.
    CUDA_ARRAY_DESCRIPTOR desc;
    DEMAND_CUDA_CHECK( cuArrayGetDescriptor( &desc, mipLevelArray ) );

    return make_uint2( static_cast<unsigned int>( desc.Width ), static_cast<unsigned int>( desc.Height ) );
}


// Get the dimensions of the specified tile, which might be a partial tile.
uint2 SparseTexture::getTileDimensions( unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    unsigned int startX = tileX * getTileWidth();
    unsigned int startY = tileY * getTileHeight();
    unsigned int endX   = startX + getTileWidth();
    unsigned int endY   = startY + getTileHeight();

    // TODO: cache the level dimensions.
    uint2 levelDims = getMipLevelDims( mipLevel );
    endX            = std::min( endX, levelDims.x );
    endY            = std::min( endY, levelDims.y );

    return make_uint2( endX - startX, endY - startY );
}


void SparseTexture::fillTile( CUstream                     stream,
                              unsigned int                 mipLevel,
                              unsigned int                 tileX,
                              unsigned int                 tileY,
                              const char*                  tileData,
                              size_t                       tileSize,
                              CUmemGenericAllocationHandle tileHandle,
                              size_t                       tileOffset ) const
{
    // Make device current.
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Map tile backing storage into array
    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType               = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = mipLevel;

    mapInfo.subresource.sparseLevel.offsetX = tileX * getTileWidth();
    mapInfo.subresource.sparseLevel.offsetY = tileY * getTileHeight();

    uint2 tileDims                               = getTileDimensions( mipLevel, tileX, tileY );
    mapInfo.subresource.sparseLevel.extentWidth  = tileDims.x;
    mapInfo.subresource.sparseLevel.extentHeight = tileDims.y;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = tileHandle;
    mapInfo.offset              = tileOffset;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );

    // Get CUDA array for the specified miplevel.
    CUarray mipLevelArray;
    DEMAND_CUDA_CHECK( cuMipmappedArrayGetLevel( &mipLevelArray, m_array, mipLevel ) );

    // Copy tile data into CUDA array
    const unsigned int pixelSize = m_info.numChannels * getBytesPerChannel( m_info.format );
    CUDA_MEMCPY2D      copyArgs  = {};
    copyArgs.srcMemoryType       = CU_MEMORYTYPE_HOST;
    copyArgs.srcHost             = tileData;
    copyArgs.srcPitch            = getTileWidth() * pixelSize;

    copyArgs.dstXInBytes = tileX * getTileWidth() * pixelSize;
    copyArgs.dstY        = tileY * getTileHeight();

    copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyArgs.dstArray      = mipLevelArray;

    copyArgs.WidthInBytes = tileDims.x * pixelSize;
    copyArgs.Height       = tileDims.y;

    DEMAND_CUDA_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );
}


void SparseTexture::unmapTile( CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const
{
    // Make device current.
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType               = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = mipLevel;

    mapInfo.subresource.sparseLevel.offsetX = tileX * getTileWidth();
    mapInfo.subresource.sparseLevel.offsetY = tileY * getTileHeight();

    uint2 tileDims                               = getTileDimensions( mipLevel, tileX, tileY );
    mapInfo.subresource.sparseLevel.extentWidth  = tileDims.x;
    mapInfo.subresource.sparseLevel.extentHeight = tileDims.y;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = 0ULL;
    mapInfo.offset              = 0ULL;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}


void SparseTexture::fillMipTail( CUstream stream, const char* mipTailData, size_t mipTailSize, CUmemGenericAllocationHandle tileHandle, size_t tileOffset ) const
{
    // Make device current.
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType            = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = getMipTailSize();

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = tileHandle;
    mapInfo.offset              = tileOffset;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );

    // Fill each level in the mip tail.
    size_t             offset    = 0;
    const unsigned int pixelSize = m_info.numChannels * getBytesPerChannel( m_info.format );
    for( unsigned int mipLevel = getMipTailFirstLevel(); mipLevel < m_info.numMipLevels; ++mipLevel )
    {
        CUarray mipLevelArray;
        DEMAND_CUDA_CHECK( cuMipmappedArrayGetLevel( &mipLevelArray, m_array, mipLevel ) );
        uint2 levelDims = getMipLevelDims( mipLevel );

        CUDA_MEMCPY2D copyArgs{};
        copyArgs.srcMemoryType = CU_MEMORYTYPE_HOST;
        copyArgs.srcHost       = mipTailData + offset;
        copyArgs.srcPitch      = levelDims.x * pixelSize;

        copyArgs.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyArgs.dstArray      = mipLevelArray;

        copyArgs.WidthInBytes = levelDims.x * pixelSize;
        copyArgs.Height       = levelDims.y;

        DEMAND_CUDA_CHECK( cuMemcpy2DAsync( &copyArgs, stream ) );

        offset += levelDims.x * levelDims.y * pixelSize;
    }
}


void SparseTexture::unmapMipTail( CUstream stream ) const
{
    // Make device current.
    DEMAND_ASSERT( m_isInitialized );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUarrayMapInfo mapInfo{};
    mapInfo.resourceType    = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_array;

    mapInfo.subresourceType            = CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = getMipTailSize();

    mapInfo.memOperationType    = CU_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType       = CU_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = 0ULL;
    mapInfo.offset              = 0ULL;
    mapInfo.deviceBitMask       = 1U << m_deviceIndex;

    DEMAND_CUDA_CHECK( cuMemMapArrayAsync( &mapInfo, 1, stream ) );
}


SparseTexture::~SparseTexture()
{
    if( m_isInitialized )
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        DEMAND_CUDA_CHECK( cuTexObjectDestroy( m_texture ) );

        // It's not necessary to unmap the tiles / mip tail when destroying the array.
        DEMAND_CUDA_CHECK( cuMipmappedArrayDestroy( m_array ) );
    }
}

}  // namespace demandLoading
