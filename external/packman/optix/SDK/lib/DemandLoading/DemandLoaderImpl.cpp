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

#include "DemandLoaderImpl.h"

#include "RequestProcessor.h"
#include "Util/Exception.h"
#include "Util/NVTXProfiling.h"
#include "Util/Stopwatch.h"

#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/TileIndexing.h>

#include <cuda.h>

#include <algorithm>
#include <set>

namespace {

demandLoading::Options configure( demandLoading::Options options )
{
    // If maxTexMemPerDevice is 0, consider it to be unlimited
    if( options.maxTexMemPerDevice == 0 )
        options.maxTexMemPerDevice = 0xfffffffffffffffful;

    // PagingSystem::pushMappings requires enough capacity to handle all the requested pages.
    if( options.maxFilledPages < options.maxRequestedPages )
        options.maxFilledPages = options.maxRequestedPages;

    // Anticipate at lease one active stream per device.
    int deviceCount;
    DEMAND_CUDA_CHECK( cudaGetDeviceCount( &deviceCount ) );
    options.maxActiveStreams = std::max( static_cast<unsigned int>( deviceCount ), options.maxActiveStreams );

    return options;
}

bool supportsSparseTextures( unsigned int deviceIndex )
{
    int sparseSupport = 0;
    DEMAND_CUDA_CHECK( cuDeviceGetAttribute( &sparseSupport, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED, deviceIndex ) );
    return static_cast<bool>( sparseSupport );
}

}  // anonymous namespace

namespace demandLoading {

DemandLoaderImpl::DemandLoaderImpl( const std::vector<unsigned int>& devices, const Options& options )
    : m_options( configure( options ) )
    , m_numDevices( static_cast<unsigned int>( devices.size() ) )
    , m_deviceMemoryManagers( devices.size() )
    , m_pagingSystems( devices.size() )
    , m_samplerRequestHandler( this )
    , m_pageTableManager( options.numPages )
    , m_requestProcessor( &m_pageTableManager )
    , m_pinnedMemoryManager( options )
{
    // Create per-device DeviceMemoryManager, PagingSystem, and RequestHandler
    unsigned int numCapableDevices = 0;
    for( unsigned int deviceIndex : devices )
    {
        if( supportsSparseTextures( deviceIndex ) )
        {
            ++numCapableDevices;
            m_deviceMemoryManagers[deviceIndex].reset( new DeviceMemoryManager( deviceIndex, m_options ) );
            m_pagingSystems[deviceIndex].reset( new PagingSystem( deviceIndex, options,
                                                                  m_deviceMemoryManagers[deviceIndex].get(),
                                                                  &m_pinnedMemoryManager, &m_requestProcessor ) );
        }
    }
    if( numCapableDevices == 0 )
        throw Exception( "No devices that support CUDA sparse textures were found (sm_60+ required)." );

    // Reserve virtual address space for texture samplers, which is associated with the sampler request handler.
    m_pageTableManager.reserve( m_options.numPageTableEntries, &m_samplerRequestHandler );

    m_requestProcessor.start( m_options.maxThreads );
}

DemandLoaderImpl::~DemandLoaderImpl()
{
    m_requestProcessor.stop();
}

// Create a demand-loaded texture.  The image is not opened until the texture sampler is requested
// by device code (via pagingMapOrRequest in Tex2D).
const DemandTexture& DemandLoaderImpl::createTexture( std::shared_ptr<ImageReader> imageReader, const TextureDescriptor& textureDesc )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // The texture id will be the next index in the texture array.
    unsigned int textureId = static_cast<unsigned int>( m_textures.size() );

    // Add new texture to the end of the list of textures.  The texture holds a pointer to the
    // image, from which tile data is obtained on demand.
    m_textures.emplace_back( new DemandTextureImpl( textureId, m_numDevices, textureDesc, imageReader, this ) );

    return *m_textures.back();
}

unsigned int DemandLoaderImpl::createResource( unsigned int numPages, ResourceCallback callback )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // Create a request handler that wraps the callback.  These are individually allocated to avoid
    // dangling pointers in the PageTableManager when the request handler vector is resized.
    m_resourceRequestHandlers.emplace_back( new ResourceRequestHandler( callback, this ) );

    // Reserve virtual address space for the resource, which is associated with the request handler.
    m_pageTableManager.reserve( numPages, m_resourceRequestHandlers.back().get() );

    // Return the start page.
    return m_resourceRequestHandlers.back()->getStartPage();
}


// Returns false if the device doesn't support sparse textures.
bool DemandLoaderImpl::launchPrepare( unsigned int deviceIndex, CUstream stream, DeviceContext& context )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    PagingSystem* pagingSystem = m_pagingSystems.at( deviceIndex ).get();
    if( pagingSystem == nullptr )
        return false;

    // Get DeviceContext from pool and copy it to output parameter.
    context = *m_deviceMemoryManagers[deviceIndex]->getDeviceContextPool()->allocate();

    pagingSystem->pushMappings( context, stream );
    return true;
}

// Process page requests.
std::shared_ptr<Ticket> DemandLoaderImpl::processRequests( unsigned int deviceIndex, CUstream stream, const DeviceContext& context )
{
    Stopwatch stopwatch;
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    std::unique_lock<std::mutex> lock( m_mutex );

    // Create a Ticket that the caller can use to track request processing.
    std::shared_ptr<TicketImpl> ticket( std::make_shared<TicketImpl>() );

    // Pull requests from the device.  This launches a kernel on the given stream to scan the
    // request bits copies the requested page ids to host memory (asynchronously).
    PagingSystem* pagingSystem = m_pagingSystems[deviceIndex].get();
    unsigned int  startPage    = 0;
    unsigned int  endPage      = m_pageTableManager.getHighestUsedPage();
    pagingSystem->pullRequests( context, stream, startPage, endPage, ticket );

    m_totalProcessingTime += stopwatch.elapsed();
    return ticket;
}

void DemandLoaderImpl::unmapTileResource( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    // Ask the PageTableManager for the RequestHandler associated with the given page index.
    TextureRequestHandler* handler = dynamic_cast<TextureRequestHandler*>( m_pageTableManager.getRequestHandler( pageId ) );
    DEMAND_ASSERT_MSG( handler != nullptr, "Page request does not correspond to a known resource" );
    DemandTextureImpl* texture   = handler->getTexture();
    const unsigned int tileIndex = pageId - texture->getSampler().startPage;

    // Unmap the tile or mip tail
    if( tileIndex == 0 )
    {
        texture->unmapMipTail( deviceIndex, stream );
    }
    else
    {
        unsigned int mipLevel;
        unsigned int tileX;
        unsigned int tileY;
        unpackTileIndex( texture->getSampler(), tileIndex, mipLevel, tileX, tileY );
        texture->unmapTile( deviceIndex, stream, mipLevel, tileX, tileY );
    }
}

void DemandLoaderImpl::freeTiles( unsigned int deviceIndex, CUstream stream )
{
    // TODO: the code below just adds the tiles to the free list without making sure they
    // can't be used on the GPU. Freeing tiles must be asynchronous and stream-aware.
    TilePool*    tilePool           = m_deviceMemoryManagers[deviceIndex]->getTilePool();
    unsigned int desiredTilesToFree = tilePool->getDesiredTilesToFree();
    if( desiredTilesToFree > 0 )
    {
        PagingSystem*            pagingSystem = m_pagingSystems[deviceIndex].get();
        std::vector<PageMapping> staleMemBlocks;
        pagingSystem->invalidateStalePages( staleMemBlocks, desiredTilesToFree );
        for( PageMapping m : staleMemBlocks )
        {
            unmapTileResource( deviceIndex, stream, m.id );
            tilePool->freeBlock( m.page );
        }
    }
}

Statistics DemandLoaderImpl::getStatistics() const
{
    std::unique_lock<std::mutex> lock( m_mutex );
    Statistics                   stats{};
    stats.requestProcessingTime = m_totalProcessingTime;

    // Multiple textures might share the same ImageReader, so we create a set as we go to avoid
    // duplicate counting.
    std::set<ImageReader*> images;
    for( auto& tex : m_textures )
    {
        ImageReader* image = tex->getImageReader();
        if( images.find( image ) == images.end() )
        {
            images.insert( image );
            stats.numTilesRead += image->getNumTilesRead();
            stats.numBytesRead += image->getNumBytesRead();
            stats.readTime += image->getTotalReadTime();
        }
    }

    size_t maxNumDevices = sizeof( Statistics::memoryUsedPerDevice ) / sizeof( size_t );
    for( unsigned int i = 0; i < m_deviceMemoryManagers.size() && i < maxNumDevices; ++i )
    {
        if( m_deviceMemoryManagers[i] )
            stats.memoryUsedPerDevice[i] = m_deviceMemoryManagers[i]->getTotalDeviceMemory();
    }
    return stats;
}

DemandLoader* createDemandLoader( const std::vector<unsigned int>& devices, const Options& options )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    return new DemandLoaderImpl( devices, options );
}

void destroyDemandLoader( DemandLoader* manager )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();
    delete manager;
}

}  // namespace demandLoading
