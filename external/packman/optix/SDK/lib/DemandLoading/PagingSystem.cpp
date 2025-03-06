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

#include "PagingSystem.h"
#include "DemandLoaderImpl.h"
#include "Memory/PinnedMemoryManager.h"
#include "PagingSystemKernels.h"
#include "RequestProcessor.h"

#include <algorithm>

namespace demandLoading {

PagingSystem::PagingSystem( unsigned int         deviceIndex,
                            const Options&       options,
                            DeviceMemoryManager* deviceMemoryManager,
                            PinnedMemoryManager* pinnedMemoryManager,
                            RequestProcessor*    requestProcessor )
    : m_options( options )
    , m_deviceIndex( deviceIndex )
    , m_deviceMemoryManager( deviceMemoryManager )
    , m_pinnedMemoryManager( pinnedMemoryManager )
    , m_requestProcessor( requestProcessor )
{
    DEMAND_ASSERT( m_options.maxFilledPages >= m_options.maxRequestedPages );
    DEMAND_CUDA_CHECK( cudaSetDevice( deviceIndex ) );

    // Allocate host-side page table
    m_pageTable.resize( m_options.numPages, 0ul );
    m_residenceBits.resize( m_options.numPages, false );

    // Allocate pinned memory for page mappings and invalidated pages (see pushMappings).  Note that
    // it's not necessary to free m_pageMappingsContext in the destructor, since it's pool
    // allocated.  (Nor is it possible, because doing so requires a stream.)
    m_pageMappingsContext = m_pinnedMemoryManager->getPageMappingsContextPool()->allocate();
    m_pageMappingsContext->clear();
}

void PagingSystem::incrementLruThreshold( unsigned int returnedStalePages, unsigned int requestedStalePages, unsigned int medianLruVal )
{
    // Don't change the value if no stale pages were requested
    if( requestedStalePages == 0 )
        return;

    // Heuristic to increment the lruThreshold. The basic idea is to aggressively reduce the threshold
    // if not enough stale pages are returned, but only gradually increase the threshold if it is too low.
    if( returnedStalePages < requestedStalePages / 2 )
        m_lruThreshold -= std::min( m_lruThreshold, 4u );
    else if( returnedStalePages < requestedStalePages )
        m_lruThreshold -= std::min( m_lruThreshold, 2u );
    else if( medianLruVal > m_lruThreshold )
        m_lruThreshold++;
}

// Argument struct for processRequestsCallback.
struct ProcessRequestCallbackArg
{
    PagingSystem*               pagingSystem;
    DeviceContext               context;
    RequestContext*             pinnedContext;
    CUstream                    stream;
    std::shared_ptr<TicketImpl> ticket;
};

// Free function adapter for the processRequests method. pullRequests() uses cuLaunchHostFunc() to
// enqueue a call to this function after it launches a kernel to pull requests from the device.
// That's why this is a free function that takes a void* argument, since it must adhere to the
// CUhostFn function type.
void processRequestsCallback( void* ptr )
{
    ProcessRequestCallbackArg* arg = reinterpret_cast<ProcessRequestCallbackArg*>( ptr );
    arg->pagingSystem->processRequests( arg->context, arg->pinnedContext, arg->stream, arg->ticket );
    delete arg;
}

void PagingSystem::pullRequests( const DeviceContext&        context,
                                 CUstream                    stream,
                                 unsigned int                startPage,
                                 unsigned int                endPage,
                                 std::shared_ptr<TicketImpl> ticket )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // The array lengths are accumulated across multiple device threads, so they must be initialized to zero.
    DEMAND_CUDA_CHECK( cudaMemsetAsync( context.arrayLengths.data, 0, context.arrayLengths.capacity * sizeof( unsigned int ), stream ) );

    DEMAND_ASSERT( startPage <= endPage );
    DEMAND_ASSERT( endPage < m_options.numPages );
    m_launchNum++;

    launchPullRequests( stream, context, m_launchNum, m_lruThreshold, startPage, endPage );

    // Get a RequestContext from the pinned memory manager, which will serve as the destination for
    // asynchronous copies of the requested pages, etc.
    RequestContext* pinnedContext = m_pinnedMemoryManager->getRequestContextPool()->allocate();

    // Copy the requested page list from this device.  The actual length is unknown, so we copy the entire capacity
    // and update the length below.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( pinnedContext->requestedPages, context.requestedPages.data,
                                        pinnedContext->maxRequestedPages * sizeof( unsigned int ), cudaMemcpyDeviceToHost, stream ) );

    // Get the stale pages from the device. This may be a subset of the actual stale pages.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( pinnedContext->stalePages, context.stalePages.data,
                                        pinnedContext->maxStalePages * sizeof( StalePage ), cudaMemcpyDeviceToHost, stream ) );

    // Get the sizes of the requested/stale page lists.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( pinnedContext->arrayLengths, context.arrayLengths.data,
                                        pinnedContext->numArrayLengths * sizeof( unsigned int ), cudaMemcpyDeviceToHost, stream ) );

    // Enqueue host function call to process the page requests once the kernel launch and copies have completed.
    ProcessRequestCallbackArg* arg = new ProcessRequestCallbackArg{this, context, pinnedContext, stream, ticket};
    DEMAND_CUDA_CHECK( cuLaunchHostFunc( stream, processRequestsCallback, arg ) );
}

// Note: this method must not make any CUDA API calls, because it's invoked via cuLaunchHostFunc.
void PagingSystem::processRequests( const DeviceContext& context, RequestContext* requestContext, CUstream stream, std::shared_ptr<TicketImpl> ticket )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Return device context to pool.  The DeviceContext has been copied, but DeviceContextPool is designed to permit that.
    m_deviceMemoryManager->getDeviceContextPool()->free( const_cast<DeviceContext*>( &context ) );

    unsigned int numRequestedPages = requestContext->arrayLengths[PAGE_REQUESTS_LENGTH];
    unsigned int numStalePages     = requestContext->arrayLengths[STALE_PAGES_LENGTH];

    // Enqueue the requests for asynchronous processing.
    m_requestProcessor->addRequests( m_deviceIndex, stream, requestContext->requestedPages, numRequestedPages, ticket );

    unsigned int medianLruVal = 0;
    if( numStalePages > 0 )
    {
        if( context.lruTable != nullptr )
        {
            std::sort( requestContext->stalePages, requestContext->stalePages + numStalePages,
                       []( StalePage a, StalePage b ) { return a.lruVal < b.lruVal; } );
            medianLruVal = requestContext->stalePages[numStalePages / 2].lruVal;
        }
        else
        {
            std::random_shuffle( requestContext->stalePages, requestContext->stalePages + numStalePages );
        }
    }

    incrementLruThreshold( numStalePages, requestContext->maxStalePages, medianLruVal );

    // TODO: how to handle freeTiles?
    // m_demandLoader->freeTiles( stream );

    // Return the RequestContext to its pool.
    m_pinnedMemoryManager->getRequestContextPool()->free( requestContext );
}


void PagingSystem::addMapping( unsigned int pageId, unsigned int lruVal, unsigned long long entry )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    DEMAND_ASSERT_MSG( m_pageMappingsContext->numFilledPages < m_pageMappingsContext->maxFilledPages,
                       "Maximum number of filled pages exceeded (Options::maxFilledPages)" );
    m_pageMappingsContext->filledPages[m_pageMappingsContext->numFilledPages++] = PageMapping{pageId, lruVal, entry};

    DEMAND_ASSERT( m_residenceBits[pageId] == false );
    m_pageTable[pageId] = entry;

    DEMAND_ASSERT( pageId < m_pageTable.size() );
    m_residenceBits[pageId] = true;

}

bool PagingSystem::isResident( unsigned int pageId )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    return m_residenceBits[pageId];
}

void PagingSystem::clearMapping( unsigned int pageId )
{
    // Mutex is acquired in caller (invalidateStalePages).
    DEMAND_ASSERT( m_pageMappingsContext->numInvalidatedPages < m_pageMappingsContext->maxInvalidatedPages );
    m_pageMappingsContext->invalidatedPages[m_pageMappingsContext->numInvalidatedPages++] = pageId;

    DEMAND_ASSERT( pageId < m_pageTable.size() );
    m_pageTable[pageId] = 0ull;

    DEMAND_ASSERT( m_residenceBits[pageId] == true );
    m_residenceBits[pageId] = false;
}

unsigned int PagingSystem::pushMappings( const DeviceContext& context, CUstream stream )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    const unsigned int numFilledPages = m_pageMappingsContext->numFilledPages;
    if( numFilledPages > 0 )
    {
        DEMAND_CUDA_CHECK( cudaMemcpyAsync( context.filledPages.data, m_pageMappingsContext->filledPages,
                                            numFilledPages * sizeof( PageMapping ), cudaMemcpyHostToDevice, stream ) );
        launchPushMappings( stream, context, numFilledPages );
    }

    const unsigned int numInvalidatedPages = m_pageMappingsContext->numInvalidatedPages;
    if( numInvalidatedPages > 0 )
    {
        DEMAND_CUDA_CHECK( cudaMemcpyAsync( context.invalidatedPages.data, m_pageMappingsContext->invalidatedPages,
                                            numInvalidatedPages * sizeof( unsigned int ), cudaMemcpyHostToDevice, stream ) );
        launchInvalidatePages( stream, context, numInvalidatedPages );
    }

    // Free the current PageMappingsContext (it's not reused until the preceding operations on the stream are done)
    // and allocate another one.
    m_pinnedMemoryManager->getPageMappingsContextPool()->free( m_pageMappingsContext, m_deviceIndex, stream );
    m_pageMappingsContext = m_pinnedMemoryManager->getPageMappingsContextPool()->allocate();
    m_pageMappingsContext->clear();

    // Zero out the reference bits
    unsigned int referenceBitsSizeInBytes = idivCeil( context.maxNumPages, 8 );
    DEMAND_CUDA_CHECK( cudaMemsetAsync( context.referenceBits, 0, referenceBitsSizeInBytes, stream ) );

    return numFilledPages;
}

void PagingSystem::invalidateStalePages( std::vector<PageMapping>& pageMappings, unsigned int maxMappings )
{
#if 0 
    // This needs work.  The list of stale pages now resides in RequestContext, which lives only for
    // the duration of processRequests().
    std::unique_lock<std::mutex> lock( m_mutex );
    unsigned int                 numInvalidated = 0;
    for( StalePage& sp : m_stalePages )
    {
        if( numInvalidated >= maxMappings || m_pageMappingsContext->numInvalidatedPages >= m_options.maxInvalidatedPages )
            break;

        if( m_residenceBits[sp.pageId] )
        {
            pageMappings.push_back( PageMapping{sp.pageId, sp.lruVal, m_pageTable[sp.pageId]} );
            clearMapping( sp.pageId );
            numInvalidated++;
        }
    }
#endif
}

}  // namespace demandLoading
