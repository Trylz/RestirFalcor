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
#pragma once

#include <DemandLoading/DeviceContext.h> // for PageMapping
#include <DemandLoading/Options.h>

#include <cuda.h>

#include <memory>
#include <mutex>
#include <vector>

namespace demandLoading {

struct DeviceContext;
class DeviceMemoryManager;
struct PageMappingsContext;
class PinnedMemoryManager;
struct RequestContext;
class RequestProcessor;
class TicketImpl;

class PagingSystem
{
  public:
    /// Create paging system, allocating device memory based on the given options.
    PagingSystem( unsigned int         deviceIndex,
                  const Options&       options,
                  DeviceMemoryManager* deviceMemoryManager,
                  PinnedMemoryManager* pinnedMemoryManager,
                  RequestProcessor*    requestProcessor );

    /// Pull requests from device to system memory.
    void pullRequests( const DeviceContext& context, CUstream stream, unsigned int startPage, unsigned int endPage, std::shared_ptr<TicketImpl> ticket );

    /// Get the device index for this paging system.
    unsigned int getDeviceIndex() const { return m_deviceIndex; }

    /// Add a page mapping (thread safe).  The device-side page table (etc.) is not updated until
    /// pushMappings is called.
    void addMapping( unsigned int pageId, unsigned int lruVal, unsigned long long entry );

    /// Check whether the specified page is resident (thread safe).
    bool isResident( unsigned int pageId );

    /// Clear a page mapping (thread safe).  The device-side page table is not updated until
    /// pushMappings is called.
    void clearMapping( unsigned int pageId );

    /// Push tile mappings to the device.  Returns the total number of new mappings.
    unsigned int pushMappings( const DeviceContext& context, CUstream stream );

    /// Invalidate (remove from m_pageTable) and return page mappings for entries in m_stalePages
    void invalidateStalePages( std::vector<PageMapping>& pageMappings, unsigned int maxMappings );

  private:
    Options              m_options{};
    unsigned int         m_deviceIndex = 0;
    DeviceMemoryManager* m_deviceMemoryManager{};
    PinnedMemoryManager* m_pinnedMemoryManager{};
    RequestProcessor*    m_requestProcessor{};

    PageMappingsContext* m_pageMappingsContext; // owned by PinnedMemoryManager::m_pageMappingsContextPool

    std::vector<bool>               m_residenceBits;  // Host-side. Not copied to/from device.
    std::vector<unsigned long long> m_pageTable;      // Host-side. Not copied to/from device. Used for eviction.
    std::mutex                      m_mutex;          // Guards m_filledPages (see addMapping).

    // Variables related to LRU
    unsigned int m_launchNum    = 0;
    unsigned int m_lruThreshold = 0;

    // A host function callback is used to invoke processRequests().
    friend void processRequestsCallback( void* ptr );

    // Process requests, inserting them in the global request queue.
    void processRequests( const DeviceContext& context, RequestContext* requestContext, CUstream stream, std::shared_ptr<TicketImpl> ticket );

    /// Increment the lru threshold value
    void incrementLruThreshold( unsigned int returnedStalePages, unsigned int requestedStalePages, unsigned int medianLruVal );
};


}  // namespace demandLoading
