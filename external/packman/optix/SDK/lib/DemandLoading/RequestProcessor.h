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

#include "RequestQueue.h"

#include <cuda.h>

#include <thread>
#include <vector>

namespace demandLoading {

class PageTableManager;

class RequestProcessor
{
  public:
    /// Construct request processor, which uses the given PageTableManager to
    /// find the RequestHandler associated with a range of pages.
    RequestProcessor( PageTableManager* pageTableManager )
        : m_pageTableManager( pageTableManager )
    {
    }

    /// Start processing requests using the specified number of threads.  If the number of specified
    /// threads is zero, std::thread::hardware_concurrency is used.
    void start( unsigned int maxThreads );

    /// Stop processing requests, terminating threads.
    void stop();

    /// Add a batch of page requests from the specified device to the request queue.
    void addRequests( unsigned int deviceIndex, CUstream stream, const unsigned int* pageIds, unsigned int numPageIds, std::shared_ptr<TicketImpl> ticket )
    {
        m_requests.push( deviceIndex, stream, pageIds, numPageIds, ticket );
    }

  private:
    PageTableManager*        m_pageTableManager;
    RequestQueue             m_requests;
    std::vector<std::thread> m_threads;

    // Per-thread worker function.
    void worker();
};

}  // namespace demandLoading
