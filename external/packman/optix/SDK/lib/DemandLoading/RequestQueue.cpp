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

#include "RequestQueue.h"
#include "Util/Exception.h"

#include <algorithm>

namespace demandLoading {

void RequestQueue::shutDown()
{
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_isShutDown = true;
    }
    m_requestAvailable.notify_all();
}

bool RequestQueue::popOrWait( PageRequest* requestPtr )
{
    // Wait until the queue is non-empty or destroyed.
    std::unique_lock<std::mutex> lock( m_mutex );
    m_requestAvailable.wait( lock, [this] { return !m_requests.empty() || m_isShutDown; } );

    if( m_isShutDown )
        return false;

    PageRequest request = m_requests.front();
    m_requests.pop_front();

    *requestPtr = request;
    return true;
}

void RequestQueue::push( unsigned int deviceIndex, CUstream stream, const unsigned int* pageIds, unsigned int numPageIds, std::shared_ptr<TicketImpl> ticket )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // Don't push requests if the queue is shut down.
    if( m_isShutDown )
        numPageIds = 0;

    // Update the ticket, now that the number of tasks is known.
    ticket->update( numPageIds );

    if( numPageIds == 0 )
        return;

    for( unsigned int i = 0; i < numPageIds; ++i )
    {
        m_requests.emplace_back( pageIds[i], deviceIndex, stream, ticket );
    }

    // Notify any threads in popOrWait().
    m_requestAvailable.notify_all();
}

}  // namespace demandLoading
