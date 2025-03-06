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

#include "Util/Exception.h"

#include <DemandLoading/Ticket.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace demandLoading {

/// A TicketImpl tracks the progress of a number of tasks.
class TicketImpl : public Ticket
{
  public:
    /// Initially the number of tasks is unknown (represented by -1).
    TicketImpl() {}

    /// The ticket is updated when the number of tasks are known.
    void update( unsigned int numTasks )
    {
        {
            std::unique_lock<std::mutex> lock( m_mutex );
            m_numTasksTotal     = numTasks;
            m_numTasksRemaining = numTasks;
        }
        // If there are no tasks, notify any threads waiting on the condition variable.
        if( numTasks == 0 )
            m_isDone.notify_all();
    }

    /// Get the total number of tasks tracked by this ticket.  Returns -1 if the number of tasks is
    /// unknown, which indicates that task processing has not yet started.
    int numTasksTotal() const override { return m_numTasksTotal; }

    /// Get the number of tasks remaining.  Returns -1 if the number of tasks is unknown, which
    /// indicates that task processing has not yet started.
    int numTasksRemaining() const override { return m_numTasksRemaining.load(); }

    /// Wait for the tasks to be done.
    void wait() override
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_isDone.wait( lock, [this] { return m_numTasksRemaining == 0; } );
    }

    /// Decrement the number of tasks remaining, notifying any waiting threads
    /// when all the tasks are done.
    void notify( unsigned int tasksDone = 1 )
    {
        // Atomically decrement the number of tasks remaining.
        DEMAND_ASSERT( m_numTasksRemaining.load() > 0 );
        m_numTasksRemaining -= tasksDone;

        // If there are no tasks remaining, notify any threads waiting on the condition variable.
        // It's not necessary to acquire the mutex.  Redundant notifications are OK.
        if( m_numTasksRemaining.load() == 0 )
            m_isDone.notify_all();
    }

    /// Not copyable.
    TicketImpl( const TicketImpl& ) = delete;

    /// Not assignable.
    TicketImpl& operator=( const TicketImpl& ) = delete;

  private:
    int                     m_numTasksTotal{-1};
    std::atomic<int>        m_numTasksRemaining{-1};
    mutable std::mutex      m_mutex;
    std::condition_variable m_isDone;
};

}  // namespace demandLoading
