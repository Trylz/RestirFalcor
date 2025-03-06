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

#include <cstddef>

namespace demandLoading {

/// Demand loading configuration options.  \see createDemandLoader
struct Options
{
    unsigned int numPages            = 1 << 26;     ///< max virtual pages (approx. 4 TB of texture tiles)
    unsigned int numPageTableEntries = 256 * 1024;  ///< page table entries are needed for samplers, but not tiles.

    unsigned int maxRequestedPages   = 8192;  ///< max requests to pull from device
    unsigned int maxFilledPages      = 8192;  ///< num slots to push mappings back to device
    unsigned int maxStalePages       = 8192;  ///< max stale pages to pull from device
    unsigned int maxEvictablePages   = 8192;  ///< max evictable pages to pull from device
    unsigned int maxInvalidatedPages = 8192;  ///< max slots to push invalidated pages back to device

    bool   useLruTable        = false;              ///< enable eviction (not yet fully implemented)
    size_t maxTexMemPerDevice = 0;                  ///< max texture data to be allocated per device (0 is unlimited)
    size_t maxPinnedMemory    = 640 * 1024 * 1024;  ///< maximum pinned memory.

    unsigned int maxThreads = 0;        ///< max number of threads to use when processing requests;
                                        ///< zero means use std::thread::hardware_concurrency.
    unsigned int maxActiveStreams = 4;  ///< number of active streams across all devices.
};
}
