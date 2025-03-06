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

#include "Textures/SamplerRequestHandler.h"
#include "DemandLoaderImpl.h"
#include "PagingSystem.h"
#include "Textures/DemandTextureImpl.h"
#include "Util/NVTXProfiling.h"

#include <DemandLoading/Paging.h>  // for NON_EVICTABLE_LRU_VAL

namespace demandLoading {

void SamplerRequestHandler::fillRequest( unsigned int deviceIndex, CUstream stream, unsigned int pageId )
{
    SCOPED_NVTX_RANGE_FUNCTION_NAME();

    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to create the same sampler.
    unsigned int index = pageId - m_startPage;
    MutexArrayLock lock( m_mutex.get(), index);

    // Do nothing if the request has already been filled.
    if( m_loader->getPagingSystem( deviceIndex )->isResident( pageId ) )
        return;

    // The samplers were the first resource that were assigned page table entries (via
    // PageTableManager), so the samplers occupy the first N page table entries.  The device code in
    // Texture2D.h relies on this invariant, but this code does not.
    unsigned int       samplerId = pageId - m_startPage;
    DemandTextureImpl* texture   = m_loader->getTexture( samplerId );

    // Initialize the texture, reading image info from file header on the first call and
    // creating a per-device CUDA texture object.
    const bool ok = texture->init( deviceIndex );
    DEMAND_ASSERT_MSG( ok, "ImageReader::init() failed" );

    // Allocate sampler buffer in pinned memory.
    PinnedItemPool<TextureSampler>* pinnedSamplerPool = m_loader->getPinnedMemoryManager()->getPinnedSamplerPool();
    TextureSampler*                 pinnedSampler     = pinnedSamplerPool->allocate();

    // Copy the canonical sampler from the DemandTexture and set its CUDA texture object, which differs per device.
    *pinnedSampler         = texture->getSampler();
    pinnedSampler->texture = texture->getTextureObject( deviceIndex );

    // Allocate device memory for sampler.
    SamplerPool*    samplerPool = m_loader->getDeviceMemoryManager( deviceIndex )->getSamplerPool();
    TextureSampler* devSampler  = samplerPool->allocate();

    // Copy sampler to device memory.
    DEMAND_CUDA_CHECK( cudaMemcpyAsync( devSampler, pinnedSampler, sizeof( TextureSampler ), cudaMemcpyHostToDevice, stream ) );

    // Free the pinned memory buffer.  This doesn't immediately reclaim it: an event is recorded on
    // the stream, and the buffer isn't reused until all preceding operations are complete,
    // including the asynchronous memcpy issued by fillTile().
    pinnedSamplerPool->free( pinnedSampler, deviceIndex, stream );

    // Push mapping for sampler to update page table.
    m_loader->getPagingSystem( deviceIndex )->addMapping( pageId, NON_EVICTABLE_LRU_VAL, reinterpret_cast<unsigned long long>( devSampler ) );
}

}  // namespace demandLoading
