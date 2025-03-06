//
//  Copyright (c) 2021, NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include "Util/Exception.h"

#include <DemandLoading/TextureSampler.h>

#include <cuda_runtime.h>

#include <mutex>
#include <vector>

namespace demandLoading {

// Owns device memory for texture samplers.
class SamplerPool
{
  public:
    /// Construct sampler pool for the specified device.
    explicit SamplerPool( unsigned int deviceIndex )
        : m_deviceIndex( deviceIndex )
    {
    }

    /// Destroy the sampler pool, reclaiming its resources.
    ~SamplerPool()
    {
        DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );
        for( uint8_t* arena : m_arenas )
        {
            DEMAND_CUDA_CHECK_NOTHROW( cudaFree( arena ) );
        }
    }

    /// Not copyable
    SamplerPool( const SamplerPool& ) = delete;

    /// Not assignable
    SamplerPool& operator=( const SamplerPool& ) = delete;

    /// Allocate device memory for texture sampler.
    TextureSampler* allocate()
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        // We allocate a 64KB arena at a time to reduce allocation frequency.
        if( m_arenas.empty() || m_offset + sizeof( TextureSampler ) > m_arenaSize )
        {
            // Set current device.
            DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

            uint8_t* arena;
            DEMAND_CUDA_CHECK( cudaMalloc( &arena, m_arenaSize * sizeof( TextureSampler ) ) );
            m_arenas.push_back( arena );
            m_offset = 0;
        }

        TextureSampler* sampler = reinterpret_cast<TextureSampler*>( m_arenas.back() + m_offset );
        m_offset += sizeof( TextureSampler );
        return sampler;
    }

    size_t getTotalDeviceMemory() const { return m_arenas.size() * m_arenaSize; }

  private:
    unsigned int          m_deviceIndex;
    const unsigned int    m_arenaSize = 64 * 1024;
    std::vector<uint8_t*> m_arenas;
    unsigned int          m_offset = 0;  // offset of available storage in current arena
    std::mutex            m_mutex;
};

}  // namespace demandLoading
