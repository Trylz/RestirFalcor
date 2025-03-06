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

#include <cuda.h>

#include <deque>
#include <mutex>
#include <vector>

namespace demandLoading {

// Fraction of total tiles to keep available
const float DESIRED_FREE_TILE_FRACTION = 0.1f;

class TileBlockDesc
{
  public:
    bool isValid() const { return numTiles != 0; }

    unsigned long long getData() { return rawData; }

    TileBlockDesc( unsigned long long data )
        : rawData( data )
    {
    }

    TileBlockDesc( unsigned int arenaId_, unsigned short tileId_, unsigned short numTiles_ )
        : arenaId( arenaId_ )
        , tileId( tileId_ )
        , numTiles( numTiles_ )
    {
    }

  private:
    friend class TilePool;

    TileBlockDesc() = default;

    union
    {
        struct
        {
            unsigned int   arenaId;
            unsigned short tileId;
            unsigned short numTiles;
        };
        unsigned long long rawData = 0;
    };
};

class TilePool
{
  public:
    /// Construct tile pool for the specified device.
    explicit TilePool( unsigned int deviceIndex, size_t maxTileMemUsage );

    /// Destroy the tile pool, reclaiming its resources.
    ~TilePool();

    /// Not copyable
    TilePool( const TilePool& ) = delete;

    /// Not assignable.
    TilePool& operator=( const TilePool& ) = delete;

    /// Allocate a number of bytes and return a tile block descriptor (as uint64). This can be
    /// used to get the memory handle for the block, or free the block later. Returns 0 on failure.
    TileBlockDesc allocate( size_t numBytes );

    /// Get memory handle and offset from a tile block descriptor.
    void getHandle( TileBlockDesc block, CUmemGenericAllocationHandle* handle, size_t* offset );

    /// free a block of tiles (reclaim it for future use)
    void freeBlock( TileBlockDesc block );

    /// Returns the amount of device memory allocated across all arenas.
    size_t getTotalDeviceMemory() const;

    /// Increment (or decrement) the number of pending free tiles
    void incPendingFreeTiles( int pendingTiles );

    /// Return the number of tiles that the pool wants free to maintain the size of the free list.
    unsigned int getDesiredTilesToFree() const;

  private:
    unsigned int                              m_deviceIndex{};
    std::vector<CUmemGenericAllocationHandle> m_arenas;
    size_t                                    m_arenaSize{};
    mutable std::mutex                        m_mutex;
    std::deque<TileBlockDesc>                 m_freeTileBlocks;
    size_t                                    m_maxTexMem{};
    unsigned int                              m_desiredFreeTiles{};
    unsigned int                              m_pendingFreeTiles{};

    CUmemGenericAllocationHandle createArena() const;
};

}  // namespace demandLoading
