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

#include "Memory/TilePool.h"

#include "Memory/Buffers.h"
#include "Util/Exception.h"
#include "Util/Math.h"

#include <algorithm>

namespace demandLoading {

TilePool::TilePool( unsigned int deviceIndex, size_t maxTexMem )
    : m_deviceIndex( deviceIndex )
    , m_maxTexMem( maxTexMem )
{
    // Set current device.
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    // Use the recommended allocation granularity as the arena size.  Typically this gives 32 tiles per arena.
    CUmemAllocationProp prop{};
    prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( m_deviceIndex )};
    prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
    DEMAND_CUDA_CHECK( cuMemGetAllocationGranularity( &m_arenaSize, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ) );
    DEMAND_ASSERT( m_arenaSize >= sizeof( TileBuffer ) );

    unsigned int totalTiles = static_cast<int>( maxTexMem / sizeof( TileBuffer ) );
    m_desiredFreeTiles      = static_cast<unsigned int>( totalTiles * DESIRED_FREE_TILE_FRACTION );
}

TilePool::~TilePool()
{
    // Set current device.
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    for( CUmemGenericAllocationHandle arena : m_arenas )
    {
        DEMAND_CUDA_CHECK( cuMemRelease( arena ) );
    }
}

TileBlockDesc TilePool::allocate( size_t numBytes )
{
    std::unique_lock<std::mutex> lock( m_mutex );

    unsigned int requestedTiles = static_cast<unsigned int>( ( numBytes + sizeof( TileBuffer ) - 1 ) / sizeof( TileBuffer ) );
    int          idx;

    // Try to find a free block with enough space.
    // Use end of list for single tile request.
    // Search from beginning for multiple tile request
    if( requestedTiles == 1 && m_freeTileBlocks.size() > 0 )
    {
        idx = static_cast<int>( m_freeTileBlocks.size() - 1 );
    }
    else
    {
        for( idx = 0; idx < static_cast<int>( m_freeTileBlocks.size() ); ++idx )
            if( m_freeTileBlocks[idx].numTiles >= requestedTiles )
                break;
    }

    // Create a new arena if necessary, and put it on the free list.
    if( idx >= static_cast<int>( m_freeTileBlocks.size() ) )
    {
        // Only expand past the recommended max memory limit for multi-tile requests.
        if( m_arenas.size() * m_arenaSize >= m_maxTexMem && requestedTiles == 1 )
            return TileBlockDesc{};

        m_arenas.push_back( createArena() );
        const unsigned int   arenaId       = static_cast<unsigned int>( m_arenas.size() - 1 );
        const unsigned short tilesPerArena = static_cast<unsigned short>( m_arenaSize / sizeof( TileBuffer ) );
        m_freeTileBlocks.push_front( TileBlockDesc{arenaId, 0, tilesPerArena} );
        idx = 0;
    }

    // Reduce size of block in free list
    unsigned short tileId = m_freeTileBlocks[idx].tileId;
    m_freeTileBlocks[idx].tileId += requestedTiles;
    m_freeTileBlocks[idx].numTiles -= requestedTiles;

    // Make the tile block to return
    TileBlockDesc tileBlock = TileBlockDesc{m_freeTileBlocks[idx].arenaId, tileId, static_cast<unsigned short>( requestedTiles )};

    // Remove the block from the free list if empty
    if( m_freeTileBlocks[idx].numTiles == 0 )
        m_freeTileBlocks.erase( m_freeTileBlocks.begin() + idx );

    return tileBlock;
}

void TilePool::getHandle( TileBlockDesc tileBlock, CUmemGenericAllocationHandle* handle, size_t* offset )
{
    DEMAND_ASSERT( tileBlock.arenaId < static_cast<unsigned int>( m_arenas.size() ) );

    std::unique_lock<std::mutex> lock( m_mutex );
    *handle = m_arenas[tileBlock.arenaId];
    *offset = tileBlock.tileId * sizeof( TileBuffer );
}

void TilePool::freeBlock( TileBlockDesc block )
{
    DEMAND_ASSERT( block.arenaId < static_cast<unsigned int>( m_arenas.size() ) );

    std::unique_lock<std::mutex> lock( m_mutex );

    // TODO: Add the capability to coalesce free blocks
    // TODO: Second chance algorithm (don't reload requested block if already loaded)
    // TODO: Only put the block in the free list after it is truly safe to do so.
    // Insert free blocks at position 1 to so they will not be reused immediately
    unsigned int idx = ( m_freeTileBlocks.size() > 0 ) ? 1 : 0;
    m_freeTileBlocks.insert( m_freeTileBlocks.begin() + idx, block );
}

size_t TilePool::getTotalDeviceMemory() const
{
    std::unique_lock<std::mutex> lock( m_mutex );
    return m_arenas.size() * m_arenaSize;
}

void TilePool::incPendingFreeTiles( int pendingTiles )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    m_pendingFreeTiles += pendingTiles;
}

unsigned int TilePool::getDesiredTilesToFree() const
{
    std::unique_lock<std::mutex> lock( m_mutex );

    // TODO: Revisit this heuristic for the desired number of tiles.

    // Add up the (approximate) number of free tiles in the freeTileBlocks list
    unsigned int currFreeTiles = static_cast<unsigned int>( m_freeTileBlocks.size() );
    if( currFreeTiles > 0 )
        currFreeTiles += m_freeTileBlocks[0].numTiles - 1;

    // Add tiles for arenas that have not been allocated
    if( m_maxTexMem / m_arenaSize >= m_arenas.size() )
    {
        unsigned int numAvailableArenas = static_cast<unsigned int>( m_maxTexMem / m_arenaSize - m_arenas.size() );
        unsigned int tilesPerArena      = static_cast<unsigned int>( m_arenaSize / sizeof( TileBuffer ) );
        currFreeTiles += tilesPerArena * numAvailableArenas;
    }

    // If the current free tiles plus pending free tiles is more than half the desired free tiles,
    // return 0 (we don't want any more right now).
    if( currFreeTiles + m_pendingFreeTiles > m_desiredFreeTiles / 2 )
        return 0;

    // Return the number of tiles needed to get m_desiredFreeTiles
    return m_desiredFreeTiles - ( currFreeTiles + m_pendingFreeTiles );
}

CUmemGenericAllocationHandle TilePool::createArena() const
{
    DEMAND_CUDA_CHECK( cudaSetDevice( m_deviceIndex ) );

    CUmemAllocationProp prop{};
    prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location         = {CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>( m_deviceIndex )};
    prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;
    CUmemGenericAllocationHandle arena;
    DEMAND_CUDA_CHECK( cuMemCreate( &arena, m_arenaSize, &prop, 0 ) );
    return arena;
}

}  // namespace demandLoading
