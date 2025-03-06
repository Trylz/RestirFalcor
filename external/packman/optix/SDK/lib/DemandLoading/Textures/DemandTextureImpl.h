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

#include "Textures/SparseTexture.h"
#include "Textures/TextureRequestHandler.h"
#include "Util/Exception.h"

#include <DemandLoading/DemandTexture.h>
#include <DemandLoading/ImageReader.h>
#include <DemandLoading/TextureDescriptor.h>
#include <DemandLoading/TextureSampler.h>

#include <cuda.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace demandLoading {

class DemandLoaderImpl;
class ImageReader;
class TilePool;

/// Demand-loaded textures are created by the DemandLoader.
class DemandTextureImpl : public DemandTexture
{
  public:
    /// Default constructor.
    DemandTextureImpl() = default;

    /// Construct demand loaded texture with the specified id (which is used as an index into the
    /// device-side sampler array) the the given descriptor (which specifies the wrap mode, filter
    /// mode, etc.).  The given image reader is retained and used by subsequent readTile() calls.
    DemandTextureImpl( unsigned int                 id,
                       unsigned int                 maxNumDevices,
                       const TextureDescriptor&     descriptor,
                       std::shared_ptr<ImageReader> image,
                       DemandLoaderImpl*            loader );

    /// Default destructor.
    ~DemandTextureImpl() override = default;

    /// Get the texture id, which is used as an index into the device-side sampler array.
    unsigned int getId() const override;

    /// Initialize the texture on the specified device.  When first called, this method opens the
    /// image reader that was provided to the constructor.  Returns false on error.
    bool init( unsigned int deviceIndex );

    /// Get the image info.  Valid only after the image has been initialized (e.g. opened).
    const TextureInfo& getInfo() const;

    /// Get the canonical sampler for this texture, excluding the CUDA texture object, which differs
    /// for each device (see getTextureObject).
    const TextureSampler& getSampler() const;

    /// Get the CUDA texture object for the specified device.
    CUtexObject getTextureObject( unsigned int deviceIndex ) const;

    /// Get the texture descriptor
    const TextureDescriptor& getDescriptor() const;

    /// Get the dimensions of the specified miplevel.
    uint2 getMipLevelDims( unsigned int mipLevel ) const;

    /// Get tile width.
    unsigned int getTileWidth() const;

    /// Get tile height.
    unsigned int getTileHeight() const;

    /// Get the number of miplevels.
    unsigned int IsMipmapped() const { return getInfo().numMipLevels > getMipTailFirstLevel(); }

    /// Get the first miplevel in the mip tail.
    unsigned int getMipTailFirstLevel() const;

    /// Get the request handler for this texture.
    TextureRequestHandler* getRequestHandler() { return m_requestHandler.get(); }

    /// Get the ImageReader (for gathering statistics).
    ImageReader* getImageReader() const { return m_image.get(); }

    /// Read the specified tile into the given buffer.
    bool readTile( unsigned int mipLevel, unsigned int tileX, unsigned int tileY, char* tileBuffer, size_t tileBufferSize ) const;

    /// Fill the device tile backing storage for a texture tile and with the given data.
    void fillTile( unsigned int                 deviceIndex,
                   CUstream                     stream,
                   unsigned int                 mipLevel,
                   unsigned int                 tileX,
                   unsigned int                 tileY,
                   const char*                  tileData,
                   size_t                       tileSize,
                   CUmemGenericAllocationHandle handle,
                   size_t                       offset ) const;

    /// Unmap backing storage for a tile
    void unmapTile( unsigned int deviceIndex, CUstream stream, unsigned int mipLevel, unsigned int tileX, unsigned int tileY ) const;

    /// Read all the levels in the mip tail into the given buffer, resizing it if necessary.
    bool readMipTail( char* buffer, size_t bufferSize ) const;

    /// Fill the device backing storage for the mip tail with the given data.
    void fillMipTail( unsigned int                 deviceIndex,
                      CUstream                     stream,
                      const char*                  mipTailData,
                      size_t                       mipTailSize,
                      CUmemGenericAllocationHandle handle,
                      size_t                       offset ) const;

    /// Unmap backing storage for the mip tail
    void unmapMipTail( unsigned int deviceIndex, CUstream stream ) const;

    /// DemandTextureImpl cannot be copied because the PageTableManager holds a pointer to the
    /// RequestHandler it provides.
    DemandTextureImpl( const DemandTextureImpl& ) = delete;

    /// Not assignable.
    DemandTextureImpl& operator=( const DemandTextureImpl& ) = delete;
    
  private:
    // A mutex guards against concurrent initialization, which can arise when the sampler
    // is requested on multiple devices.  Tiles can be filled concurrently, along with the mip tail.
    std::mutex m_initMutex;

    // The texture identifier is used as an index into the device-side sampler array.
    const unsigned int m_id = 0;

    // The texture descriptor specifies wrap and filtering modes, etc.  Invariant after init(), and not valid before then.
    TextureDescriptor m_descriptor{};

    // The image provides a read() method that fills requested miplevels.
    const std::shared_ptr<ImageReader> m_image;

    // The DemandLoader provides access to the PageTableManager, etc.
    DemandLoaderImpl* const m_loader;

    // The image is lazily opened.  Invariant after init().
    bool m_isInitialized = false;

    // Image info, including dimensions and format.  Invariant after init(), and not valid before then.
    TextureInfo        m_info{};
    TextureSampler     m_sampler{};
    unsigned int       m_tileWidth         = 0;
    unsigned int       m_tileHeight        = 0;
    unsigned int       m_mipTailFirstLevel = 0;
    size_t             m_mipTailSize       = 0;
    std::vector<uint2> m_mipLevelDims;

    // Sparse textures (one per device).  This vector does not grow after construction, which is
    // important for thread safety.
    std::vector<SparseTexture> m_textures;

    // Request handler.
    std::unique_ptr<TextureRequestHandler> m_requestHandler;

    void         initSampler();
    unsigned int getNumTilesInLevel( unsigned int mipLevel ) const;
};

}  // namespace demandLoading
