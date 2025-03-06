#
#  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

if( CUDA_VERSION VERSION_LESS "11.1" )
  message( WARNING "Skipping DemandLoading library and samples, which require CUDA 11.1 or later." )
  return()
endif()

# CUDA compilation doesn't pick up the DemandLoading include path target property.
include_directories( include )

cuda_wrap_srcs( PagingSystemKernels OBJ generated_files PagingSystemKernels.cu "" OPTIONS "" )

add_library( DemandLoading STATIC
  DemandLoaderImpl.cpp
  DemandLoaderImpl.h
  DeviceContextImpl.cpp
  DeviceContextImpl.h
  PageMappingsContext.h
  PageTableManager.h
  PagingSystem.cpp
  PagingSystem.h
  PagingSystemKernels.h
  RequestContext.h
  RequestHandler.h
  RequestProcessor.cpp
  RequestProcessor.h
  RequestQueue.cpp
  RequestQueue.h
  ResourceRequestHandler.cpp
  ResourceRequestHandler.h
  TicketImpl.h
  include/DemandLoading/CheckerBoardImage.h
  include/DemandLoading/DemandLoader.h
  include/DemandLoading/DemandTexture.h
  include/DemandLoading/DeviceContext.h
  include/DemandLoading/ImageReader.h
  include/DemandLoading/MultiCheckerImage.h
  include/DemandLoading/Options.h
  include/DemandLoading/Resource.h
  include/DemandLoading/Statistics.h
  include/DemandLoading/Texture2D.h
  include/DemandLoading/TextureDescriptor.h
  include/DemandLoading/TextureInfo.h
  include/DemandLoading/TextureSampler.h
  include/DemandLoading/Ticket.h
  include/DemandLoading/TileIndexing.h
  Memory/Allocators.h
  Memory/AsyncItemPool.h
  Memory/Buffers.h
  Memory/BulkMemory.h
  Memory/BulkPinnedItemPool.h
  Memory/DeviceContextPool.cpp
  Memory/DeviceContextPool.h
  Memory/DeviceMemoryManager.h
  Memory/EventPool.h
  Memory/FixedPool.h
  Memory/PinnedItemPool.h
  Memory/PinnedMemoryManager.h
  Memory/PinnedRequestContextPool.h
  Memory/SamplerPool.h
  Memory/TilePool.cpp
  Memory/TilePool.h
  Textures/CheckerBoardImage.cpp
  Textures/DemandTextureImpl.cpp
  Textures/DemandTextureImpl.h
  Textures/ImageReader.cpp
  Textures/SamplerRequestHandler.cpp
  Textures/SamplerRequestHandler.h
  Textures/SparseTexture.cpp
  Textures/SparseTexture.h
  Textures/TextureInfo.cpp
  Textures/TextureRequestHandler.cpp
  Textures/TextureRequestHandler.h
  Util/DeviceSet.cpp
  Util/DeviceSet.h
  Util/Exception.h
  Util/Math.h
  Util/MutexArray.h
  Util/NVTXProfiling.h
  Util/Stopwatch.h
  # EXRReader.cpp is added below if OpenEXR is available.
  ${generated_files}
  )

source_group( "Header Files\\Implementation" FILES
  DemandLoaderImpl.h
  DeviceContextImpl.h
  PageMappingsContext.h
  PageTableManager.h
  PagingSystem.h
  PagingSystemKernels.h
  RequestContext.h
  RequestHandler.h
  RequestProcessor.h
  RequestQueue.h
  ResourceRequestHandler.h
  TicketImpl.h
  Memory/Allocators.h
  Memory/AsyncItemPool.h
  Memory/Buffers.h
  Memory/BulkMemory.h
  Memory/BulkPinnedItemPool.h
  Memory/DeviceContextPool.h
  Memory/DeviceMemoryManager.h
  Memory/EventPool.h
  Memory/FixedPool.h
  Memory/PinnedItemPool.h
  Memory/PinnedMemoryManager.h
  Memory/PinnedRequestContextPool.h
  Memory/SamplerPool.h
  Memory/TilePool.h
  Textures/DemandTextureImpl.h
  Textures/SamplerRequestHandler.h
  Textures/SparseTexture.h
  Textures/TextureRequestHandler.h
  Util/DeviceSet.h
  Util/Exception.h
  Util/Math.h
  Util/MutexArray.h
  Util/NVTXProfiling.h
  Util/Stopwatch.h
  )

# Not sure why glad include directory must be specified here.  It should come with the library dependency below,
# but that doesn't seem to be working on the Mac.
target_include_directories( DemandLoading PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  include
  ../../support
  )

target_link_libraries( DemandLoading PUBLIC
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )

find_package( OpenEXR )
if( NOT OpenEXR_FOUND )
  message( WARNING "OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in optixDemandTextureAdvanced." )
endif()

find_package( ZlibStatic )
if ( NOT ZlibStatic_FOUND )
  message( WARNING "Zlib not found (see ZlibStatic_ROOT).  Will use procedural texture in optixDemandTextureAdvanced." )
endif()

if( OpenEXR_FOUND AND ZlibStatic_FOUND )
  # Build our EXRImage class if OpenEXR is available, adding the necessary include path and libraries.
  target_sources( DemandLoading PRIVATE
    Textures/EXRReader.cpp
    include/DemandLoading/EXRReader.h )
  target_compile_definitions( DemandLoading PUBLIC OPTIX_SAMPLE_USE_OPEN_EXR )
  target_link_libraries( DemandLoading PUBLIC OpenEXR::IlmImf Zlib::Static )
endif()

# NVTX Profiling
option( DEMAND_LOADING_USE_NVTX "Enable NVTX profiling" OFF )

find_package( NVTX )
if( DEMAND_LOADING_USE_NVTX AND NVTX_FOUND )
  message( "-- Found NVTX: Enabling NVTX profiling in the DemandLoading library" )
  include_directories( ${NVTX_INCLUDE} )
  target_compile_definitions( DemandLoading PUBLIC ENABLE_NVTX_PROFILING )
else()
  message( "-- Not Found NVTX: Disabling NVTX profiling in the DemandLoading library" )
endif()

set_property(TARGET DemandLoading PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

set( DEMAND_LOADING_BUILD_DOCS TRUE CACHE BOOL "Build Demand Loading library docs (requires Doxygen)" )
set( DEMAND_LOADING_EXTRACT_ALL YES CACHE STRING "Extract all symbols for documentation (suppresses undocumented symbol warnings)" )
set( DEMAND_LOADING_WARN_IF_UNDOCUMENTED YES CACHE STRING "Warn about undocumented symbols when building documentation" )

if ( DEMAND_LOADING_BUILD_DOCS )
  add_subdirectory( docs/API )
endif()
