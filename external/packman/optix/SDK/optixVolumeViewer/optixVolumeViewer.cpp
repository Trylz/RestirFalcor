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

#include "optixVolumeViewer.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "nanovdb/util/IO.h" // this is required to read (and write) NanoVDB files on the host

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>

#include <GLFW/glfw3.h>

#include <optix_stack_size.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>



// ----------------------------------------------------------------------------
// This sample demonstrates the basic mechanism of visualizing a NanoVDB
// volume using Optix.
//
// The NanoVDB volume is represented in the Optix acceleration structure by its
// overall AABB. Its intersection program advances t-max to the ray's entry
// into the volume or, if current t-max falls inside of the volume, leaves it
// unchanged. In addition to advancing t-max, the intersection program also
// reports the t-value of the ray exit location from the volume in payload 0.
//
// The volume's closest hit shader fires a continuation ray to
// a) obtain radiance as the ray enters the volume and
// b) a depth value with the continution ray's intersection.
// Then the transmittance through the volume is computed using a 3D DDA
// algorithm. For this the volume's density is integrated for the ray segment
// from t-max to either the continuation-ray depth or the exit-t, whichever is
// is closer. This makes it possible for solids to essentially displace the
// volume. Result radiance is computed by multiplying the incoming radiance
// with the transmittance.
//
// The (simplistic) use of a single AABB to represent a potentially large
// (relative to scene extent), sparse volume leads to integration happening
// along ray segments that may traverse large stretches of "empty volume".
// To leverage hardware accelerated ray-object intersection and "skip over"
// areas where the volume function is zero one may represent the volume in
// Optix's BVH by a series of AABBs of inner nodes of the NanoVDB tree
// structure. This was done in the viewer example that ships with the OpenVDB/
// NanoVDB distribution (https://github.com/AcademySoftwareFoundation/openvdb).
// For the level-set rendering in that example, bounding-boxes for all leaf-
// nodes were added. 
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
//
// Global settings
//
// ----------------------------------------------------------------------------

const float plane_scale_factor = 2.5f;
const float cube_scale_factor = 0.25f;

// ----------------------------------------------------------------------------
//
// Global state
//
// ----------------------------------------------------------------------------

OptixDeviceContext g_context;

Volume g_volume = {};
Plane  g_plane  = {};
Cube   g_cube = {};

VolumeAccel g_volume_accel = {};
PlaneAccel  g_plane_accel  = {};
CubeAccel   g_cube_accel  = {};
IAS         ias          = {};
			
Params launch_params = {};

OptixModule module                     = 0;
ProgramGroups           program_groups = {};
OptixPipeline           pipeline       = 0;
OptixShaderBindingTable sbt            = {};
  

// ----------------------------------------------------------------------------
//
// Global state for GUI
//
// ----------------------------------------------------------------------------
bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t mouse_button = -1;

int32_t width  = 768;
int32_t height = 768;


// device-pointers to state that can be interactively changed, i.e.
// moving the cube via arrow-keys, moving the plane with 'u' and 'd',
// and changing the volume's opacity
std::vector<CUdeviceptr> dp_opacities; // the volume's opacity in the SBT (one for each ray-type)
CUdeviceptr              dp_plane_y;   // plane's y-coordinate in the device-instance array
CUdeviceptr              dp_cube_x;    // cube's x-coordinate in the device-instance array
CUdeviceptr              dp_cube_z;    // cube's z-coordinate in the device-instance array

// boolean to track if object transformations were changed (e.g. plane_y, cube_x/y)
// when true the IAS must be rebuilt for the next frame
bool ias_changed = false;


// -----------------------------------------------------------------------------
// state controls
// -----------------------------------------------------------------------------

// retrieve a single float from a device pointer location
float peek( CUdeviceptr location )
{
    float value;
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( &value ),
        reinterpret_cast<void*>( location ),
        sizeof( float ),
        cudaMemcpyDeviceToHost
    ) );
    return value;
}

// put a single float into a device pointer location
void poke( CUdeviceptr location, float value )
{
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( location ),
        reinterpret_cast<void*>( &value ),
        sizeof( float ),
        cudaMemcpyHostToDevice
    ) );
}

inline void increaseOpacity()
{
    for( CUdeviceptr dp_opacity : dp_opacities )
    {
        float opacity = peek(dp_opacity);
        opacity *= 1.1f;
        if( opacity > 1.0f ) opacity = 1.0f;
        poke(dp_opacity, opacity);
    }
    camera_changed = true;
}

inline void decreaseOpacity()
{
    for( CUdeviceptr dp_opacity : dp_opacities )
    {
        float opacity = peek(dp_opacity);
        opacity /= 1.1f;
        if( opacity > 1.0f ) opacity = 1.0f;
        poke( dp_opacity, opacity );
    }
    camera_changed = true;
}

inline void raisePlane() 
{
    float plane_y = peek( dp_plane_y );
    plane_y += 5.0f;
    poke( dp_plane_y, plane_y );
    ias_changed = true;
}

inline void lowerPlane()
{
    float plane_y = peek( dp_plane_y );
    plane_y -= 5.0f;
    poke( dp_plane_y, plane_y );
    ias_changed = true;
}

inline void zoomIn() 
{
    // decrease field-of-view angle by 5%
    camera.setFovY(camera.fovY() / 1.05f); 
    camera_changed = true;
}

inline void zoomOut()
{
    // increase field-of-view angle by 5%
    camera.setFovY(camera.fovY() * 1.05f); 
    camera_changed = true;
}

inline void incCubeX()
{
    float cube_x = peek( dp_cube_x );
    cube_x += 5.0f;
    poke( dp_cube_x, cube_x );
    ias_changed = true;
}

inline void decCubeX()
{
    float cube_x = peek( dp_cube_x );
    cube_x -= 5.0f;
    poke( dp_cube_x, cube_x );
    ias_changed = true;
}

inline void incCubeZ()
{
    float cube_z = peek( dp_cube_z );
    cube_z += 5.0f;
    poke( dp_cube_z, cube_z );
    ias_changed = true;
}

inline void decCubeZ()
{
    float cube_z = peek( dp_cube_z );
    cube_z -= 5.0f;
    poke( dp_cube_z, cube_z );
    ias_changed = true;
}

inline void toggleCubeVisibility()
{
    launch_params.params.solid_objects ^= CUBE_OBJECT;
}

inline void togglePlaneVisibility()
{
    launch_params.params.solid_objects ^= PLANE_OBJECT;
}

inline void toggleVolumeVisibility()
{
    launch_params.params.volume_object ^= VOLUME_OBJECT;
}

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    width   = res_x;
    height  = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
        else if (key == GLFW_KEY_V )
        {
            toggleVolumeVisibility();
        }
        else if (key == GLFW_KEY_P )
        {
            togglePlaneVisibility();
        }
        else if (key == GLFW_KEY_C )
        {
            toggleCubeVisibility();
        }
        else if( key == GLFW_KEY_KP_ADD )
        {
            increaseOpacity();
        }
        else if( key == GLFW_KEY_KP_SUBTRACT )
        {
            decreaseOpacity();
        }
        else if( key == GLFW_KEY_U )
        {
            raisePlane();
        }
        else if( key == GLFW_KEY_D )
        {
            lowerPlane();
        }
        else if( key == GLFW_KEY_I )
        {
            zoomIn();
        }
        else if( key == GLFW_KEY_O )
        {
            zoomOut();
        }
        else if( key == GLFW_KEY_RIGHT )
        {
            incCubeX();
        }
        else if( key == GLFW_KEY_LEFT )
        {
            decCubeX();
        }
        else if( key == GLFW_KEY_UP )
        {
            incCubeZ();
        }
        else if( key == GLFW_KEY_DOWN )
        {
            decCubeZ();
        }
        camera_changed = true;
    }
}

static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( ( int )yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

void printShortcuts()
{
    std::cerr << "Interactive controls:\n";
    std::cerr << "    - 'v', 'p', 'c':  toggle volume/plane/cube visibility\n";
    std::cerr << "    - Cursor keys:    move the cube around the horizontal plane\n";
    std::cerr << "    - Keypad +/-:     increase/decrease volume density\n";
    std::cerr << "    - 'u'/'d':        move the ground plane up or down\n";
    std::cerr << "    - 'i'/'o':        zoom in and out in 5% increments" << std::endl;
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage: " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --volume <volume.nvdb>      Specify volume to render (required)\n";
    std::cerr << "         --help | -h                 Print this usage message\n" << std::endl;
    printShortcuts();
    exit( 0 );
}


void initLaunchParams( Params& launch_params, const OptixTraversableHandle& handle, const sutil::Aabb& aabb ) 
{
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &launch_params.params.accum_buffer ),
                width*height*sizeof( float4 )
                ) );
    launch_params.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    launch_params.params.width          = width;
    launch_params.params.height         = height;
    launch_params.params.subframe_index = 0u;

    const float loffset = aabb.maxExtent();


    std::vector<Light> lights( 2 );
    lights[0].type            = Light::Type::POINT;
    lights[0].point.color     = {1.0f, 1.0f, 0.8f};
    lights[0].point.intensity = 5.0f;
    lights[0].point.position  = aabb.center() + make_float3( loffset );
    lights[0].point.falloff   = Light::Falloff::QUADRATIC;

    lights[1].type            = Light::Type::POINT;
    lights[1].point.color     = {0.8f, 0.8f, 1.0f};
    lights[1].point.intensity = 3.0f;
    lights[1].point.position  = aabb.center() + make_float3( -loffset, 1.f * loffset, -1.0f * loffset );
    lights[1].point.falloff   = Light::Falloff::QUADRATIC;


    launch_params.params.lights.count  = static_cast<uint32_t>( lights.size() );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &launch_params.params.lights.data ),
                lights.size() * sizeof( Light )
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( launch_params.params.lights.data ),
                lights.data(),
                lights.size() * sizeof( Light ),
                cudaMemcpyHostToDevice
                ) );

    launch_params.params.miss_color   = make_float3( 0.3f, 0.3f, 0.9f );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &launch_params.d_params ), sizeof( LaunchParams ) ) );

    launch_params.params.handle = handle;

    launch_params.params.solid_objects = PLANE_OBJECT | CUBE_OBJECT;
    launch_params.params.volume_object = VOLUME_OBJECT;
}

void cleanupLaunchParams( Params& launch_params )
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( launch_params.params.accum_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( launch_params.params.lights.data  ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( launch_params.d_params            ) ) );
}


void handleCameraUpdate( LaunchParams& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( width, height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( launch_params.params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &launch_params.params.accum_buffer ),
                width*height*sizeof( float4 )
                ) );
}


void handleIASUpdate()
{
    if( !ias_changed )
        return;
    ias_changed = false;
    updateIAS(ias, g_context);
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, LaunchParams& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleIASUpdate();
    handleCameraUpdate( params );
    handleResize( output_buffer );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    launch_params.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( launch_params.d_params ),
                &launch_params.params,
                sizeof( LaunchParams ),
                cudaMemcpyHostToDevice,
                0 // stream
                ) );

    OPTIX_CHECK( optixLaunch(
                pipeline,
                0,             // stream
                reinterpret_cast<CUdeviceptr>( launch_params.d_params ),
                sizeof( LaunchParams ),
                &sbt,
                width,  // launch width
                height, // launch height
                1       // launch depth
                ) );
    output_buffer.unmap();

    CUDA_SYNC_CHECK();
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


void createContext( OptixDeviceContext& context )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );
}


// ----------------------------------------------------------------------------
// Functions for manipulating Volume instances
// ----------------------------------------------------------------------------

void loadVolume( Volume& grid, const std::string& filename )
{
    // NanoVDB files are containers for NanoVDB Grids.
    // Each Grid represents a distinct volume, point-cloud, or level-set.
    // For the purpose of this sample only the first grid is loaded, additional
    // grids are ignored.
    auto list = nanovdb::io::readGridMetaData( filename );
    std::cerr << "Opened file " << filename << std::endl;
    std::cerr << "    grids:" << std::endl;
    for (auto& m : list) {
        std::cerr << "        " << m.gridName << std::endl;
    }
    assert( list.size() > 0 );
    // load the first grid in the file 
    createGrid( grid, filename, list[0].gridName );
}

 void createGrid( Volume& grid, std::string filename, std::string gridname )
{
    nanovdb::GridHandle<> gridHdl;

	if( gridname.length() > 0 )
		gridHdl = nanovdb::io::readGrid<>( filename, gridname );
	else
		gridHdl = nanovdb::io::readGrid<>( filename );

    if( !gridHdl ) 
    {
        std::stringstream ss;
        ss << "Unable to read " << gridname << " from " << filename;
        throw std::runtime_error( ss.str() );
    }

    // NanoVDB Grids can represent several kinds of 3D data, but this sample is
    // only concerned with volumetric data.
    auto* meta = gridHdl.gridMetaData();
    if( meta->isPointData() )
        throw std::runtime_error("NanoVDB Point Data cannot be handled by VolumeViewer");
    if( meta->isLevelSet() )
        throw std::runtime_error("NanoVDB Level Sets cannot be handled by VolumeViewer");

    // NanoVDB files represent the sparse data-structure as flat arrays that can be
    // uploaded to the device "as-is".
    assert( gridHdl.size() != 0 );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &grid.d_volume ), gridHdl.size() ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( grid.d_volume ), gridHdl.data(), gridHdl.size(),
        cudaMemcpyHostToDevice ) );

    grid.handle = std::move( gridHdl );
}


void cleanupVolume( Volume& volume )
{
    // OptiX cleanup
	CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( volume.d_volume ) ) );
}

void buildVolumeAccel( VolumeAccel& accel, const Volume& volume, const OptixDeviceContext& context )
{
    // Build accel for the volume and store it in a VolumeAccel struct.
    //
    // For Optix the NanoVDB volume is represented as a 3D box in index coordinate space. The volume's
    // GAS is created from a single AABB. Because the index space is by definition axis aligned with the
    // volume's voxels, this AABB is the bounding-box of the volume's "active voxels".

    {
        auto grid_handle = volume.handle.grid<float>();

		// get this grid's aabb
        sutil::Aabb aabb;
        {
            // indexBBox returns the extrema of the (integer) voxel coordinates.
            // Thus the actual bounds of the space covered by those voxels extends
            // by one unit (or one "voxel size") beyond those maximum indices.
            auto bbox = grid_handle->indexBBox();
            nanovdb::Coord boundsMin( bbox.min() );
            nanovdb::Coord boundsMax( bbox.max() + nanovdb::Coord( 1 ) ); // extend by one unit

            float3 min = { 
                static_cast<float>( boundsMin[0] ), 
                static_cast<float>( boundsMin[1] ), 
                static_cast<float>( boundsMin[2] )};
            float3 max = {
                static_cast<float>( boundsMax[0] ),
                static_cast<float>( boundsMax[1] ),
                static_cast<float>( boundsMax[2] )};

            aabb =sutil::Aabb( min, max );
        }

		// up to device
        CUdeviceptr d_aabb;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb ), sizeof( sutil::Aabb ) ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void* >(  d_aabb ), &aabb, 
            sizeof( sutil::Aabb ), cudaMemcpyHostToDevice ) );

        // Make build input for this grid
        uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
        build_input.customPrimitiveArray.flags = &aabb_input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.numPrimitives = 1;
        build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.customPrimitiveArray.primitiveIndexOffset = 0;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, 
            &build_input, 1, &gas_buffer_sizes ) );

        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ),
            gas_buffer_sizes.tempSizeInBytes ) );
        CUdeviceptr d_output_buffer_gas;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_output_buffer_gas ),
            gas_buffer_sizes.outputSizeInBytes ) );
        CUdeviceptr d_compacted_size;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_compacted_size ), sizeof( size_t ) ) );

        OptixAccelEmitDesc emit_property = {};
        emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = d_compacted_size;

        OPTIX_CHECK( optixAccelBuild( context,
            0,
            &accel_options,
            &build_input,
            1,
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_output_buffer_gas,
            gas_buffer_sizes.outputSizeInBytes,
            &accel.handle,
            &emit_property,
            1 ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_aabb ) ) );
        size_t compacted_size;
        CUDA_CHECK( cudaMemcpy( &compacted_size, reinterpret_cast<void*>( emit_property.result ),
            sizeof( size_t ), cudaMemcpyDeviceToHost ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_compacted_size ) ) );
        if( compacted_size < gas_buffer_sizes.outputSizeInBytes ) 
        {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &accel.d_buffer ), compacted_size ) );
            OPTIX_CHECK( optixAccelCompact( context, 0, accel.handle,
                accel.d_buffer, compacted_size, &accel.handle ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_output_buffer_gas ) ) );
        }
        else 
        {
            accel.d_buffer = d_output_buffer_gas;
        }
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    }
}

void cleanupVolumeAccel( VolumeAccel& accel )
{
	CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( accel.d_buffer ) ) );
}

void getOptixTransform( const Volume& grid, float transform[] )
{
    // Extract the index-to-world-space affine transform from the Grid and convert
    // to 3x4 row-major matrix for Optix.
	auto* grid_handle = grid.handle.grid<float>();
	const nanovdb::Map& map = grid_handle->map();
	transform[0] = map.mMatF[0]; transform[1] = map.mMatF[1]; transform[2]  = map.mMatF[2]; transform[3]  = map.mVecF[0];
	transform[4] = map.mMatF[3]; transform[5] = map.mMatF[4]; transform[6]  = map.mMatF[5]; transform[7]  = map.mVecF[1];
	transform[8] = map.mMatF[6]; transform[9] = map.mMatF[7]; transform[10] = map.mMatF[8]; transform[11] = map.mVecF[2];
}

sutil::Aabb worldAabb( const Volume& grid )
{
	auto* meta = grid.handle.gridMetaData();
	auto bbox = meta->worldBBox();
	float3 min = { static_cast<float>( bbox.min()[0] ),
                   static_cast<float>( bbox.min()[1] ),
                   static_cast<float>( bbox.min()[2] ) };
	float3 max = { static_cast<float>( bbox.max()[0] ),
                   static_cast<float>( bbox.max()[1] ),
                   static_cast<float>( bbox.max()[2] ) };

	return sutil::Aabb( min, max );
}

// ----------------------------------------------------------------------------
// Plane stuff
// ----------------------------------------------------------------------------

void createPlane( Plane& plane, const sutil::Aabb& aabb )
{
    plane.transform = sutil::Matrix4x4::identity();

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &plane.d_indices ), 
        plane.num_indices * sizeof( unsigned int ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( plane.d_indices ), plane.indices,
        plane.num_indices * sizeof(unsigned int), cudaMemcpyHostToDevice));

    float extent = plane_scale_factor * aabb.maxExtent();
    float3 center = aabb.center();

    plane.positions[0] = make_float3( center.x - extent, aabb.m_min.y - 1.0f, center.z - extent );
	plane.positions[1] = make_float3( center.x - extent, aabb.m_min.y - 1.0f, center.z + extent );
	plane.positions[2] = make_float3( center.x + extent, aabb.m_min.y - 1.0f, center.z + extent );
	plane.positions[3] = make_float3( center.x + extent, aabb.m_min.y - 1.0f, center.z - extent );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &plane.d_positions ), 
        plane.num_positions * sizeof( float3 ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( plane.d_positions ), plane.positions,
        plane.num_positions * sizeof( float3 ), cudaMemcpyHostToDevice ) );
    
    plane.material.base_color = make_float3( 0.1f, 0.1f, 0.1f );

    plane.aabb = sutil::Aabb(
        make_float3( -10.0f, 0.0f, -10.0f ),
        make_float3(  10.0f, 0.0f,  10.0f ) );
}

void cleanupPlane( Plane& plane )
{
    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( plane.d_indices ) ) );	
    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( plane.d_positions ) ) );	
}

void buildPlaneAccel( PlaneAccel& plane_accel, const Plane& plane, const OptixDeviceContext& context )
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                         | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // needed to compute normals in closest-hit program
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;
    uint32_t triangle_input_flags        = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof( float3 );
    build_input.triangleArray.numVertices = plane.num_positions;
    build_input.triangleArray.vertexBuffers = &( plane.d_positions );
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof( unsigned int ) * 3;
    build_input.triangleArray.numIndexTriplets = plane.num_indices / 3;
    build_input.triangleArray.indexBuffer = plane.d_indices;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &build_input,
		1, &gas_buffer_sizes ) );

    CUdeviceptr d_temp = 0;
    CUdeviceptr d_temp_output;
    CUdeviceptr d_temp_compactedSizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp ), gas_buffer_sizes.tempSizeInBytes ) ); 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_output ), gas_buffer_sizes.outputSizeInBytes ) ); 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_compactedSizes ), sizeof(size_t) ) ); 

    emitProperty.result = d_temp_compactedSizes;

    OPTIX_CHECK( optixAccelBuild(
        context,
        0,   // CUDA stream
        &accel_options,
        &build_input,
        1,
        d_temp,
        gas_buffer_sizes.tempSizeInBytes,
        d_temp_output,
        gas_buffer_sizes.outputSizeInBytes,
        &plane_accel.handle,
        &emitProperty,  // emitted property list
        1               // num emitted properties
    ) );

    size_t compacted_size = 0;
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( &compacted_size ),
        reinterpret_cast<void*>( d_temp_compactedSizes ), sizeof( size_t ),
        cudaMemcpyDeviceToHost
    ) );

    if( gas_buffer_sizes.outputSizeInBytes > compacted_size )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &plane_accel.d_buffer ), compacted_size ) );
        OPTIX_CHECK( optixAccelCompact( context, 0, plane_accel.handle, plane_accel.d_buffer,
            compacted_size, &plane_accel.handle ) );
    }
    else
    {
        plane_accel.d_buffer = d_temp_output;
        d_temp_output = 0;
    }

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_output ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_compactedSizes ) ) );
}

void cleanupPlaneAccel( PlaneAccel& plane_accel )
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( plane_accel.d_buffer ) ) );
}


// ----------------------------------------------------------------------------
// Cube stuff
// ----------------------------------------------------------------------------

void createCube( Cube& cube, const sutil::Aabb& aabb )
{
    cube.transform = sutil::Matrix4x4::identity();

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &cube.d_indices ), 
        cube.num_indices * sizeof( unsigned int ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( cube.d_indices ), cube.indices,
        cube.num_indices * sizeof(unsigned int), cudaMemcpyHostToDevice));

    float extent = cube_scale_factor * aabb.maxExtent();
    float3 center = aabb.center();
    float height_offset = center.y - extent;

    cube.positions[0] = make_float3( 0, height_offset, extent );
    cube.positions[1] = make_float3( 0, height_offset, 0 );
    cube.positions[2] = make_float3( extent, height_offset, 0 ); 
    cube.positions[3] = make_float3( extent, height_offset, extent );

    cube.positions[4] = make_float3( 0, height_offset + extent, extent );
    cube.positions[5] = make_float3( extent, height_offset + extent, extent );
    cube.positions[6] = make_float3( extent, height_offset + extent, 0 );
    cube.positions[7] = make_float3( 0, height_offset + extent, 0 );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &cube.d_positions ), 
        cube.num_positions * sizeof( float3 ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( cube.d_positions ), cube.positions,
        cube.num_positions * sizeof( float3 ), cudaMemcpyHostToDevice ) );
    
    cube.material.base_color = make_float3( 0.2f, 0.05f, 0.05f ); // redish cube

    cube.aabb = sutil::Aabb(
        make_float3( 0, 0, 0 ),
        make_float3( extent, extent, extent )
    );
}

void cleanupCube( Cube& cube )
{
    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( cube.d_indices ) ) );	
    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( cube.d_positions ) ) );	
}

void buildCubeAccel( CubeAccel& cube_accel, const Cube& cube, const OptixDeviceContext& context )
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                         | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // needed to compute normals in closest-hit program
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;
    uint32_t triangle_input_flags        = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof( float3 );
    build_input.triangleArray.numVertices = cube.num_positions;
    build_input.triangleArray.vertexBuffers = &( cube.d_positions );
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof( unsigned int ) * 3;
    build_input.triangleArray.numIndexTriplets = cube.num_indices / 3;
    build_input.triangleArray.indexBuffer = cube.d_indices;
    build_input.triangleArray.flags = &triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &build_input,
		1, &gas_buffer_sizes ) );

    CUdeviceptr d_temp = 0;
    CUdeviceptr d_temp_output;
    CUdeviceptr d_temp_compactedSizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp ), gas_buffer_sizes.tempSizeInBytes ) ); 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_output ), gas_buffer_sizes.outputSizeInBytes ) ); 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_compactedSizes ), sizeof(size_t) ) ); 

    emitProperty.result = d_temp_compactedSizes;

    OPTIX_CHECK( optixAccelBuild(
        context,
        0,   // CUDA stream
        &accel_options,
        &build_input,
        1,
        d_temp,
        gas_buffer_sizes.tempSizeInBytes,
        d_temp_output,
        gas_buffer_sizes.outputSizeInBytes,
        &cube_accel.handle,
        &emitProperty,  // emitted property list
        1               // num emitted properties
    ) );

    size_t compacted_size = 0;
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( &compacted_size ),
        reinterpret_cast<void*>( d_temp_compactedSizes ), sizeof( size_t ),
        cudaMemcpyDeviceToHost
    ) );

    if( gas_buffer_sizes.outputSizeInBytes > compacted_size )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &cube_accel.d_buffer ), compacted_size ) );
        OPTIX_CHECK( optixAccelCompact( context, 0, cube_accel.handle, cube_accel.d_buffer,
            compacted_size, &cube_accel.handle ) );
    }
    else
    {
        cube_accel.d_buffer = d_temp_output;
        d_temp_output = 0;
    }

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_output ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_compactedSizes ) ) );
}

void cleanupCubeAccel( CubeAccel& cube_accel )
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( cube_accel.d_buffer ) ) );
}


// ----------------------------------------------------------------------------
// IAS stuff
// ----------------------------------------------------------------------------

void buildIAS( IAS& ias, int rayTypeCount, 
    const Volume& volume, const VolumeAccel& volume_accel, 
    const Plane& plane, const PlaneAccel& plane_accel,
    const Cube& cube, const CubeAccel& cube_accel,
    const OptixDeviceContext& context )
{
    // The three scene elements, volume, box, and plane, are added to a single-
    // level IAS; each receiving its own world-space transform. For the volume,
    // this is the transform stored in the Grid.
    std::vector<OptixInstance> optix_instances;
    optix_instances.reserve( 2 );

    unsigned int sbt_offset = 0;

    // process plane
    {
        OptixInstance optix_instance = {};

        optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId = static_cast<unsigned int>( optix_instances.size() );
        optix_instance.sbtOffset = sbt_offset;
        optix_instance.visibilityMask = PLANE_OBJECT;
        optix_instance.traversableHandle = plane_accel.handle;
        memcpy( optix_instance.transform, plane.transform.getData(), sizeof( float ) * 12 );

        sbt_offset += rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
        optix_instances.push_back( optix_instance );
    }

    // process cube
    {
        OptixInstance optix_instance = {};

        optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId = static_cast<unsigned int>( optix_instances.size() );
        optix_instance.sbtOffset = sbt_offset;
        optix_instance.visibilityMask = CUBE_OBJECT;
        optix_instance.traversableHandle = cube_accel.handle;
        memcpy( optix_instance.transform, cube.transform.getData(), sizeof( float ) * 12 );

        sbt_offset += rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
        optix_instances.push_back( optix_instance );
    }

    // process volume
    {
        OptixInstance optix_instance = {};

        optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId = static_cast<unsigned int>( optix_instances.size() );
        optix_instance.sbtOffset = sbt_offset;
        optix_instance.visibilityMask = VOLUME_OBJECT;
        optix_instance.traversableHandle = volume_accel.handle;
        getOptixTransform( volume, optix_instance.transform ); // transform as stored in Grid

        sbt_offset += rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
        optix_instances.push_back( optix_instance );
    }

    const size_t instances_size_in_bytes = sizeof( OptixInstance ) * optix_instances.size();
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_instances ), instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( ias.d_instances ),
                optix_instances.data(),
                instances_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );
    dp_plane_y = ias.d_instances + 7 * sizeof( float ); // seventh element of first transform is y offset

    dp_cube_x = ias.d_instances + sizeof( OptixInstance ) + 3 * sizeof( float ); // 3rd element of second transform is x offset
    dp_cube_z = ias.d_instances + sizeof( OptixInstance ) + 11 * sizeof( float ); // 11th element of second transform is z offset

    ias.build_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias.build_input.instanceArray.instances    = ias.d_instances;
    ias.build_input.instanceArray.numInstances = static_cast<unsigned int>( optix_instances.size() );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                context,
                &accel_options,
                &ias.build_input,
                1, // num build inputs
                &ias.buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_temp_buffer ),
                ias.buffer_sizes.tempSizeInBytes
                ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &ias.d_buffer ),
                ias.buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                context,
                nullptr,                  // CUDA stream
                &accel_options,
                &ias.build_input,
                1,                  // num build inputs
                d_temp_buffer,
                ias.buffer_sizes.tempSizeInBytes,
                ias.d_buffer,
                ias.buffer_sizes.outputSizeInBytes,
                &ias.handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );

    // make update temp buffer for ias
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_update_buffer ),
        ias.buffer_sizes.tempUpdateSizeInBytes ) );
}


void updateIAS( IAS& ias, const OptixDeviceContext& context )
{
    // Rebuild the IAS after scene elements were moved.

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_UPDATE;

    OPTIX_CHECK( optixAccelBuild(
                context,
                nullptr,            // CUDA stream
                &accel_options,
                &ias.build_input,
                1,                  // num build inputs
                ias.d_update_buffer,
                ias.buffer_sizes.tempUpdateSizeInBytes,
                ias.d_buffer,
                ias.buffer_sizes.outputSizeInBytes,
                &ias.handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );
}

void cleanupIAS( IAS& ias )
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ias.d_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ias.d_instances ) ) );
}


void createModule( OptixModule& module, const OptixDeviceContext& context )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur            = false;
    pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues          = NUM_PAYLOAD_VALUES;
    pipeline_compile_options.numAttributeValues        = 0; // TODO
    pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";


    char log[2048];
    size_t sizeof_log = sizeof( log );

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "volume.cu", inputSize );
    module = {};
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &module
    ) );
}


// ----------------------------------------------------------------------------
// ProgramGroups
// ----------------------------------------------------------------------------

void createProgramGroups( ProgramGroups& program_groups, 
    const OptixModule& module, 
    const OptixDeviceContext& context )
{
    OptixProgramGroupOptions program_group_options = {};

    char log[2048];
    size_t sizeof_log = sizeof( log );

    //
    // Ray generation
    //
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
			context,
			&raygen_prog_group_desc,
			1,                             // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&program_groups.raygen
		) );
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
			context,
			&miss_prog_group_desc,
			1,                             // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&program_groups.miss_radiance
		) );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
			context,
			&miss_prog_group_desc,
			1,                             // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&program_groups.miss_occlusion
		) );
    }

    //
    // Mesh hit group
    //
    // This implements simple diffuse shading for the mesh surfaces (plane and cube).
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance_mesh";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
			context,
			&hit_prog_group_desc,
			1,                             // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&program_groups.mesh_radiance
		) );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion_mesh";
        sizeof_log = sizeof( log );
        OPTIX_CHECK( optixProgramGroupCreate(
			context,
			&hit_prog_group_desc,
			1,                             // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&program_groups.mesh_occlusion
		) );
    }

    //
    // Volume hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance_volume";
        hit_prog_group_desc.hitgroup.moduleAH = nullptr;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        hit_prog_group_desc.hitgroup.moduleIS = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__volume";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            context,
            &hit_prog_group_desc,
            1,                             // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &program_groups.volume_radiance
        ) );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion_volume";
        hit_prog_group_desc.hitgroup.moduleAH = nullptr;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        hit_prog_group_desc.hitgroup.moduleIS = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__volume";
        sizeof_log = sizeof( log );
        OPTIX_CHECK( optixProgramGroupCreate(
            context,
            &hit_prog_group_desc,
            1,                             // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &program_groups.volume_occlusion
        ) );
    }
}

void cleanupProgramGroups( ProgramGroups& program_groups )
{
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.raygen           ) );
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.miss_radiance    ) );
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.miss_occlusion   ) );
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.mesh_radiance   ) );
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.mesh_occlusion  ) );
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.volume_radiance  ) );
    OPTIX_CHECK( optixProgramGroupDestroy( program_groups.volume_occlusion ) );
}


void createPipeline( OptixPipeline& pipeline, const ProgramGroups& programs, const OptixDeviceContext& context )
{
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur            = false;
    pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues          = NUM_PAYLOAD_VALUES;
    pipeline_compile_options.numAttributeValues        = 0; // TODO
    pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 4;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                &programs.raygen,                      // ptr to first program group
                sizeof( ProgramGroups ) / sizeof( OptixProgramGroup ), // number of program groups
                log,
                &sizeof_log,
                &pipeline
                ) );

	// We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( programs.raygen,          &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( programs.miss_radiance,   &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( programs.miss_occlusion,  &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( programs.mesh_radiance,  &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( programs.mesh_occlusion, &stack_sizes ) );

    uint32_t max_trace_depth = 4;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 4;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                max_cc_depth,
                max_dc_depth,
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size
                ) );

    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK( optixPipelineSetStackSize(
                pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth
                ) );

}


void createSBT( OptixShaderBindingTable& sbt, const ProgramGroups& program_groups, 
    const Volume& volume, const Plane& plane, const Cube& cube )
{
    {
        const size_t raygen_record_size = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.raygenRecord ), raygen_record_size ) );

        sutil::EmptyRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( program_groups.raygen, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( sbt.raygenRecord ),
			&rg_sbt,
			raygen_record_size,
			cudaMemcpyHostToDevice
		) );
    }

    {
        const size_t miss_record_size = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc(
			reinterpret_cast<void**>( &sbt.missRecordBase ),
			miss_record_size * RAY_TYPE_COUNT
		) );

        sutil::EmptyRecord ms_sbt[RAY_TYPE_COUNT];
        OPTIX_CHECK( optixSbtRecordPackHeader( program_groups.miss_radiance,  &ms_sbt[0] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( program_groups.miss_occlusion, &ms_sbt[1] ) );

        CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( sbt.missRecordBase ),
			ms_sbt,
			miss_record_size * RAY_TYPE_COUNT,
			cudaMemcpyHostToDevice
		) );
        sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
        sbt.missRecordCount         = RAY_TYPE_COUNT;
    }

    {
        std::vector<HitGroupRecord> hitgroup_records;
        // plane sbt record
        {
            HitGroupRecord rec = {};
            OPTIX_CHECK(optixSbtRecordPackHeader( program_groups.mesh_radiance, &rec ) );

            rec.data.material_data.lambert = plane.material;
            hitgroup_records.push_back( rec );
            
            OPTIX_CHECK( optixSbtRecordPackHeader( program_groups.mesh_occlusion, &rec ) );
            hitgroup_records.push_back( rec );
        }

        // cube sbt record
        {
            HitGroupRecord rec = {};
            OPTIX_CHECK(optixSbtRecordPackHeader( program_groups.mesh_radiance, &rec ) );

            rec.data.material_data.lambert = cube.material;
            hitgroup_records.push_back( rec );
            
            OPTIX_CHECK( optixSbtRecordPackHeader( program_groups.mesh_occlusion, &rec ) );
            hitgroup_records.push_back( rec );
        }

        // volume sbt record
        {
            HitGroupRecord rec = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( program_groups.volume_radiance, &rec ) );
            rec.data.geometry_data.volume.grid = reinterpret_cast<void*>( volume.d_volume );
            rec.data.material_data.volume.opacity = 0.125f;
            hitgroup_records.push_back( rec );

            OPTIX_CHECK(optixSbtRecordPackHeader( program_groups.volume_occlusion, &rec ) );
            hitgroup_records.push_back( rec );
        }

        const size_t hitgroup_record_size = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &sbt.hitgroupRecordBase ),
                    hitgroup_record_size*hitgroup_records.size()
                    ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                    hitgroup_records.data(),
                    hitgroup_record_size*hitgroup_records.size(),
                    cudaMemcpyHostToDevice
                    ) );

        sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>( hitgroup_record_size );
        sbt.hitgroupRecordCount         = static_cast<unsigned int>( hitgroup_records.size() );

        // get device pointer to the opacity value in first volume hit record (radiosity)
        CUdeviceptr dp_opacity = sbt.hitgroupRecordBase + 4 * sizeof( HitGroupRecord )
            + OPTIX_SBT_RECORD_HEADER_SIZE + sizeof( GeometryData );
        {
            dp_opacities.push_back( dp_opacity );
            dp_opacity += sizeof( HitGroupRecord ); // advance device pointer to second opacity (occlusion) 
            dp_opacities.push_back( dp_opacity );
            dp_opacity += sizeof( HitGroupRecord );
        }
    }
}

void cleanupSBT( OptixShaderBindingTable& sbt )
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
}


void initCameraState( const sutil::Aabb& aabb )
{
    camera.setFovY( 45.0f );
    camera.setLookat( aabb.center() );
    camera.setEye   ( aabb.center() + make_float3( 0.0f, 0.0f, 1.5f * aabb.maxExtent() ) );

    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock( true );
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;
    std::string infile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--volume" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            infile = argv[++i];
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( infile.empty() ) // ...use default
        infile = sutil::sampleDataFilePath( "Volumes/smoke.nvdb" );

    try
    {
        createContext( g_context );

        loadVolume( g_volume, infile.c_str() );
        buildVolumeAccel( g_volume_accel, g_volume, g_context );
        createPlane( g_plane, worldAabb( g_volume ) );
        buildPlaneAccel( g_plane_accel, g_plane, g_context );
        createCube( g_cube, worldAabb( g_volume ) );
        buildCubeAccel( g_cube_accel, g_cube, g_context );
        createModule( module, g_context );
        createProgramGroups( program_groups, module, g_context );
        createPipeline( pipeline, program_groups, g_context );
        createSBT( sbt, program_groups, g_volume, g_plane, g_cube );
        buildIAS( ias, RAY_TYPE_COUNT,
            g_volume, g_volume_accel,
            g_plane, g_plane_accel,
            g_cube, g_cube_accel,
            g_context );

        initCameraState( worldAabb( g_volume) );
        initLaunchParams( launch_params, ias.handle, worldAabb( g_volume ) );

        printShortcuts();

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixVolumeViewer", width, height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );


                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, launch_params.params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers(window);

                    ++launch_params.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
			if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
			{
				sutil::initGLFW(); // For GL context
				sutil::initGL();
			}

			sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
			handleCameraUpdate( launch_params.params);
			handleResize( output_buffer );
			launchSubframe( output_buffer );

			sutil::ImageBuffer buffer;
			buffer.data = output_buffer.getHostPointer();
			buffer.width = output_buffer.width();
			buffer.height = output_buffer.height();
			buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

			sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

		cleanupSBT( sbt );
		OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
		cleanupProgramGroups( program_groups );
		OPTIX_CHECK( optixModuleDestroy( module ) );
		cleanupLaunchParams( launch_params );
		cleanupIAS( ias );
		cleanupPlaneAccel( g_plane_accel );
		cleanupVolumeAccel( g_volume_accel );
		cleanupPlane( g_plane );
		cleanupVolume( g_volume );
        optixDeviceContextDestroy( g_context );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
