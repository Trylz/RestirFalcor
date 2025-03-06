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

// This sample shows how to run module creation in a separate process
// so that it can be aborted at any time by killing that process
// (by pressing the 'A' key). The compiled result is stored in the OptiX
// disk cache, so that when the main application creates the module again,
// there is no compile time.
//
// This sample is a modified version of the optixBoundValues sample.

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixModuleCreateAbort.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <spawn.h>
#include <signal.h>
#endif



bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 16;

// The number of samples to calculate the light are specified in the launch parameters.
// That value can be specialized to a fixed value at compile time.
// Note than when changing the number of light samples at runtime by pressing the PLUS
// or MINUS keys with specialization enabled recompilation of the closest hit module
// is necessary or it needs to be loaded from the cache.
unsigned int light_samples = 1;
bool         specialize    = true;

// Keep a list of all compile operations in flight to be able to check their state each frame until completed
struct CompileOperation
{
    std::string ptx;
    std::string temp_file;
    OptixModuleCompileOptions module_compile_options;
    OptixPipelineCompileOptions pipeline_compile_options;
    std::vector<OptixModuleCompileBoundValueEntry> bound_values;
    OptixModule* target_module;
#ifdef WIN32
    HANDLE process_handle;
#else
    pid_t process_id;
#endif
};
std::vector<CompileOperation> compile_operations_in_flight;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};


struct Instance
{
    float transform[12];
};


struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle         gas_handle               = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer      = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices               = 0;

    OptixModule                    ptx_module               = 0;
    OptixModule                    ptx_module_radiance      = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline                 = 0;

    OptixProgramGroup              raygen_prog_group        = 0;
    OptixProgramGroup              radiance_miss_group      = 0;
    OptixProgramGroup              occlusion_miss_group     = 0;
    OptixProgramGroup              radiance_hit_group       = 0;
    OptixProgramGroup              occlusion_hit_group      = 0;

    CUstream                       stream                   = 0;
    Params                         params;
    Params*                        d_params;

    OptixShaderBindingTable        sbt                      = {};
};


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t TRIANGLE_COUNT = 32;
const int32_t MAT_COUNT      = 4;

const static std::array<Vertex, TRIANGLE_COUNT* 3> g_vertices =
{  {
    // Floor  -- white lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,    0.0f, 0.0f },

    // Ceiling -- white lambert
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    // Back wall -- white lambert
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    // Right wall -- green lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },

    // Left wall -- red lambert
    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },

    // Short block -- white lambert
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },

    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {  240.0f,    0.0f,  272.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  290.0f,    0.0f,  114.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  130.0f,    0.0f,   65.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {   82.0f,    0.0f,  225.0f, 0.0f },

    // Tall block -- white lambert
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },

    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  472.0f,    0.0f,  406.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  314.0f,    0.0f,  456.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  265.0f,    0.0f,  296.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  423.0f,    0.0f,  247.0f, 0.0f },

    // Ceiling light -- emmissive
    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },

    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },
    {  343.0f,  548.6f,  332.0f, 0.0f }
} };

static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = {{
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
}};



const std::array<float3, MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    { 15.0f, 15.0f,  5.0f }

} };


const std::array<float3, MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f }
} };


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
        trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params& params = static_cast<PathTracerState*>(glfwGetWindowUserPointer( window ))->params;

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params.width, params.height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params.width, params.height );
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

    Params& params = static_cast<PathTracerState*>(glfwGetWindowUserPointer( window ))->params;
    params.width  = res_x;
    params.height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}

void createRadianceModule( PathTracerState& state );

static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
    else if( key == GLFW_KEY_KP_ADD )
    {
        ++light_samples;
        if( specialize )
            createRadianceModule( *static_cast<PathTracerState*>(glfwGetWindowUserPointer( window )) );
    }
    else if( key == GLFW_KEY_KP_SUBTRACT )
    {
        if( light_samples > 1 )
        {
            --light_samples;
            if( specialize )
                createRadianceModule( *static_cast<PathTracerState*>(glfwGetWindowUserPointer( window )) );
        }
    }
    else if( key == GLFW_KEY_S )
    {
        specialize = !specialize;
        // Invoke compile process for new module, pipeline is then later updated in the render loop
        createRadianceModule( *static_cast<PathTracerState*>(glfwGetWindowUserPointer( window )) );
    }
    else if( key == GLFW_KEY_A )
    {
        // Terminate all compile processes in flight
        for( const CompileOperation& operation : compile_operations_in_flight )
        {
#ifdef WIN32
            SUTIL_ASSERT( TerminateProcess( operation.process_handle, 1 ) );
            SUTIL_ASSERT( DeleteFileA( operation.temp_file.c_str() ) );
#else
            SUTIL_ASSERT( kill( operation.process_id, SIGTERM ) == 0 );
            SUTIL_ASSERT( remove( operation.temp_file.c_str() ) == 0 );
#endif
        }
        compile_operations_in_flight.clear();
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --light-samples  | -l       Number of radiance samples (default 1)\n";
    std::cerr << "         --no-specialize             ...\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( PathTracerState& state )
{
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &state.params.accum_buffer ),
                state.params.width * state.params.height * sizeof( float4 )
                ) );
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.light_samples      = light_samples;
    state.params.subframe_index     = 0u;

    state.params.light.emission = make_float3( 15.0f, 15.0f, 5.0f );
    state.params.light.corner   = make_float3( 343.0f, 548.5f, 227.0f );
    state.params.light.v1       = make_float3( 0.0f, 0.0f, 105.0f );
    state.params.light.v2       = make_float3( -130.0f, 0.0f, 0.0f );
    state.params.light.normal   = normalize( cross( state.params.light.v1, state.params.light.v2 ) );
    state.params.handle         = state.gas_handle;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

}


void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                params.width * params.height * sizeof( float4 )
                ) );
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    params.light_samples = light_samples;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state )
{
    // Cannot launch while modules are still compiling (on startup)
    if( state.pipeline == nullptr )
        return;

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( state.d_params ),
                &state.params, sizeof( Params ),
                cudaMemcpyHostToDevice, state.stream
                ) );

    OPTIX_CHECK( optixLaunch(
                state.pipeline,
                state.stream,
                reinterpret_cast<CUdeviceptr>( state.d_params ),
                sizeof( Params ),
                &state.sbt,
                state.params.width,   // launch width
                state.params.height,  // launch height
                1                     // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
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


void initCameraState()
{
    camera.setEye( make_float3( 278.0f, 273.0f, -900.0f ) );
    camera.setLookat( make_float3( 278.0f, 273.0f, 330.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 35.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame(
            make_float3( 1.0f, 0.0f, 0.0f ),
            make_float3( 0.0f, 0.0f, 1.0f ),
            make_float3( 0.0f, 1.0f, 0.0f )
            );
    trackball.setGimbalLock( true );
}


void createContext( PathTracerState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
#ifdef DEBUG
    // Enables validation mode for OptiX:
    //   Enables all debug exceptions during optix launches and stops on first exception.
    //   Enables verification of launch parameter specialization values with the current launch parameter values.
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}


void buildMeshAccel( PathTracerState& state )
{
    //
    // copy mesh data to device
    //
    const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_vertices ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( state.d_vertices ),
                g_vertices.data(), vertices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr  d_mat_indices             = 0;
    const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_mat_indices ),
                g_mat_indices.data(),
                mat_indices_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    //
    // Build triangle GAS
    //
    uint32_t triangle_input_flags[MAT_COUNT] =  // One per SBT record for this build input
    {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
    triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() );
    triangle_input.triangleArray.vertexBuffers               = &state.d_vertices;
    triangle_input.triangleArray.flags                       = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords               = MAT_COUNT;
    triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &triangle_input,
                1,  // num_build_inputs
                &gas_buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
                state.context,
                0,                                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                                  // num build inputs
                d_temp_buffer,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,                      // emitted property list
                1                                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_mat_indices ) ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}


void createModule( OptixDeviceContext context, const std::string& ptx, const OptixModuleCompileOptions& module_compile_options, const OptixPipelineCompileOptions& pipeline_compile_options, OptixModule& module )
{
    // Get current CUDA device and pass that info to the compile process
    int device = 0;
    CUDA_CHECK(cudaGetDevice( &device ));

    // Write PTX to a temporary file that can be passed on to the compile process
    static int invocation = 0;
    const std::string filename = "temp_" + std::to_string( reinterpret_cast<uintptr_t>(&module) ) + "_" + std::to_string(invocation++) + ".ptx";
    std::ofstream temp_file( filename, std::ios::binary  );
    SUTIL_ASSERT( temp_file.is_open() );
    temp_file.write( ptx.data(), ptx.size() );
    temp_file.close();

    // Build command-line for the compile process
    std::string cmd;
    cmd += " --file " + filename;
    cmd += " --device " + std::to_string( device );

    cmd += " --maxRegisterCount ";
    cmd += std::to_string( module_compile_options.maxRegisterCount );
    cmd += " --optLevel ";
    cmd += std::to_string( static_cast<int>(module_compile_options.optLevel) );
    cmd += " --debugLevel ";
    cmd += std::to_string( static_cast<int>(module_compile_options.debugLevel) );
    for( unsigned int i = 0; i < module_compile_options.numBoundValues; ++i )
    {
        const OptixModuleCompileBoundValueEntry& bound_value = module_compile_options.boundValues[i];

        cmd += " --boundValue ";
        cmd += std::to_string( bound_value.pipelineParamOffsetInBytes );
        cmd += " ";
        cmd += std::to_string( bound_value.sizeInBytes );
        cmd += " \"";
        cmd += bound_value.annotation;
        cmd += "\" \"";
        // Encode bound value data as a string
        for( size_t byte = 0; byte < bound_value.sizeInBytes; ++byte )
        {
            const char byte_value = static_cast<const char*>(bound_value.boundValuePtr)[byte];
            cmd += '0' + ( byte_value       & 0xF); // byte_low
            cmd += '0' + ((byte_value >> 4) & 0xF); // byte_high
        }
        cmd += "\"";
    }

    cmd += " --usesMotionBlur ";
    cmd += pipeline_compile_options.usesMotionBlur ? "1" : "0";
    cmd += " --traversableGraphFlags ";
    cmd += std::to_string( pipeline_compile_options.traversableGraphFlags );
    cmd += " --numPayloadValues ";
    cmd += std::to_string( pipeline_compile_options.numPayloadValues );
    cmd += " --numAttributeValues ";
    cmd += std::to_string( pipeline_compile_options.numAttributeValues );
    cmd += " --exceptionFlags ";
    cmd += std::to_string( pipeline_compile_options.exceptionFlags );
    cmd += " --pipelineLaunchParamsVariableName ";
    cmd += pipeline_compile_options.pipelineLaunchParamsVariableName;
    cmd += " --usesPrimitiveTypeFlags ";
    cmd += std::to_string( pipeline_compile_options.usesPrimitiveTypeFlags );

    CompileOperation operation;
    operation.ptx = ptx;
    operation.temp_file = filename;
    operation.module_compile_options = module_compile_options;
    operation.pipeline_compile_options = pipeline_compile_options;
    operation.bound_values.assign(module_compile_options.boundValues, module_compile_options.boundValues + module_compile_options.numBoundValues);
    operation.target_module = &module;

#ifdef WIN32
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi = {};
    SUTIL_ASSERT( CreateProcessA( "optixModuleCreateProcess.exe", const_cast<char*>(cmd.c_str()), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi ) );

    operation.process_handle = pi.hProcess;
#else
    std::vector<char*> argv;
    // Split combined argument string into separate arguments
    cmd += ' ';
    for( size_t offset = 0, next; (next = cmd.find( ' ', offset )) != std::string::npos; offset = next + 1 )
    {
        cmd[next] = '\0';
        argv.push_back( const_cast<char*>(cmd.c_str()) + offset );
    }

    char process_name[] = "optixModuleCreateProcess";
    argv[0] = process_name;
    argv.push_back( nullptr ); // Terminate argument list

    extern char **environ;
    SUTIL_ASSERT( posix_spawn( &operation.process_id, process_name, nullptr, nullptr, argv.data(), environ ) == 0 );
#endif

    compile_operations_in_flight.push_back( operation );
}


void createRadianceModule( PathTracerState& state )
{
    OptixModuleCompileOptions module_compile_options ={};
    OptixModuleCompileBoundValueEntry boundValue ={};
    if( specialize )
    {
        boundValue.pipelineParamOffsetInBytes = offsetof( Params, light_samples );
        boundValue.sizeInBytes                = sizeof( Params::light_samples );
        boundValue.boundValuePtr              = &light_samples;
        boundValue.annotation                 = "light_samples";
        module_compile_options.numBoundValues = 1;
        module_compile_options.boundValues    = &boundValue;
    }

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixModuleCreateAbort_ch.cu", inputSize );
    std::string ptx( input, inputSize );
    createModule( state.context, ptx, module_compile_options, state.pipeline_compile_options, state.ptx_module_radiance );
}

void createModule( PathTracerState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 2;
    state.pipeline_compile_options.numAttributeValues    = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixModuleCreateAbort.cu", inputSize );
    std::string ptx( input, inputSize );

    createModule( state.context, ptx, module_compile_options, state.pipeline_compile_options, state.ptx_module );

    createRadianceModule( state );
}

void createRadianceProgramGroup( PathTracerState& state )
{
    char   log[2048];
    size_t sizeof_log = sizeof( log );

    OptixProgramGroupOptions  program_group_options ={};
    OptixProgramGroupDesc hit_prog_group_desc        ={};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module_radiance;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    sizeof_log                                       = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &hit_prog_group_desc,
        1,  // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.radiance_hit_group
    ) );
}

void createProgramGroups( PathTracerState& state )
{
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        OptixProgramGroupDesc raygen_prog_group_desc    = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    state.context, &raygen_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &state.raygen_prog_group
                    ) );
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log                                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    state.context, &miss_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &state.radiance_miss_group
                    ) );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log                                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    state.context, &miss_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &state.occlusion_miss_group
                    ) );
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc        = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log                                       = sizeof( log );
        OPTIX_CHECK( optixProgramGroupCreate(
                    state.context,
                    &hit_prog_group_desc,
                    1,  // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &state.occlusion_hit_group
                    ) );
    }
    createRadianceProgramGroup( state );
}

void createPipeline( PathTracerState& state )
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.occlusion_miss_group,
        state.radiance_hit_group,
        state.occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 2;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                state.context,
                &state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &state.pipeline
                ) );

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.raygen_prog_group,    &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.radiance_miss_group,  &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.occlusion_miss_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.radiance_hit_group,   &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.occlusion_hit_group,  &stack_sizes ) );

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
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

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK( optixPipelineSetStackSize(
                state.pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth
                ) );
}

void allocateSBT( PathTracerState& state )
{
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.sbt.raygenRecord), raygen_record_size ) );

    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.sbt.missRecordBase), miss_record_size * RAY_TYPE_COUNT ) );
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount             = RAY_TYPE_COUNT;

    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>(&state.sbt.hitgroupRecordBase),
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT
    ) );
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * MAT_COUNT;
}

void fillSBT( PathTracerState& state )
{
    const size_t raygen_record_size = sizeof( RayGenRecord );
    RayGenRecord rg_sbt ={};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>(state.sbt.raygenRecord),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ) );


    const size_t miss_record_size = sizeof( MissRecord );
    MissRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_miss_group, &ms_sbt[0] ) );
    ms_sbt[0].data.bg_color = make_float4( 0.0f );
    OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_miss_group, &ms_sbt[1] ) );
    ms_sbt[1].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>(state.sbt.missRecordBase),
        ms_sbt,
        miss_record_size*RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ) );
}

void fillHitGroupSBT( PathTracerState& state )
{
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
    for( int i = 0; i < MAT_COUNT; ++i )
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

            OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.emission_color = g_emission_colors[i];
            hitgroup_records[sbt_idx].data.diffuse_color  = g_diffuse_colors[i];
            hitgroup_records[sbt_idx].data.vertices       = reinterpret_cast<float4*>(state.d_vertices);
        }

        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( state.occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
        }
    }

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>(state.sbt.hitgroupRecordBase),
        hitgroup_records,
        hitgroup_record_size*RAY_TYPE_COUNT*MAT_COUNT,
        cudaMemcpyHostToDevice
    ) );
}

void updatePipeline( PathTracerState& state )
{
    // destroy old stuff
    if (state.pipeline != nullptr)
    {
        OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
        OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_miss_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_miss_group ) );
    }

    createProgramGroups( state );
    createRadianceProgramGroup( state );
    createPipeline( state );
    fillSBT(state);
    fillHitGroupSBT( state );
}

void updatePipelineWhenChanged( PathTracerState& state )
{
    if( compile_operations_in_flight.empty() )
        return;

    bool need_pipeline_update = false;
    for( auto it = compile_operations_in_flight.begin(); it != compile_operations_in_flight.end(); )
    {
#ifdef WIN32
        // Check if the compile process has already exited
        if( WaitForSingleObject( it->process_handle, 0 ) != WAIT_OBJECT_0 )
        {
            ++it;
            continue;
        }

        // Delete temporary file
        const BOOL delete_success = DeleteFileA( it->temp_file.c_str() );

        // Process has exited, so check the exit code to ensure module compilation was successfull
        DWORD exit_code = EXIT_FAILURE;
        GetExitCodeProcess( it->process_handle, &exit_code );

        CloseHandle( it->process_handle );

        SUTIL_ASSERT_MSG( exit_code == 0, "Compile process was not successfull" );
        SUTIL_ASSERT( delete_success );
#else
        int status = -1;
        if( waitpid( it->process_id, &status, WNOHANG ) == 0 )
        {
            ++it;
            continue;
        }

        SUTIL_ASSERT_MSG( WEXITSTATUS(status) == 0, "Compile process was not successfull" );
        SUTIL_ASSERT( remove( it->temp_file.c_str() ) == 0 );
#endif

        it->module_compile_options.boundValues = it->bound_values.data();

        // The module should be in cache, so simply create it again in this context
        // This API call should now return instantenously (unless caching failed, in which case this would recompile the module again)
        OPTIX_CHECK( optixModuleCreateFromPTX(
            state.context,
            &it->module_compile_options,
            &it->pipeline_compile_options,
            it->ptx.c_str(),
            it->ptx.size(),
            nullptr, 0,
            it->target_module
        ) );

        need_pipeline_update = true;

        // This compile operation is finished, so remove it from the list
        it = compile_operations_in_flight.erase( it );
    }

    if( need_pipeline_update )
    {
        updatePipeline( state );
    }
}

void cleanupState( PathTracerState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.radiance_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.occlusion_miss_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module_radiance ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_vertices ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params ) ) );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void displaySpecializationInfo( GLFWwindow* window )
{
    static char display_text[256];

    sutil::beginFrameImGui();
    sprintf( display_text,
        "light samples  [+/-]: %d\n"
        "specialization [S]  : %s\n%s",
        light_samples,
        specialize ? "on" : "off",
        !compile_operations_in_flight.empty() ? "compiling ... [A] to abort\n" : "");
    Params& params = static_cast<PathTracerState*>(glfwGetWindowUserPointer( window ))->params;
    sutil::displayText( display_text, 10.0f, (float)params.height - 70.f );
    sutil::endFrameImGui();
}

int main( int argc, char* argv[] )
{
    PathTracerState state;
    state.params.width                             = 768;
    state.params.height                            = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
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
        else if( arg == "--no-specialize" )
        {
            specialize = false;
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
        }
        else if( arg == "--light-samples" || arg == "-l" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            light_samples = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

        //
        // Set up OptiX state
        //
        createContext( state );
        buildMeshAccel( state );
        createModule( state );
        allocateSBT( state );
        initLaunchParams( state );

        GLFWwindow* window = sutil::initUI( "optixModuleCreateAbort", state.params.width, state.params.height );
        glfwSetMouseButtonCallback( window, mouseButtonCallback );
        glfwSetCursorPosCallback( window, cursorPosCallback );
        glfwSetWindowSizeCallback( window, windowSizeCallback );
        glfwSetWindowIconifyCallback( window, windowIconifyCallback );
        glfwSetKeyCallback( window, keyCallback );
        glfwSetScrollCallback( window, scrollCallback );
        glfwSetWindowUserPointer( window, &state );

        //
        // Render loop
        //
        {
            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );

            output_buffer.setStream( state.stream );
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time( 0.0 );
            std::chrono::duration<double> render_time( 0.0 );
            std::chrono::duration<double> display_time( 0.0 );

            do
            {
                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                updatePipelineWhenChanged( state );

                updateState( output_buffer, state.params );
                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                launchSubframe( output_buffer, state );
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe( output_buffer, gl_display, window );
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                sutil::displayStats( state_update_time, render_time, display_time );

                displaySpecializationInfo( window );

                glfwSwapBuffers( window );

                ++state.params.subframe_index;
            } while( !glfwWindowShouldClose( window ) );
            CUDA_SYNC_CHECK();
        }

        sutil::cleanupUI( window );

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
