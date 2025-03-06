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

#include "optixDynamicGeometry.h"
#include "vertices.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;


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

struct DynamicGeometryState
{
    OptixDeviceContext context = 0;

    size_t                         temp_buffer_size = 0;
    CUdeviceptr                    d_temp_buffer = 0;
    CUdeviceptr                    d_temp_vertices = 0;
    CUdeviceptr                    d_instances = 0;

    unsigned int                   triangle_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput                ias_instance_input = {};
    OptixBuildInput                triangle_input = {};

    OptixTraversableHandle         ias_handle;
    OptixTraversableHandle         static_gas_handle;
    OptixTraversableHandle         deforming_gas_handle;
    OptixTraversableHandle         exploding_gas_handle;

    CUdeviceptr                    d_ias_output_buffer = 0;
    CUdeviceptr                    d_static_gas_output_buffer;
    CUdeviceptr                    d_deforming_gas_output_buffer;
    CUdeviceptr                    d_exploding_gas_output_buffer;

    size_t                         ias_output_buffer_size = 0;
    size_t                         static_gas_output_buffer_size = 0;
    size_t                         deforming_gas_output_buffer_size = 0;
    size_t                         exploding_gas_output_buffer_size = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group;
    OptixProgramGroup              miss_group = 0;
    OptixProgramGroup              hit_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params*                        d_params;

    float                          time = 0.f;
    float                          last_exploding_sphere_rebuild_time = 0.f;

    OptixShaderBindingTable        sbt = {};
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t g_tessellation_resolution = 128;

const float g_exploding_gas_rebuild_frequency = 10.f;

const int32_t INST_COUNT = 4;

const std::array<float3, INST_COUNT> g_diffuse_colors =
{ {
    { 0.70f, 0.70f, 0.70f },
    { 0.80f, 0.80f, 0.80f },
    { 0.90f, 0.90f, 0.90f },
    { 1.00f, 1.00f, 1.00f }
} };

struct Instance
{
    float m[12];
};

const std::array<Instance, INST_COUNT> g_instances =
{ {
    {{1, 0, 0, -4.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
    {{1, 0, 0, -1.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
    {{1, 0, 0,  1.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
    {{1, 0, 0,  4.5f, 0, 1, 0, 0, 0, 0, 1, 0}}
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
        trackball.startTracking( static_cast< int >( xpos ), static_cast< int >( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast< Params* >( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast< int >( xpos ), static_cast< int >( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast< int >( xpos ), static_cast< int >( ypos ), params->width, params->height );
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

    Params* params = static_cast< Params* >( glfwGetWindowUserPointer( window ) );
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


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
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( ( int )yscroll ) )
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
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --time | -t                 Animation time for image output (default 1)\n";
    std::cerr << "         --frames | -n               Number of animation frames for image output (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 1024x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( DynamicGeometryState& state )
{
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped
    state.params.subframe_index = 0u;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_params ), sizeof( Params ) ) );
}


void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast< float >( params.width ) / static_cast< float >( params.height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, DynamicGeometryState& state )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast< void* >( state.d_params ),
        &state.params, sizeof( Params ),
        cudaMemcpyHostToDevice, state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast< CUdeviceptr >( state.d_params ),
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
    camera.setEye( make_float3( 0.f, 1.f, -20.f ) );
    camera.setLookat( make_float3( 0, 0, 0 ) );
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


void createContext( DynamicGeometryState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}

void launchGenerateAnimatedVertices( DynamicGeometryState& state, AnimationMode animation_mode )
{
    generateAnimatedVetrices( (float3*)state.d_temp_vertices, animation_mode, state.time, g_tessellation_resolution, g_tessellation_resolution );
}

void updateMeshAccel( DynamicGeometryState& state )
{
    // Generate deformed sphere vertices
    launchGenerateAnimatedVertices( state, AnimationMode_Deform );

    // Update deforming GAS

    OptixAccelBuildOptions gas_accel_options = {};
    gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        state.stream,                       // CUDA stream
        &gas_accel_options,
        &state.triangle_input,
        1,                                  // num build inputs
        state.d_temp_buffer,
        state.temp_buffer_size,
        state.d_deforming_gas_output_buffer,
        state.deforming_gas_output_buffer_size,
        &state.deforming_gas_handle,
        nullptr,                           // emitted property list
        0                                   // num emitted properties
    ) );

    // Generate exploding sphere vertices
    launchGenerateAnimatedVertices( state, AnimationMode_Explode );

    // Update exploding GAS

    // Occasionally rebuild to maintain AS quality
    if( state.time - state.last_exploding_sphere_rebuild_time > 1 / g_exploding_gas_rebuild_frequency )
    {
        gas_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        state.last_exploding_sphere_rebuild_time = state.time;

        // We don't compress the AS so the size of the GAS won't change and we can rebuild the GAS in-place.
    }

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        state.stream,                       // CUDA stream
        &gas_accel_options,
        &state.triangle_input,
        1,                                  // num build inputs
        state.d_temp_buffer,
        state.temp_buffer_size,
        state.d_exploding_gas_output_buffer,
        state.exploding_gas_output_buffer_size,
        &state.exploding_gas_handle,
        nullptr,                           // emitted property list
        0                                   // num emitted properties
    ) );

    // Update the IAS
    // We refit the IAS as the relative positions of the spheres don't change much so AS quality after update is fine.

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    if( g_instances.size() > 1 )
    {
        float t = sinf( state.time * 4.f );
        CUDA_CHECK( cudaMemcpy( ( ( OptixInstance* )state.d_instances )[1].transform + 7, &t, sizeof( float ), cudaMemcpyHostToDevice ) );
    }

    OPTIX_CHECK( optixAccelBuild( state.context, state.stream, &ias_accel_options, &state.ias_instance_input, 1, state.d_temp_buffer, state.temp_buffer_size,
        state.d_ias_output_buffer, state.ias_output_buffer_size, &state.ias_handle, nullptr, 0 ) );

    CUDA_SYNC_CHECK();
}

void buildMeshAccel( DynamicGeometryState& state )
{
    // Allocate temporary space for vertex generation.
    // The same memory space is reused for generating the deformed and exploding vertices before updates.
    uint32_t numVertices = g_tessellation_resolution * g_tessellation_resolution * 6;
    const size_t vertices_size_in_bytes = numVertices * sizeof( float3 );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_temp_vertices ), vertices_size_in_bytes ) );

    // Build static triangulated sphere.
    launchGenerateAnimatedVertices( state, AnimationMode_None );

    // Build an AS over the triangles.
    // We use un-indexed triangles so we can explode the sphere per triangle.
    state.triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    state.triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    state.triangle_input.triangleArray.vertexStrideInBytes = sizeof( float3 );
    state.triangle_input.triangleArray.numVertices = static_cast< uint32_t >( numVertices );
    state.triangle_input.triangleArray.vertexBuffers = &state.d_temp_vertices;
    state.triangle_input.triangleArray.flags = &state.triangle_flags;
    state.triangle_input.triangleArray.numSbtRecords = 1;
    state.triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
    state.triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    state.triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &state.triangle_input,
        1,  // num_build_inputs
        &gas_buffer_sizes
    ) );

    state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >( &d_buffer_temp_output_gas_and_compacted_size ),
        compactedSizeOffset + 8
    ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( ( char* )d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0,                                  // CUDA stream
        &accel_options,
        &state.triangle_input,
        1,                                  // num build inputs
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.static_gas_handle,
        &emitProperty,                      // emitted property list
        1                                   // num emitted properties
    ) );

    // Replicate the uncompressed GAS for the exploding sphere.
    // The exploding sphere is occasionally rebuild. We don't want to compress the GAS after every rebuild so we use the uncompressed GAS for the exploding sphere.
    // The memory requirements for the uncompressed exploding GAS won't change so we can rebuild in-place.
    state.exploding_gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    OptixAccelRelocationInfo relocationInfo;
    OPTIX_CHECK( optixAccelGetRelocationInfo( state.context, state.static_gas_handle, &relocationInfo ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_exploding_gas_output_buffer ), state.exploding_gas_output_buffer_size ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_exploding_gas_output_buffer, ( const void* )d_buffer_temp_output_gas_and_compacted_size, state.exploding_gas_output_buffer_size, cudaMemcpyDeviceToDevice ) );
    OPTIX_CHECK( optixAccelRelocate( state.context, 0, &relocationInfo, 0, 0, state.d_exploding_gas_output_buffer, state.exploding_gas_output_buffer_size, &state.exploding_gas_handle ) );

    // Compress GAS

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, ( void* )emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_static_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.static_gas_handle, state.d_static_gas_output_buffer, compacted_gas_size, &state.static_gas_handle ) );

        CUDA_CHECK( cudaFree( ( void* )d_buffer_temp_output_gas_and_compacted_size ) );

        state.static_gas_output_buffer_size = compacted_gas_size;
    }
    else
    {
        state.d_static_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

        state.static_gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
    }

    // Replicate the compressed GAS for the deforming sphere.
    // The deforming sphere is never rebuild so we refit the compressed GAS without requiring recompression.
    state.deforming_gas_output_buffer_size = state.static_gas_output_buffer_size;
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_deforming_gas_output_buffer ), state.deforming_gas_output_buffer_size ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_deforming_gas_output_buffer, ( const void* )state.d_static_gas_output_buffer, state.deforming_gas_output_buffer_size, cudaMemcpyDeviceToDevice ) );
    OPTIX_CHECK( optixAccelRelocate( state.context, 0, &relocationInfo, 0, 0, state.d_deforming_gas_output_buffer, state.deforming_gas_output_buffer_size, &state.deforming_gas_handle ) );

    // Build the IAS

    std::vector<OptixInstance> instances( g_instances.size() );

    for( size_t i = 0; i < g_instances.size(); ++i )
    {
        memcpy( instances[i].transform, g_instances[i].m, sizeof( float ) * 12 );
        instances[i].sbtOffset = static_cast< unsigned int >( i );
        instances[i].visibilityMask = 255;
    }

    instances[0].traversableHandle = state.static_gas_handle;
    instances[1].traversableHandle = state.static_gas_handle;
    instances[2].traversableHandle = state.deforming_gas_handle;
    instances[3].traversableHandle = state.exploding_gas_handle;

    size_t      instances_size_in_bytes = sizeof( OptixInstance ) * instances.size();
    CUDA_CHECK( cudaMalloc( ( void** )&state.d_instances, instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice ) );

    state.ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    state.ias_instance_input.instanceArray.instances = state.d_instances;
    state.ias_instance_input.instanceArray.numInstances = static_cast<int>( instances.size() );

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &ias_accel_options, &state.ias_instance_input, 1, &ias_buffer_sizes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    compactedSizeOffset = roundUp<size_t>( ias_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_ias_and_compacted_size ), compactedSizeOffset + 8 ) );

    CUdeviceptr d_ias_temp_buffer;
    bool        needIASTempBuffer = ias_buffer_sizes.tempSizeInBytes > state.temp_buffer_size;
    if( needIASTempBuffer )
    {
        CUDA_CHECK( cudaMalloc( (void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes ) );
    }
    else
    {
        d_ias_temp_buffer = state.d_temp_buffer;
    }

    emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( state.context, 0, &ias_accel_options, &state.ias_instance_input, 1, d_ias_temp_buffer,
                                  ias_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_ias_and_compacted_size,
                                  ias_buffer_sizes.outputSizeInBytes, &state.ias_handle, &emitProperty, 1 ) );

    if( needIASTempBuffer )
    {
        CUDA_CHECK( cudaFree( (void*)d_ias_temp_buffer ) );
    }

    // Compress the IAS

    size_t compacted_ias_size;
    CUDA_CHECK( cudaMemcpy( &compacted_ias_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_ias_size < ias_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), compacted_ias_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.ias_handle, state.d_ias_output_buffer,
                                        compacted_ias_size, &state.ias_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_ias_and_compacted_size ) );

        state.ias_output_buffer_size = compacted_ias_size;
    }
    else
    {
        state.d_ias_output_buffer = d_buffer_temp_output_ias_and_compacted_size;

        state.ias_output_buffer_size = ias_buffer_sizes.outputSizeInBytes;
    }

    // allocate enough temporary update space for updating the deforming GAS, exploding GAS and IAS.
    size_t maxUpdateTempSize = std::max( ias_buffer_sizes.tempUpdateSizeInBytes, gas_buffer_sizes.tempUpdateSizeInBytes );
    if( state.temp_buffer_size < maxUpdateTempSize )
    {
        CUDA_CHECK( cudaFree( (void*)state.d_temp_buffer ) );
        state.temp_buffer_size = maxUpdateTempSize;
        CUDA_CHECK( cudaMalloc( (void**)&state.d_temp_buffer, state.temp_buffer_size ) );
    }

    state.params.handle = state.ias_handle;
}


void createModule( DynamicGeometryState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    state.pipeline_compile_options.numPayloadValues = 3;
    state.pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixDynamicGeometry.cu", inputSize );

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.ptx_module
    ) );
}


void createProgramGroups( DynamicGeometryState& state )
{
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
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
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log, &sizeof_log,
            &state.miss_group
        ) );
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &state.hit_group
        ) );
    }
}


void createPipeline( DynamicGeometryState& state )
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.miss_group,
        state.hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

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
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.raygen_prog_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.miss_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.hit_group, &stack_sizes ) );

    uint32_t max_trace_depth = 1;
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

    // This is 2 since the largest depth is IAS->GAS
    const uint32_t max_traversable_graph_depth = 2;

    OPTIX_CHECK( optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversable_graph_depth
    ) );
}


void createSBT( DynamicGeometryState& state )
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_raygen_record ),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ) );

    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &d_miss_records ), miss_record_size ) );

    MissRecord ms_sbt[1];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_group, &ms_sbt[0] ) );
    ms_sbt[0].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_miss_records ),
        ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ) );

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >( &d_hitgroup_records ),
        hitgroup_record_size * g_instances.size()
    ) );

    std::vector<HitGroupRecord> hitgroup_records( g_instances.size() );
    for( int i = 0; i < static_cast<int>( g_instances.size() ); ++i )
    {
        const int sbt_idx = i;

        OPTIX_CHECK( optixSbtRecordPackHeader( state.hit_group, &hitgroup_records[sbt_idx] ) );
        hitgroup_records[sbt_idx].data.color = g_diffuse_colors[i];
    }

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_hitgroup_records ),
        hitgroup_records.data(),
        hitgroup_record_size*hitgroup_records.size(),
        cudaMemcpyHostToDevice
    ) );

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast< uint32_t >( miss_record_size );
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast< uint32_t >( hitgroup_record_size );
    state.sbt.hitgroupRecordCount = static_cast< uint32_t >( hitgroup_records.size() );
}


void cleanupState( DynamicGeometryState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hit_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_temp_vertices ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_static_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_deforming_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_exploding_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_instances ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_ias_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_params ) ) );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    DynamicGeometryState state;
    state.params.width  = 1024;
    state.params.height = 768;
    state.time = 0.f;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    int num_frames = 16;
    float animation_time = 1.f;

    //
    // Parse command line options
    //
    std::string outfile;

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
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int               w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else if( arg == "--time" || arg == "-t" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            animation_time = (float)atof( argv[++i] );
        }
        else if( arg == "--frames" || arg == "-n" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            num_frames = atoi( argv[++i] );
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

        createModule( state );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );
        initLaunchParams( state );

        buildMeshAccel( state );

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixDynamicGeometry", state.params.width, state.params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &state.params );

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

                auto tstart = std::chrono::system_clock::now();

                state.last_exploding_sphere_rebuild_time = 0.f;

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    auto tnow = std::chrono::system_clock::now();
                    std::chrono::duration<double> time = tnow - tstart;
                    state.time = (float)time.count();

                    updateMeshAccel( state );

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

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            state.last_exploding_sphere_rebuild_time = 0.f;

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                state.params.width,
                state.params.height
            );

            handleCameraUpdate( state.params );
            handleResize( output_buffer, state.params );

            // run animation frames
            for( unsigned int i = 0; i < static_cast<unsigned int>( num_frames ); ++i )
            {
                state.time = i * ( animation_time / ( num_frames - 1 ) );
                updateMeshAccel( state );
                launchSubframe( output_buffer, state );
            }

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

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
