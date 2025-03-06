//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"  // Needs to be included before gl_interop
#undef TINYOBJLOADER_IMPLEMENTATION

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Aabb.h>
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

#include "optixMotionGeometry.h"
#include "vertices.h"
#include "motionHelper.hpp"

#include <cstdlib>
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
//
//------------------------------------------------------------------------------

class ObjMesh
{
public:
    ~ObjMesh() { destroy(); }

    void load( const OptixDeviceContext context, const std::string& fileName, const std::vector<std::string>& allowMeshNames, const std::vector<std::string>& denyMeshNames )
    {
        destroy();

        tinyobj::attrib_t                attrib;
        std::vector<tinyobj::shape_t>    shapes;
        std::vector<tinyobj::material_t> materials;

        // load object
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj( &attrib, &shapes, &materials, &warn, &err, fileName.c_str(), nullptr, true );
        if( !warn.empty() )
            std::cout << "obj WARNING: " << warn << std::endl;
        if( !ret )
        {
            std::cout << "Failed to load ObjMesh file'" << fileName << "': " << err << std::endl;
            throw sutil::Exception( err.c_str() );
        }

        // count indices
        size_t numVertexIndices = 0;
        for( const auto& shape : shapes )
        {
            numVertexIndices += shape.mesh.vertex_indices.size();
        }

        // allocate cuda memory
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_indexBuffer ), sizeof( unsigned int ) * numVertexIndices ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_vertexBuffer ), sizeof( float ) * attrib.vertices.size() ) );
        CUDA_CHECK( cudaMemcpy( (void*)m_vertexBuffer, (void*)attrib.vertices.data(),
            sizeof( float ) * attrib.vertices.size(), cudaMemcpyHostToDevice ) );

        //////////////////////////////////////////////////////////////////////////
        // no need for these attributes
        //if( attrib.normals.size() )
        //{
        //    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_normalIndexBuffer ), sizeof( unsigned int ) * numVetrexIndices ) );
        //    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_normalBuffer ), sizeof( float ) * attrib.normals.size() ) );
        //    CUDA_CHECK( cudaMemcpy( (void*)m_normalBuffer, (void*)attrib.normals.data(),
        //        sizeof( float ) * attrib.normals.size(), cudaMemcpyHostToDevice ) );
        //}

        //if( attrib.texcoords.size() )
        //{
        //    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_texcoordsIndexBuffer ), sizeof( unsigned int ) * numVetrexIndices ) );
        //    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_texcoordsBuffer ), sizeof( float ) * attrib.texcoords.size() ) );
        //    CUDA_CHECK( cudaMemcpy( (void*)m_texcoordsBuffer, (void*)attrib.texcoords.data(),
        //        sizeof( float ) * attrib.texcoords.size(), cudaMemcpyHostToDevice ) );
        //}
        //////////////////////////////////////////////////////////////////////////

        // setup build inputs
        numVertexIndices = 0;

        unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        for( auto& shape : shapes )
        {
            if( !allowMeshNames.empty() && std::find(allowMeshNames.begin(), allowMeshNames.end(), shape.name) == allowMeshNames.end() )
                continue;
            if( std::find( denyMeshNames.begin(), denyMeshNames.end(), shape.name ) != denyMeshNames.end() )
                continue;

            OptixBuildInput buildInput = {};

            buildInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            buildInput.triangleArray.numVertices   = static_cast<unsigned int>( attrib.vertices.size() ) / 3;
            buildInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            buildInput.triangleArray.vertexBuffers = &m_vertexBuffer;

            buildInput.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>( shape.mesh.vertex_indices.size() ) / 3;
            buildInput.triangleArray.indexBuffer      = m_indexBuffer + numVertexIndices * sizeof( unsigned int );
            CUdeviceptr normalIndexBuffer             = m_normalIndexBuffer + numVertexIndices * sizeof( unsigned int );
            CUdeviceptr texcoordIndexBuffer = m_texcoordsIndexBuffer + numVertexIndices * sizeof( unsigned int );
            buildInput.triangleArray.indexStrideInBytes   = 0;
            buildInput.triangleArray.primitiveIndexOffset = static_cast<unsigned int>( numVertexIndices ) / 3;

            CUDA_CHECK( cudaMemcpy( (void*)buildInput.triangleArray.indexBuffer, (void*)shape.mesh.vertex_indices.data(),
                sizeof( unsigned int ) * shape.mesh.vertex_indices.size(), cudaMemcpyHostToDevice ) );

            if( m_normalIndexBuffer != 0 && shape.mesh.normal_indices.size() != 0 )
                CUDA_CHECK( cudaMemcpy( (void*)normalIndexBuffer, (void*)shape.mesh.normal_indices.data(),
                    sizeof( unsigned int ) * shape.mesh.normal_indices.size(), cudaMemcpyHostToDevice ) );

            if( m_texcoordsIndexBuffer != 0 && shape.mesh.texcoord_indices.size() != 0 )
                CUDA_CHECK( cudaMemcpy( (void*)texcoordIndexBuffer, (void*)shape.mesh.texcoord_indices.data(),
                    sizeof( unsigned int ) * shape.mesh.texcoord_indices.size(), cudaMemcpyHostToDevice ) );

            buildInput.triangleArray.numSbtRecords = 1;
            buildInput.triangleArray.flags         = &flags;

            m_buildInputs.push_back( buildInput );

            numVertexIndices += shape.mesh.vertex_indices.size();
        }

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, m_buildInputs.data(),
            static_cast<unsigned int>( m_buildInputs.size() ),  // num_build_inputs
            &gas_buffer_sizes ) );

        // allocate tmp memory
        CUdeviceptr d_temp_buffer    = 0;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

#if 0
        // allocate non-compacted output + compacted size
        size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_accelBuffer ), compactedSizeOffset + 8 ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = ( CUdeviceptr )( (char*)m_accelBuffer + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( context,
            0,  // CUDA stream
            &accel_options, m_buildInputs.data(),
            static_cast<unsigned int>( m_buildInputs.size() ),  // num_build_inputs
            d_temp_buffer, gas_buffer_sizes.tempSizeInBytes, m_accelBuffer,
            gas_buffer_sizes.outputSizeInBytes, &m_gas_handle,
            &emitProperty,  // emitted property list
            1               // num emitted properties
        ) );
#else
        // allocate non-compacted output + compacted size
        size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        size_t aabbOffset = compactedSizeOffset + 8;
        size_t totalSize = aabbOffset + 24;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_accelBuffer ), totalSize ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = ( CUdeviceptr )( (char*)m_accelBuffer + compactedSizeOffset );
        {
            OptixAccelEmitDesc emitProperties[2] = {};
            emitProperties[0].type             = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperties[0].result           = ( CUdeviceptr )( (char*)m_accelBuffer + compactedSizeOffset );
            emitProperties[1].type             = OPTIX_PROPERTY_TYPE_AABBS;
            emitProperties[1].result           = ( CUdeviceptr )( (char*)m_accelBuffer + aabbOffset );

            OPTIX_CHECK( optixAccelBuild( context,
                                          0,  // CUDA stream
                                          &accel_options, m_buildInputs.data(),
                                          static_cast<unsigned int>( m_buildInputs.size() ),  // num_build_inputs
                                          d_temp_buffer, gas_buffer_sizes.tempSizeInBytes, m_accelBuffer,
                                          gas_buffer_sizes.outputSizeInBytes, &m_gas_handle,
                                          emitProperties,  // emitted property list
                                          2              // num emitted properties
                                          ) );
        }

        CUDA_CHECK( cudaMemcpy( gasAabb.data(), (char*)m_accelBuffer + aabbOffset, 24, cudaMemcpyDeviceToHost ) );
#endif

        CUDA_CHECK( cudaFree( (void*)d_temp_buffer ) );

        // Compress GAS

        size_t compacted_gas_size;
        CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            CUdeviceptr uncompactedAccel = m_accelBuffer;

            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_accelBuffer ), compacted_gas_size ) );

            // use handle as input and output
            OPTIX_CHECK( optixAccelCompact( context, 0, m_gas_handle, m_accelBuffer, compacted_gas_size, &m_gas_handle ) );

            CUDA_CHECK( cudaFree( (void*)uncompactedAccel ) );
        }
    }

    OptixTraversableHandle getHandle() { return m_gas_handle; }

    size_t getNumSbtRecords() { return m_buildInputs.size(); }

    const sutil::Aabb& meshAabb() const
    {
        return gasAabb;
    }

    void destroy()
    {
        if( m_indexBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_indexBuffer ) );
            m_indexBuffer = 0;
        }

        if( m_vertexBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_vertexBuffer ) );
            m_vertexBuffer = 0;
        }

        if( m_normalBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_normalBuffer ) );
            m_normalBuffer = 0;
        }

        if( m_normalIndexBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_normalIndexBuffer ) );
            m_normalIndexBuffer = 0;
        }

        if( m_texcoordsBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_texcoordsBuffer ) );
            m_texcoordsBuffer = 0;
        }

        if( m_texcoordsIndexBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_texcoordsIndexBuffer ) );
            m_texcoordsIndexBuffer = 0;
        }

        if( m_accelBuffer )
        {
            CUDA_CHECK( cudaFree( (void*)m_accelBuffer ) );
            m_accelBuffer = 0;
        }

        m_buildInputs.clear();
    }

private:
    sutil::Aabb gasAabb;

    OptixTraversableHandle       m_gas_handle = 0;
    std::vector<OptixBuildInput> m_buildInputs;

    CUdeviceptr m_indexBuffer          = 0;
    CUdeviceptr m_vertexBuffer         = 0;
    CUdeviceptr m_normalIndexBuffer    = 0;
    CUdeviceptr m_normalBuffer         = 0;
    CUdeviceptr m_texcoordsIndexBuffer = 0;
    CUdeviceptr m_texcoordsBuffer      = 0;
    CUdeviceptr m_accelBuffer          = 0;
};

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

class ExhaustFume
{
public:
    ExhaustFume()
    {}

    ExhaustFume(ExhaustFume&& other)
    {
        swap( other );
    }

    ExhaustFume& operator=(ExhaustFume&& other)
    {
        swap( other );
        return *this;
    }

    void swap( ExhaustFume& other )  // nothrow
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two objects,
        // the two objects are effectively swapped
        swap(d_exploding_gas_output_buffer, other.d_exploding_gas_output_buffer);
        swap(exploding_gas_handle, other.exploding_gas_handle);
        swap( srt_animation, other.srt_animation );
        swap( d_srt, other.d_srt );
        swap( timeLastASRebuild, other.timeLastASRebuild );
        swap( localTime, other.localTime );
        swap( rotationSpeed, other.rotationSpeed );
        swap( lastRotationDegree, other.lastRotationDegree );
        swap( relativeEjectionSpeed, other.relativeEjectionSpeed );
        swap( remove, other.remove );
        swap( baseP, other.baseP );
    }
    ~ExhaustFume()
    {
        cudaFree( (void*)d_exploding_gas_output_buffer );
        cudaFree( (void*)d_srt );
    }

    CUdeviceptr             d_exploding_gas_output_buffer = 0;
    OptixTraversableHandle  exploding_gas_handle          = 0;
    SRTMotionTransformArray srt_animation;
    CUdeviceptr             d_srt = 0;
    float                   timeLastASRebuild = 0.f;
    float                   localTime = 0.f;
    float                   rotationSpeed = 0.f;  // rotation speed of plane at time of ejection
    float                   lastRotationDegree = 0.f;
    float  relativeEjectionSpeed = 1;  // relative ejection speed to give the different fumes some variation
    bool   remove                = false;
    float3 baseP;
};

struct MotionGeometryState
{
    OptixDeviceContext context = 0;

    size_t                         temp_buffer_size = 0;
    CUdeviceptr                    d_temp_buffer = 0;
    CUdeviceptr                    d_temp_vertices[2] = {};
    CUdeviceptr                    d_instances = 0;
    size_t                         d_instances_size = 0;

    unsigned int                   triangle_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    OptixBuildInput                ias_instance_input = {};
    OptixBuildInput                triangle_input = {};
    OptixBuildInput                triangle_input_fume = {};

    OptixAccelBuildOptions         ias_accel_options ={};
    OptixTraversableHandle         ias_handle;
    OptixTraversableHandle         static_gas_handle;
    OptixTraversableHandle         deforming_gas_handle;
    OptixTraversableHandle         exploding_gas_handle;

    ObjMesh                        plane;
    ObjMesh                        planePropeller;

    CUdeviceptr                    d_ias_output_buffer = 0;
    CUdeviceptr                    d_static_gas_output_buffer = 0;
    CUdeviceptr                    d_deforming_gas_output_buffer = 0;

    size_t                         ias_output_buffer_size = 0;
    size_t                         static_gas_output_buffer_size = 0;
    size_t                         deforming_gas_output_buffer_size = 0;
    size_t                         exploding_gas_output_buffer_size = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group = 0;
    OptixProgramGroup              miss_group = 0;
    OptixProgramGroup              miss_group_occlusion = 0;
    OptixProgramGroup              hit_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params*                        d_params;

    float                          time = 0.f;
    float                          time_last_frame = 0.f;
    float                          time_last_fume = 0.f;
    float                          targetFrameTime = 0.0333f;

    OptixShaderBindingTable        sbt = {};

    std::vector<OptixInstance>     instances;
    std::vector<ExhaustFume>       fume;

    bool followPlane = false;
    bool renderAO = true;
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t g_tessellation_resolution = 128;
const int32_t g_tessellation_resolution_fume = g_tessellation_resolution / 8;

const float g_exploding_gas_rebuild_frequency = 10.f;

const int32_t INST_COUNT = 4;

struct Instance
{
    Matrix3x4 m;
};

struct PlaneAnimation {
    SRTMotionTransformArray srt_animation;
    SRTMotionTransformArray srt_animationPropeller;
    CUdeviceptr             d_srts;
    CUdeviceptr             d_srtsPropeller;
    float                   planeSpeed = 0.4f;
    float                   lastRotationDegree = 120;
} plane;

struct DeformSphereAnimation {
    MatrixMotionTransformArray matrix_animation;
    CUdeviceptr                d_matrices;
    float                      rotationSpeed = 0.1f;
    float                      lastRotationDegree = 0;
} deformSphere;

void addFume( MotionGeometryState& state );
void createModule( MotionGeometryState& state );
void createProgramGroups( MotionGeometryState& state );
void createPipeline( MotionGeometryState& state );
void buildMeshAccel( MotionGeometryState& state );
void createSBT( MotionGeometryState& state );


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
    Params& params = static_cast<MotionGeometryState*>( glfwGetWindowUserPointer( window ) )->params;

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast< int >( xpos ), static_cast< int >( ypos ), params.width, params.height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast< int >( xpos ), static_cast< int >( ypos ), params.width, params.height );
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

    Params& params = static_cast< MotionGeometryState* >( glfwGetWindowUserPointer( window ) )->params;
    params.width = res_x;
    params.height = res_y;
    camera_changed = true;
    resize_dirty = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    MotionGeometryState& state = *static_cast<MotionGeometryState*>( glfwGetWindowUserPointer( window ) );
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
    else if( key == GLFW_KEY_DOWN )
    {
        state.targetFrameTime *= 2;
    }
    else if( key == GLFW_KEY_UP )
    {
        state.targetFrameTime /= 2;
    }
    else if( key == GLFW_KEY_M )
    {
        plane.planeSpeed *= 1.5f;
    }
    else if( key == GLFW_KEY_N )
    {
        plane.planeSpeed /= 1.5f;
    }
    else if( key == GLFW_KEY_J )
    {
        deformSphere.rotationSpeed *= 1.5f;
    }
    else if( key == GLFW_KEY_H )
    {
        deformSphere.rotationSpeed /= 1.5f;
    }
    else if( key == GLFW_KEY_V )
    {
        state.followPlane = !state.followPlane;
    }
    else if( key == GLFW_KEY_B )
    {
        addFume( state );
    }
    else if( key == GLFW_KEY_A )
    {
        state.renderAO = !state.renderAO;
        createModule( state );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );

        state.params.ao = state.renderAO;
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


void initLaunchParams( MotionGeometryState& state )
{
    state.params.frame_buffer   = nullptr;  // Will be set when output buffer is mapped
    state.params.subframe_index = 0u;
    state.params.spp            = 1u;
    state.params.ao             = true;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_params ), sizeof( Params ) ) );
}

float3 getPlaneWSPos( const MotionGeometryState& state )
{
    OptixSRTData lerped = lerp( plane.srt_animation.motionData( 0, 0 ), plane.srt_animation.motionData( 0, 1 ), 0.5f );
    Matrix3x4    m;
    srtToMatrix( lerped, m.m );
    return m * state.plane.meshAabb().center();
}


void handleCameraUpdate( MotionGeometryState& state )
{
    if( state.followPlane )
    {
        float3 l = camera.lookat();
        float3 planePos = getPlaneWSPos( state );
        float3 offset = planePos - l;
        camera.setLookat( planePos );
        camera.setEye( camera.eye() + offset );
        trackball.reinitOrientationFromCamera();
        camera_changed = true;
    }
    if( !camera_changed )
        return;
    camera_changed = false;

    Params& params = state.params;
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



void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, MotionGeometryState& state )
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
    camera.setEye( make_float3( -7.f, 3.f, -5.f ) );
    camera.setLookat( make_float3( 0 ) );
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


void createContext( MotionGeometryState& state )
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

void launchGenerateAnimatedVertices( MotionGeometryState& state, AnimationMode animation_mode, float time_last_frame, float time_now, int tessellation_resolution )
{
    generateAnimatedVetrices( (float3*)state.d_temp_vertices[0], animation_mode, time_last_frame, tessellation_resolution, tessellation_resolution );
    generateAnimatedVetrices( (float3*)state.d_temp_vertices[1], animation_mode, time_now, tessellation_resolution, tessellation_resolution );
}

float randf()
{
    return static_cast<float>( rand() ) / static_cast<float>( RAND_MAX );
}

void addFume( MotionGeometryState& state )
{
    ExhaustFume fume ={};

    fume.baseP = getPlaneWSPos( state );
    fume.rotationSpeed = plane.planeSpeed * (randf() * 0.1f + 0.8f);
    fume.lastRotationDegree = plane.lastRotationDegree;
    fume.relativeEjectionSpeed = randf() * 0.4f + 0.6f;

    // using an array is overkill here, but 
    fume.srt_animation = SRTMotionTransformArray( 1, 2 );
    CUDA_CHECK( cudaMalloc( (void**)&fume.d_srt, fume.srt_animation.byteSize() ) );

    OptixSRTMotionTransform& t = fume.srt_animation.transform( 0 );
    t.motionOptions.flags      = 0;
    t.motionOptions.numKeys    = fume.srt_animation.numKeys();
    t.motionOptions.timeBegin  = 0;
    t.motionOptions.timeEnd    = 1;

    fume.srt_animation.motionData( 0, 0 ) = plane.srt_animation.motionData( 0, 1 );
    fume.srt_animation.motionData( 0, 1 ) = plane.srt_animation.motionData( 0, 1 );

    //// Generate exploding sphere vertices
    launchGenerateAnimatedVertices( state, AnimationMode_None, 0, 0, g_tessellation_resolution_fume );

    OptixAccelBuildOptions gas_accel_options = {};
    gas_accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE
                                               | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    gas_accel_options.operation               = OPTIX_BUILD_OPERATION_BUILD;
    gas_accel_options.motionOptions.numKeys   = 2;
    gas_accel_options.motionOptions.timeBegin = 0;
    gas_accel_options.motionOptions.timeEnd   = 1;

    OptixAccelBufferSizes s;
    optixAccelComputeMemoryUsage( state.context, &gas_accel_options, &state.triangle_input_fume, 1, &s );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &fume.d_exploding_gas_output_buffer ), state.exploding_gas_output_buffer_size ) );

    OPTIX_CHECK( optixAccelBuild( state.context,
                                  state.stream,  // CUDA stream
                                  &gas_accel_options, &state.triangle_input_fume,
                                  1,  // num build inputs
                                  state.d_temp_buffer, state.temp_buffer_size,
                                  fume.d_exploding_gas_output_buffer, state.exploding_gas_output_buffer_size,
                                  &fume.exploding_gas_handle,
                                  nullptr, 0  // emitted property list
                                  ) );

    state.fume.emplace_back( std::move( fume ) );
}

void updateMeshAccel( MotionGeometryState& state )
{
    // Generate deformed sphere vertices
    launchGenerateAnimatedVertices( state, AnimationMode_Deform, state.time_last_frame, state.time, g_tessellation_resolution );

    // Update deforming GAS

    OptixAccelBuildOptions gas_accel_options = {};
    gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    gas_accel_options.motionOptions.numKeys   = 2;
    gas_accel_options.motionOptions.timeBegin = 0;
    gas_accel_options.motionOptions.timeEnd   = 1;

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

#if 1
    {
        float timePassed = state.time - state.time_last_frame;
        for( size_t i=0; i<state.fume.size(); ++i )
        {
            ExhaustFume& fume = state.fume[i];
            if( fume.remove )
            {
                state.fume.erase( state.fume.begin() + i );
                i--;
                continue;
            }

            OptixAccelBuildOptions fume_gas_accel_options = gas_accel_options;
            fume_gas_accel_options.operation              = OPTIX_BUILD_OPERATION_UPDATE;

            // Generate exploding sphere vertices
            // and update the movement (instance) animation
            const float animationLength = M_PIf * 0.5f;
            const float timeOffset = M_PIf;
            const float maxTime = animationLength + timeOffset;
            const float localTime = fume.localTime + timeOffset;

            if( localTime + timePassed >= maxTime )
            {
                launchGenerateAnimatedVertices( state, AnimationMode_Explode, localTime, maxTime, g_tessellation_resolution_fume );

                fume_gas_accel_options.operation             = OPTIX_BUILD_OPERATION_BUILD;
                fume_gas_accel_options.motionOptions.timeEnd = ( maxTime - localTime ) / timePassed;
                fume_gas_accel_options.buildFlags |= OPTIX_MOTION_FLAG_END_VANISH;

                OptixSRTMotionTransform& t = fume.srt_animation.transform( 0 );
                t.motionOptions.timeEnd = ( maxTime - localTime ) / timePassed;
                t.motionOptions.flags |= OPTIX_MOTION_FLAG_END_VANISH;

                fume.remove = true;
            }
            else
            {
                launchGenerateAnimatedVertices( state, AnimationMode_Explode, localTime, localTime + timePassed, g_tessellation_resolution_fume );
            }

            {
                float3 scale           = make_float3( 0.05f );
                float3 shear           = make_float3( 0 );
                float3 scaleShearPivot = make_float3( 0 );
                float3 rotationPivot = make_float3( 0, -1.2f, 0.0f );
                float3 translation   = make_float3( 0, 1.2f, 0.0f );
                float3 ejectTrans    = fume.relativeEjectionSpeed * make_float3( 0, 0.2f, 0 );

                float rotations = fume.rotationSpeed * timePassed * 360;

                float oldRot = fume.lastRotationDegree;
                float newRot = oldRot + rotations;
                if( newRot >= 360 )
                {
                    oldRot -= 360;
                    newRot -= 360;
                }
                fume.lastRotationDegree = newRot;

                fume.srt_animation.motionData( 0, 0 ) =
                    buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 1, 0, 0 ), oldRot ), rotationPivot - fume.localTime * ejectTrans,
                              translation + fume.localTime * ejectTrans );
                fume.srt_animation.motionData( 0, 1 ) =
                    buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 1, 0, 0 ), newRot ), rotationPivot - ( fume.localTime + timePassed ) * ejectTrans,
                              translation + ( fume.localTime + timePassed ) * ejectTrans );

            }

            // Occasionally rebuild to maintain AS quality
            if( fume.localTime + timePassed - fume.timeLastASRebuild > 1 / g_exploding_gas_rebuild_frequency )
            {
                fume_gas_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
                fume.timeLastASRebuild           = fume.localTime + timePassed;
            }
            fume.localTime += timePassed;

            OptixAccelBufferSizes s;
            optixAccelComputeMemoryUsage( state.context, &fume_gas_accel_options, &state.triangle_input_fume, 1, &s );

            OPTIX_CHECK( optixAccelBuild( state.context, state.stream,
                                          &fume_gas_accel_options, &state.triangle_input_fume,
                                          1,  // num build inputs
                                          state.d_temp_buffer, state.temp_buffer_size,
                                          fume.d_exploding_gas_output_buffer, state.exploding_gas_output_buffer_size,
                                          &fume.exploding_gas_handle,
                                          nullptr, 0  // emitted property list
                                          ) );
        }
    }
#endif

    // Update the IAS
    // We refit the IAS as the relative positions of the spheres don't change much so AS quality after update is fine.

#if 1
    {
        {
            float3 scale           = make_float3( 0.2f );
            float3 shear           = make_float3( 0 );
            float3 scaleShearPivot = make_float3( 0 );
            float3 rotationPivot   = make_float3( 3, 0, 0 );
            float3 translation     = -rotationPivot;

            float rotationsPerSecond = deformSphere.rotationSpeed;
            float timePassed         = state.time - state.time_last_frame;
            float rotations          = rotationsPerSecond * timePassed * 360;

            float oldRot = deformSphere.lastRotationDegree;
            float newRot = oldRot + rotations;

            if( newRot >= 360 )
            {
                oldRot -= 360;
                newRot -= 360;
            }

            deformSphere.lastRotationDegree = fmodf(newRot, 360);

            for(unsigned int i=0; i<deformSphere.matrix_animation.numKeys(); ++i)
            {
                srtToMatrix( buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 0, 0, 1 ), lerp(oldRot, newRot, i / (deformSphere.matrix_animation.numKeys()-1.f)) ),
                                       rotationPivot, translation ),
                             deformSphere.matrix_animation.motionData( 0, i ) );
            }

        }
        {
            float3 scale           = make_float3( 0.02f );
            float3 shear           = make_float3( 0 );
            float3 scaleShearPivot = make_float3( 0 );
            float3 rotationPivot   = make_float3( 0, -1.2f, 0.0f );
            float3 translation     = make_float3( 0, 1.2f, 0.0f );

            float rotationsPerSecond = plane.planeSpeed;
            float timePassed = state.time - state.time_last_frame;
            float rotations = rotationsPerSecond * timePassed * 360;

            plane.srt_animation.motionData( 0, 0 ) = plane.srt_animation.motionData( 0, 1 );
            float oldRot = plane.lastRotationDegree;
            float newRot = oldRot + rotations;

            if( newRot >= 360 )
            {
                oldRot -= 360;
                newRot -= 360;
            }

            plane.lastRotationDegree = fmodf( newRot, 360 );

            plane.srt_animation.motionData( 0, 0 ) =
                buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 1, 0, 0 ), oldRot ), rotationPivot, translation );
            plane.srt_animation.motionData( 0, 1 ) =
                buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 1, 0, 0 ), newRot ), rotationPivot, translation );

            plane.srt_animation.motionData( 1, 0 ) = plane.srt_animation.motionData( 0, 0 );
            plane.srt_animation.motionData( 1, 1 ) = plane.srt_animation.motionData( 0, 1 );
        }

        {
            float3 scale           = make_float3( 1 );
            float3 shear           = make_float3( 0 );
            float3 scaleShearPivot = make_float3( 0 );
            float3 rotationPivot   = make_float3( 0.0003f, -0.4179f, 0.0f );
            float3 translation     = make_float3( 0 );

            const float rotationsPerSecond = 5;

            float lastLocalt = fmodf( state.time_last_frame, 1.f / rotationsPerSecond );
            float nowlocalt  = fmodf( state.time, 1.f / rotationsPerSecond );

            if( lastLocalt > nowlocalt )
                lastLocalt -= 1.f / rotationsPerSecond;

            for( unsigned int i = 0; i < plane.srt_animationPropeller.numKeys(); ++i )
            {
                plane.srt_animationPropeller.motionData( 0, i ) =
                    buildSRT( scale, shear, scaleShearPivot,
                              Quaternion( make_float3( 0, 0, 1 ), lerp(lastLocalt, nowlocalt, i / (plane.srt_animationPropeller.numKeys() - 1.f)) * 360.f * rotationsPerSecond ),
                              rotationPivot, translation);
            }
        }

        CUDA_CHECK( cudaMemcpy( (char*)deformSphere.d_matrices, deformSphere.matrix_animation.data(), deformSphere.matrix_animation.byteSize(), cudaMemcpyHostToDevice ) );
        CUDA_CHECK( cudaMemcpy( (char*)plane.d_srts, plane.srt_animation.data(), plane.srt_animation.byteSize(), cudaMemcpyHostToDevice ) );
        CUDA_CHECK( cudaMemcpy( (char*)plane.d_srtsPropeller, plane.srt_animationPropeller.data(), plane.srt_animationPropeller.byteSize(), cudaMemcpyHostToDevice ) );
    }
#endif

#if 1
    {
        std::vector<OptixInstance> instances = state.instances;
        for( size_t i = 0; i < state.fume.size(); ++i )
        {
            ExhaustFume& fume = state.fume[i];
            OptixInstance  oi   = {};
            memcpy( oi.transform, &Matrix3x4::Identity(), sizeof( float ) * 12 );
            oi.visibilityMask = 255;

            OptixSRTMotionTransform& srt = fume.srt_animation.transform( 0 );

            OptixTraversableHandle handle = fume.exploding_gas_handle;
            srt.child = handle;
            optixConvertPointerToTraversableHandle( state.context, fume.d_srt,
                                                    OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, &handle );
            oi.traversableHandle = handle;

            CUDA_CHECK( cudaMemcpy( (char*)fume.d_srt, fume.srt_animation.data(), fume.srt_animation.byteSize(), cudaMemcpyHostToDevice ) );
            instances.emplace_back( oi );
        }

        size_t instances_size_in_bytes = sizeof( OptixInstance ) * instances.size();
        if( state.d_instances_size < instances_size_in_bytes )
        {
            if( state.d_instances )
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_instances ) ) );

            CUDA_CHECK( cudaMalloc( (void**)&state.d_instances, instances_size_in_bytes ) );
            state.d_instances_size = instances_size_in_bytes;
        }
        CUDA_CHECK( cudaMemcpy( (void*)state.d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice ) );

        state.ias_instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        state.ias_instance_input.instanceArray.instances    = state.d_instances;
        state.ias_instance_input.instanceArray.numInstances = static_cast<int>( instances.size() );

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &state.ias_accel_options, &state.ias_instance_input, 1, &ias_buffer_sizes ) );
        // grow in size if required
        if( state.ias_output_buffer_size < ias_buffer_sizes.outputSizeInBytes )
        {
            CUDA_CHECK( cudaFree( (void*)state.d_ias_output_buffer ) );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), ias_buffer_sizes.outputSizeInBytes ) );
            state.ias_output_buffer_size = ias_buffer_sizes.outputSizeInBytes;
        }
    }
#endif

    OPTIX_CHECK( optixAccelBuild( state.context, state.stream,
                                  &state.ias_accel_options, &state.ias_instance_input, 1,
                                  state.d_temp_buffer, state.temp_buffer_size,
                                  state.d_ias_output_buffer, state.ias_output_buffer_size, &state.ias_handle, nullptr, 0 ) );
    state.params.handle = state.ias_handle;
}

void buildMeshAccel( MotionGeometryState& state )
{
    // Allocate temporary space for vertex generation.
    // The same memory space is reused for generating the deformed and exploding vertices before updates.
    uint32_t numVertices = g_tessellation_resolution * g_tessellation_resolution * 6;
    const size_t vertices_size_in_bytes = numVertices * sizeof( float3 );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_temp_vertices[0] ), vertices_size_in_bytes ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_temp_vertices[1] ), vertices_size_in_bytes ) );

    // Build static triangulated sphere.
    launchGenerateAnimatedVertices( state, AnimationMode_None, 0, 0, g_tessellation_resolution );

    // Build an AS over the triangles.
    // We use un-indexed triangles so we can explode the sphere per triangle.
    state.triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    state.triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    state.triangle_input.triangleArray.vertexStrideInBytes = sizeof( float3 );
    state.triangle_input.triangleArray.numVertices = numVertices;
    state.triangle_input.triangleArray.vertexBuffers = state.d_temp_vertices;
    state.triangle_input.triangleArray.flags = &state.triangle_flags;
    state.triangle_input.triangleArray.numSbtRecords = 1;
    state.triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
    state.triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    state.triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    state.triangle_input_fume = state.triangle_input;
    state.triangle_input_fume.triangleArray.numVertices = g_tessellation_resolution_fume * g_tessellation_resolution_fume * 6;

    OptixAccelBuildOptions gas_accel_options = {};
    gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    gas_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    gas_accel_options.motionOptions.numKeys = 2;
    gas_accel_options.motionOptions.timeBegin = 0;
    gas_accel_options.motionOptions.timeEnd   = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &gas_accel_options,
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
        state.stream,                                  // CUDA stream
        &gas_accel_options,
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

    // The memory requirements for the uncompressed exploding GAS (fume) won't change so we can rebuild in-place.
    state.exploding_gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    OptixAccelRelocationInfo relocationInfo;
    OPTIX_CHECK( optixAccelGetRelocationInfo( state.context, state.static_gas_handle, &relocationInfo ) );

    // Compress sphere GAS

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, ( void* )emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_static_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, state.stream, state.static_gas_handle, state.d_static_gas_output_buffer, compacted_gas_size, &state.static_gas_handle ) );

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
    OPTIX_CHECK( optixAccelRelocate( state.context, state.stream, &relocationInfo, 0, 0, state.d_deforming_gas_output_buffer, state.deforming_gas_output_buffer_size, &state.deforming_gas_handle ) );

    {
        deformSphere.matrix_animation = MatrixMotionTransformArray( 1, 10 );
        for(unsigned int i=0; i<deformSphere.matrix_animation.numKeys(); ++i)
        {
            *(Matrix3x4*)deformSphere.matrix_animation.motionData( 0, i ) = Matrix3x4::Identity();
        }

    }

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    {
        std::string fileName = sutil::sampleFilePath( "data/Plane", "biplane_no_propeller.obj" );
        state.plane.load( state.context, fileName, {}, {} );
    }
    {
        std::string fileName = sutil::sampleFilePath( "data/Plane", "biplane_propeller.obj" );
        state.planePropeller.load( state.context, fileName, {}, {} );
    }
    {
        plane.srt_animation = SRTMotionTransformArray( 2, 2 );

        float3 scale           = make_float3( 1 );
        float3 shear           = make_float3( 0 );
        float3 scaleShearPivot = make_float3( 0 );
        float3 rotationPivot   = make_float3( 0 );
        float3 translation     = make_float3( 0 );

        plane.srt_animation.motionData( 0, 0 ) = buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 0, 0, 1 ), 0 ), rotationPivot, translation );
        plane.srt_animation.motionData( 0, 1 ) = buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 0, 0, 1 ), 0 ), rotationPivot, translation );

        plane.srt_animation.motionData( 1, 0 ) = buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 0, 0, 1 ), 0 ), rotationPivot, translation );
        plane.srt_animation.motionData( 1, 1 ) = buildSRT( scale, shear, scaleShearPivot, Quaternion( make_float3( 0, 0, 1 ), 0 ), rotationPivot, translation );
    }
    {
        plane.srt_animationPropeller = SRTMotionTransformArray( 1, 100 );
        plane.srt_animationPropeller.motionData( 0, 0 ) = buildSRT( make_float3( 1 ), Quaternion( make_float3( 0, 0, 1 ), 0 ), make_float3( 0 ) );
        for( unsigned int i = 1; i < plane.srt_animationPropeller.numKeys(); ++i )
        {
            plane.srt_animationPropeller.motionData( 0, i ) = plane.srt_animationPropeller.motionData( 0, 0 );
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    // Build the IAS

    // alloc memory to be able to generate handles
    CUDA_CHECK( cudaMalloc( (void**)&deformSphere.d_matrices, deformSphere.matrix_animation.byteSize() ) );
    CUDA_CHECK( cudaMalloc( (void**)&plane.d_srts, plane.srt_animation.byteSize() ) );
    CUDA_CHECK( cudaMalloc( (void**)&plane.d_srtsPropeller, plane.srt_animationPropeller.byteSize() ) );

    std::vector<OptixInstance>& instances = state.instances;
    instances.resize( INST_COUNT );

    for( size_t i = 0; i < instances.size(); ++i )
    {
        memcpy( instances[i].transform, &Matrix3x4::Identity(), sizeof( float ) * 12 );
        instances[i].sbtOffset = 0;
        instances[i].visibilityMask = 255;
    }

    unsigned int iIdx = 0;
    instances[iIdx++].traversableHandle = state.static_gas_handle;

    {
        OptixTraversableHandle handle = state.deforming_gas_handle;

        OptixMatrixMotionTransform& t = deformSphere.matrix_animation.transform( 0 );
        t.child                       = handle;
        t.motionOptions.flags         = 0;
        t.motionOptions.numKeys       = deformSphere.matrix_animation.numKeys();
        t.motionOptions.timeBegin     = 0;
        t.motionOptions.timeEnd       = 1;

        optixConvertPointerToTraversableHandle( state.context, ( CUdeviceptr )( (char*)deformSphere.d_matrices ),
                                                OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM, &handle );

        instances[iIdx++].traversableHandle = handle;
    }
    {
        OptixTraversableHandle handle = state.plane.getHandle();

        unsigned int             tIdx = 0;
        OptixSRTMotionTransform& t    = plane.srt_animation.transform( tIdx );
        t.child                       = handle;
        t.motionOptions.flags         = 0;
        t.motionOptions.numKeys       = plane.srt_animation.numKeys();
        t.motionOptions.timeBegin     = 0;
        t.motionOptions.timeEnd       = 1;

        optixConvertPointerToTraversableHandle( state.context,
                                                (CUdeviceptr)((char*)plane.d_srts + plane.srt_animation.byteSizePerTransform() * tIdx),
                                                OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, &handle );

        instances[iIdx++].traversableHandle = handle;
    }
    {
        OptixTraversableHandle handle = state.planePropeller.getHandle();

        {
            unsigned int             tIdx = 0;
            OptixSRTMotionTransform& t    = plane.srt_animationPropeller.transform( tIdx );
            t.child                       = handle;
            t.motionOptions.flags         = 0;
            t.motionOptions.numKeys       = plane.srt_animationPropeller.numKeys();
            t.motionOptions.timeBegin     = 0;
            t.motionOptions.timeEnd       = 1;

            optixConvertPointerToTraversableHandle( state.context,
                                                    (CUdeviceptr)((char*)plane.d_srtsPropeller + plane.srt_animationPropeller.byteSizePerTransform() * tIdx),
                                                    OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, &handle );
        }

        {
            unsigned int             tIdx = 1;
            OptixSRTMotionTransform& t    = plane.srt_animation.transform( tIdx );
            t.child                       = handle;
            t.motionOptions.flags         = 0;
            t.motionOptions.numKeys       = plane.srt_animation.numKeys();
            t.motionOptions.timeBegin     = 0;
            t.motionOptions.timeEnd       = 1;

            optixConvertPointerToTraversableHandle( state.context,
                                                    (CUdeviceptr)((char*)plane.d_srts + plane.srt_animation.byteSizePerTransform() * tIdx),
                                                    OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, &handle );
        }

        instances[iIdx++].traversableHandle = handle;
    }
    CUDA_CHECK( cudaMemcpy( (char*)deformSphere.d_matrices, deformSphere.matrix_animation.data(), deformSphere.matrix_animation.byteSize(), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( (char*)plane.d_srts, plane.srt_animation.data(), plane.srt_animation.byteSize(), cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( (char*)plane.d_srtsPropeller, plane.srt_animationPropeller.data(), plane.srt_animationPropeller.byteSize(), cudaMemcpyHostToDevice ) );

    size_t      instances_size_in_bytes = sizeof( OptixInstance ) * instances.size();
    CUDA_CHECK( cudaMalloc( ( void** )&state.d_instances, instances_size_in_bytes ) );
    state.d_instances_size = instances_size_in_bytes;
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice ) );

    state.ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    state.ias_instance_input.instanceArray.instances = state.d_instances;
    state.ias_instance_input.instanceArray.numInstances = static_cast<int>( instances.size() );

    // we choose FAST_BUILD here as we need to rebuild every frame, no update, compaction needed
    state.ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    // In this interactive sample, build times can govern render times.
    // Hence, we build a static IAS with faster build times despite slower traversal times.
#if 1
    state.ias_accel_options.motionOptions.numKeys = 1;
#else
    state.ias_accel_options.motionOptions.numKeys = 2;
    state.ias_accel_options.motionOptions.timeBegin = 0;
    state.ias_accel_options.motionOptions.timeEnd = 1;
#endif
    state.ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &state.ias_accel_options, &state.ias_instance_input, 1, &ias_buffer_sizes ) );

    // non-compacted output
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), ias_buffer_sizes.outputSizeInBytes ) );

    size_t maxUpdateTempSize = std::max( ias_buffer_sizes.tempSizeInBytes, gas_buffer_sizes.tempUpdateSizeInBytes );
    if( maxUpdateTempSize > state.temp_buffer_size )
    {
        CUDA_CHECK( cudaFree( (void*)state.d_temp_buffer ) );
        state.temp_buffer_size = maxUpdateTempSize;
        CUDA_CHECK( cudaMalloc( (void**)&state.d_temp_buffer, state.temp_buffer_size ) );
    }

    OPTIX_CHECK( optixAccelBuild( state.context, state.stream, &state.ias_accel_options, &state.ias_instance_input, 1, state.d_temp_buffer,
                                  ias_buffer_sizes.tempSizeInBytes, state.d_ias_output_buffer,
                                  ias_buffer_sizes.outputSizeInBytes, &state.ias_handle, nullptr, 0 ) );

    state.params.handle = state.ias_handle;
}


void createModule( MotionGeometryState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = true;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipeline_compile_options.numPayloadValues = 5;
    state.pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    OptixModuleCompileBoundValueEntry boundValue = {};
    {
        boundValue.pipelineParamOffsetInBytes = offsetof( Params, ao );
        boundValue.sizeInBytes                = sizeof( Params::ao );
        boundValue.boundValuePtr              = &state.renderAO;
        boundValue.annotation                 = "ao";
        module_compile_options.numBoundValues = 1;
        module_compile_options.boundValues    = &boundValue;
    }

    size_t      inputSize = 0;
    const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixMotionGeometry.cu", inputSize );

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


void createProgramGroups( MotionGeometryState& state )
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

    {
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";
        sizeof_log                                  = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate( state.context, &miss_prog_group_desc,
                                                  1,  // num program groups
                                                  &program_group_options, log, &sizeof_log, &state.miss_group_occlusion ) );
    }
}


void createPipeline( MotionGeometryState& state )
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.miss_group,
        state.miss_group_occlusion,
        state.hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 20;
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
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.miss_group_occlusion, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.hit_group, &stack_sizes ) );

    uint32_t max_trace_depth = pipeline_link_options.maxTraceDepth;
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

    // This is 4 since the largest depth is IAS->MT->MT->GAS
    const uint32_t max_traversable_graph_depth = 4;

    OPTIX_CHECK( optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversable_graph_depth
    ) );
}


void createSBT( MotionGeometryState& state )
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
    const unsigned int numMissPrograms = 2;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &d_miss_records ), miss_record_size * numMissPrograms ) );

    MissRecord ms_sbt[numMissPrograms];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_group, &ms_sbt[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_group_occlusion, &ms_sbt[1] ) );
    ms_sbt[0].data.bg_color = make_float4( 0.0f );
    ms_sbt[1].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_miss_records ),
        ms_sbt,
        miss_record_size * numMissPrograms,
        cudaMemcpyHostToDevice
    ) );

    std::vector<HitGroupRecord> hitgroup_records( 1 );

    OPTIX_CHECK( optixSbtRecordPackHeader( state.hit_group, &hitgroup_records[0] ) );
    hitgroup_records[0].data.color = make_float3( 1 );

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >( &d_hitgroup_records ),
        hitgroup_record_size * hitgroup_records.size()
    ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_hitgroup_records ),
        hitgroup_records.data(),
        hitgroup_record_size*hitgroup_records.size(),
        cudaMemcpyHostToDevice
    ) );

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast< uint32_t >( miss_record_size );
    state.sbt.missRecordCount = numMissPrograms;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast< uint32_t >( hitgroup_record_size );
    state.sbt.hitgroupRecordCount = static_cast< uint32_t >( hitgroup_records.size() );
}


void cleanupState( MotionGeometryState& state )
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
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_temp_vertices[0] ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_temp_vertices[1] ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_static_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_deforming_gas_output_buffer ) ) );
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
    MotionGeometryState state;
    state.params.width  = 1024;
    state.params.height = 768;
    state.time = 0.f;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    int num_frames = 18;
    float animation_time = 10.0f;

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
        buildMeshAccel( state );
        createSBT( state );
        initLaunchParams( state );

        if( outfile.empty() )
        {
            std::cout << "Keys:    Up/Down                     Double/half target frame rate\n";
            std::cout << "         M/N                         Increase/reduce plane speed\n";
            std::cout << "         J/H                         Increase/reduce deform sphere orbit speed\n";
            std::cout << "         V                           Toggle: camera follow plane\n";
            std::cout << "         B                           Add exhaust fume\n";
            std::cout << "         A                           Toggle: AO rendering\n";
            GLFWwindow* window = sutil::initUI( "optixMotionGeometry", state.params.width, state.params.height );
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
                std::chrono::duration<double> full_frame_time( 1/60.0 ); // init with 60.0 fps

                auto tstart = std::chrono::system_clock::now();

                state.targetFrameTime                    = 1 / 30.0f;

                do
                {
                    cudaDeviceSynchronize();
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    //////////////////////////////////////////////////////////////////////////
                    auto                          tnow = std::chrono::system_clock::now();
                    std::chrono::duration<double> time = tnow - tstart;
                    state.time_last_frame              = state.time;
                    state.time                         = (float)time.count();
                    float timePassed                   = state.time - state.time_last_frame;

                    unsigned int targetSpp = max( 1u, (unsigned int)(state.targetFrameTime / timePassed * state.params.spp) );
                    if( abs( (float)targetSpp / state.params.spp - 1u ) > 0.2 )
                    {
                        state.params.spp = ( state.params.spp + targetSpp ) / 2;
                    }
                    else
                    {
                        if( state.time - state.time_last_frame < state.targetFrameTime )
                            state.params.spp++;
                        else
                            state.params.spp = max( 1u, state.params.spp - 1 );
                    }
                    //////////////////////////////////////////////////////////////////////////

                    if( state.time - state.time_last_fume > 0.4f )
                    {
                        addFume( state );
                        state.time_last_fume = state.time;
                    }

                    //////////////////////////////////////////////////////////////////////////

                    updateMeshAccel( state );

                    handleCameraUpdate( state );
                    handleResize( output_buffer, state.params );

                    // sync to correctly attribute where time is spent
                    cudaDeviceSynchronize();

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

                    full_frame_time = state_update_time + render_time + display_time;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    sutil::beginFrameImGui();
                    static char display_text[256];
                    sprintf( display_text,
                             "ambient occlusion: %s\n"
                             "samples per pixel: %d\n",
                             ( state.renderAO ? "on" : "off" ), state.params.spp );
                    sutil::displayText( display_text, 10.0f, 100.0f );
                    sutil::endFrameImGui();


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

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                state.params.width,
                state.params.height
            );

            handleCameraUpdate( state );
            handleResize( output_buffer, state.params );

            // run animation frames
            for( unsigned int i = 0; i < static_cast<unsigned int>( num_frames ); ++i )
            {
                state.time_last_frame = state.time;
                state.time = i * ( animation_time / ( num_frames - 1 ) );

                if( state.time - state.time_last_fume > 0.4f )
                {
                    addFume( state );
                    state.time_last_fume = state.time;
                }

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
