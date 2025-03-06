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

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

int main( int argc, char* argv[] )
{
    // Parse command line options
    int device = 0;
    std::string filename;
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};
    std::vector<OptixModuleCompileBoundValueEntry> bound_values;
    std::vector<std::vector<char>> bound_values_data;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];

        if( i + 1 >= argc )
        {
            std::cerr << "Incorrect number of arguments to option '" << arg << "'\n";
            return 1;
        }

        // Context options
        if( arg == "--file" )
        {
            filename = argv[++i];
        }
        else if( arg == "--device" )
        {
            device = atoi( argv[++i] );
        }

        // Module compile options
        else if( arg == "--maxRegisterCount" )
        {
            module_compile_options.maxRegisterCount = atoi( argv[++i] );
        }
        else if( arg == "--optLevel" )
        {
            module_compile_options.optLevel = static_cast<OptixCompileOptimizationLevel>(atoi( argv[++i] ));
        }
        else if( arg == "--debugLevel" )
        {
            module_compile_options.debugLevel = static_cast<OptixCompileDebugLevel>(atoi( argv[++i] ));
        }
        else if( arg == "--boundValue" )
        {
            if( i + 4 >= argc )
            {
                std::cerr << "Incorrect number of arguments to option '--boundValue'\n";
                return 1;
            }

            OptixModuleCompileBoundValueEntry bound_value = {};
            bound_value.pipelineParamOffsetInBytes = atoi( argv[++i] );
            bound_value.sizeInBytes = atoi( argv[++i] );
            bound_value.annotation = argv[++i];

            const std::string byte_data = argv[++i];
            if( byte_data.size() != bound_value.sizeInBytes * 2 )
            {
                std::cerr << "Incorrect size of encoded bound value data (expected " << (bound_value.sizeInBytes * 2) << " characters, but got " << byte_data.size() << ")\n";
                return 1;
            }

            // Allocate space for the value data and decode it from the command-line argument
            std::vector<char> bound_value_data( bound_value.sizeInBytes );
            for( size_t byte = 0; byte < bound_value.sizeInBytes; ++byte )
            {
                const char byte_low = (byte_data[byte * 2] - '0');
                const char byte_high = (byte_data[byte * 2 + 1] - '0') << 4;
                bound_value_data[byte] = byte_low | byte_high;
            }
            bound_value.boundValuePtr = bound_value_data.data();

            bound_values.push_back( bound_value );
            bound_values_data.push_back( std::move(bound_value_data) ); // Move vector here to data pointer stays valid
            continue;
        }

        // Pipeline compile options
        else if( arg == "--usesMotionBlur" )
        {
            pipeline_compile_options.usesMotionBlur = atoi( argv[++i] ) != 0;
        }
        else if( arg == "--traversableGraphFlags" )
        {
            pipeline_compile_options.traversableGraphFlags = atoi( argv[++i] );
        }
        else if( arg == "--numPayloadValues" )
        {
            pipeline_compile_options.numPayloadValues = atoi( argv[++i] );
        }
        else if( arg == "--numAttributeValues" )
        {
            pipeline_compile_options.numAttributeValues = atoi( argv[++i] );
        }
        else if( arg == "--exceptionFlags" )
        {
            pipeline_compile_options.exceptionFlags = atoi( argv[++i] );
        }
        else if( arg == "--pipelineLaunchParamsVariableName" )
        {
            pipeline_compile_options.pipelineLaunchParamsVariableName = argv[++i];
        }
        else if( arg == "--usesPrimitiveTypeFlags" )
        {
            pipeline_compile_options.usesPrimitiveTypeFlags = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            return 1;
        }
    }

    // Now that all bound values where parsed from the command-line, assign them to the compile options
    if( !bound_values.empty() )
    {
        module_compile_options.boundValues = bound_values.data();
        module_compile_options.numBoundValues = static_cast<unsigned int>(bound_values.size());
    }

    try
    {
        // Initialize CUDA
        CUDA_CHECK( cudaSetDevice(device) );
        CUDA_CHECK( cudaFree(0) );

        // Initialize OptiX (using the CUDA context that was just initialized)
        OptixDeviceContext context = nullptr;
        OPTIX_CHECK( optixInit() );
        OptixDeviceContextOptions options = {};
        OPTIX_CHECK( optixDeviceContextCreate( nullptr, &options, &context ) );

        // Ensure disk cache is enabled, since this relies on the compile result being cached
        int cache_enabled = false;
        OPTIX_CHECK( optixDeviceContextGetCacheEnabled( context, &cache_enabled ) );
        SUTIL_ASSERT( cache_enabled );

        // Read the temporary PTX file
        std::ifstream file( filename.c_str() );
        SUTIL_ASSERT( file.is_open() );
        std::stringstream ptx_stream;
        ptx_stream << file.rdbuf();
        const std::string ptx_string = ptx_stream.str();

        // Actually compile the module and store the result in the OptiX disk cache
        char   log[2048];
        size_t sizeof_log = sizeof(log);
        OptixModule module = nullptr;
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx_string.c_str(), ptx_string.size(),
            log, &sizeof_log,
            &module ) );

        // Clean up
        optixModuleDestroy( module );
        optixDeviceContextDestroy( context );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
