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


#include "OptiXDenoiser.h"

#include <sutil/sutil.h>
#include <sutil/Exception.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>



//------------------------------------------------------------------------------
//
//  optixDenoiser -- Demonstration of the OptiX denoising API.  
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cout
        << "Usage  : " << argv0 << " [options] {-A | --AOV aov.exr} color.exr\n"
        << "Options: -n | --normal <normal.exr>\n"
        << "         -a | --albedo <albedo.exr>\n"
        << "         -f | --flow   <flow.exr>\n"
        << "         -o | --out    <out.exr> Defaults to 'denoised.exr'\n"
        << "         -F | --Frames <int-int> first-last frame number in sequence\n"
        << "         -e | --exposure <float> apply exposure on output images\n"
        << "         -t | --tilesize <int> <int> use tiling to save GPU memory\n"
        << "         -z apply flow to input images (no denoising), write output\n"
        << "         -k use kernel prediction model even if there are no AOVs\n"
        << "in sequences, first occurrence of '+' characters substring in filenames is replaced by framenumber\n"
        << std::endl;
    exit(0);
}

// filename is copied to result and the first sequence of "+" characters is
// replaced (using leading zeros) with framename.
// true is returned if the framenumber is -1 or if the function was successful.

static bool getFrameFilename( std::string& result, const std::string& filename, int frame )
{
    result = filename;
    if( frame == -1 )
        return true;
    size_t nplus = 0;
    size_t ppos = result.find( '+' );
    if( ppos == std::string::npos )
        return true;                    // static filename without "+" characters
    size_t cpos = ppos;
    while( result[cpos] != 0 && result[cpos] == '+' )
    {
        nplus++;
        cpos++;
    }
    std::string fn = std::to_string( frame );
    if( fn.length() > nplus )
    {
        std::cout << "illegal temporal filename, framenumber requires " << fn.length() << " digits, \"+\" placeholder length: " << nplus << "too small" << std::endl;
        return false;
    }
    for( size_t i = 0; i < nplus; i++ )
        result[ppos + i] = '0';
    for( size_t i = 0; i < fn.length(); i++ )
        result[ppos + nplus - 1 - i] = fn[fn.length() - 1 - i];
    return true;
}

int32_t main( int32_t argc, char** argv )
{
    if( argc < 2 )
        printUsageAndExit( argv[0] );
    std::string color_filename = argv[argc-1];

    std::string normal_filename;
    std::string albedo_filename;
    std::string flow_filename;
    std::string output_filename = "denoised.exr";
    std::vector<std::string> aov_filenames;
    bool kpMode = false;
    bool applyFlow = false;
    float exposure = 0.f;
    int firstFrame = -1, lastFrame = -1;
    unsigned int tileWidth = 0, tileHeight = 0;

    for( int32_t i = 1; i < argc-1; ++i )
    {
        std::string arg( argv[i] );

        if( arg == "-n" || arg == "--normal" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            normal_filename = argv[++i];
        }
        else if( arg == "-a" || arg == "--albedo" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            albedo_filename = argv[++i];
        }
        else if( arg == "-e" || arg == "--exposure" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            exposure = std::stof( argv[++i] );
        }
        else if( arg == "-f" || arg == "--flow" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            flow_filename = argv[++i];
        }
        else if( arg == "-o" || arg == "--out" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            output_filename = argv[++i];
        }
        else if( arg == "-t" || arg == "--tilesize" )
        {
            if( i == argc-3 )
                printUsageAndExit( argv[0] );
            tileWidth  = atoi( argv[++i] );
            tileHeight = atoi( argv[++i] );
        }
        else if( arg == "-A" || arg == "--AOV" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            aov_filenames.push_back( std::string( argv[++i] ) );
        }
        else if( arg == "-k" )
        {
            kpMode = true;
        }
        else if( arg == "-z" )
        {
            applyFlow = true;
        }
        else if( arg == "-F" || arg == "--Frames" )
        {
            if( i == argc-2 )
                printUsageAndExit( argv[0] );
            std::string s( argv[++i] );
            size_t cpos = s.find( '-' );
            if( cpos == 0 || cpos == s.length()-1 || cpos == std::string::npos )
                printUsageAndExit( argv[0] );
            firstFrame = atoi( s.substr( 0, cpos ).c_str() );
            lastFrame = atoi( s.substr( cpos + 1 ).c_str() );
            if( firstFrame < 0 || lastFrame < 0 || firstFrame > lastFrame )
            {
                std::cout << "illegal frame range, first frame must be <= last frame and >= 0" << std::endl;
                exit( 0 );
            }
        }
        else
        {
            printUsageAndExit( argv[0] );
        }
    }

    bool temporalMode = bool( firstFrame != -1 );

    sutil::ImageBuffer color     = {};
    sutil::ImageBuffer normal    = {};
    sutil::ImageBuffer albedo    = {};
    sutil::ImageBuffer flow      = {};
    std::vector<sutil::ImageBuffer> aovs;

    try
    {
        OptiXDenoiser denoiser;
        for( int frame=firstFrame; frame <= lastFrame; frame++ )
        {
            const double t0 = sutil::currentTime();
            std::cout << "Loading inputs ";
            if( frame != -1 )
                std::cout << "for frame " << frame;
            std::cout << " ..." << std::endl;

            std::string frame_filename;
            if( !getFrameFilename( frame_filename, color_filename, frame ) )
            {
                std::cout << "cannot open color file" << std::endl;
                exit( 0 );
            }
            color = sutil::loadImage( frame_filename.c_str() );
            std::cout << "\tLoaded color image " << frame_filename << " (" << color.width << "x" << color.height << ")" << std::endl;

            if( !normal_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, normal_filename, frame ) )
                {
                    std::cout << "cannot open normal file" << std::endl;
                    exit( 0 );
                }
                // allocate four channels. only two/three channels used depending on model.
                normal = sutil::loadImage( frame_filename.c_str() );
                std::cout << "\tLoaded normal image " << frame_filename << std::endl;
            }

            if( !albedo_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, albedo_filename, frame ) )
                {
                    std::cout << "cannot open albedo file" << std::endl;
                    exit( 0 );
                }
                // allocate four channels. only three channels used.
                albedo = sutil::loadImage( frame_filename.c_str() );
                std::cout << "\tLoaded albedo image " << frame_filename << std::endl;
            }

            if( frame > firstFrame && !flow_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, flow_filename, frame ) )
                {
                    std::cout << "cannot open flow file" << std::endl;
                    exit( 0 );
                }
                // allocate four channels. only two channels used.
                // sutil::loadImage handles only 3 and 4 channels.
                flow = sutil::loadImage( frame_filename.c_str() );
                std::cout << "\tLoaded flow image " << frame_filename << std::endl;
            }

            for( size_t i=0; i < aov_filenames.size(); i++ )
            {
                if( !getFrameFilename( frame_filename, aov_filenames[i], frame ) )
                {
                    std::cout << "cannot open aov file" << std::endl;
                    exit( 0 );
                }
                aovs.push_back( sutil::loadImage( frame_filename.c_str() ) );
                std::cout << "\tLoaded aov image " << frame_filename << std::endl;
            }

            const double t1 = sutil::currentTime();
            std::cout << "\tLoad inputs from disk     :"
                      << std::fixed << std::setw( 8 ) << std::setprecision(2)
                      << ( t1-t0 )*1000.0 << " ms" << std::endl;

            SUTIL_ASSERT( color.pixel_format  == sutil::FLOAT4 );
            SUTIL_ASSERT( !albedo.data || albedo.pixel_format == sutil::FLOAT4 );
            SUTIL_ASSERT( !normal.data || normal.pixel_format == sutil::FLOAT4 );
            SUTIL_ASSERT( !flow.data || flow.pixel_format == sutil::FLOAT4 );
            for( size_t i=0; i < aov_filenames.size(); i++ )
                SUTIL_ASSERT( aovs[i].pixel_format == sutil::FLOAT4 );

            OptiXDenoiser::Data data;
            data.width  = color.width;
            data.height = color.height;
            data.color  = reinterpret_cast<float*>(  color.data );
            data.albedo = reinterpret_cast<float*>( albedo.data );
            data.normal = reinterpret_cast<float*>( normal.data );
            data.flow   = reinterpret_cast<float*>( flow.data );

            // set AOVs
            for( size_t i=0; i < aovs.size(); i++ )
                data.aovs.push_back( reinterpret_cast<float*>( aovs[i].data ) );

            // allocate outputs
            for( size_t i=0; i < 1 + aovs.size(); i++ )
                data.outputs.push_back( new float[ color.width*color.height*4  ] );

            std::cout << "Denoising ..." << std::endl;

            if( frame == firstFrame )
            {
                const double t0 = sutil::currentTime();
                denoiser.init( data, tileWidth, tileHeight, kpMode, temporalMode );
                const double t1 = sutil::currentTime();
                std::cout << "\tAPI Initialization        :"
                          << std::fixed << std::setw( 8 ) << std::setprecision(2)
                          << ( t1-t0 )*1000.0 << " ms" << std::endl;
            }
            else
            {
                denoiser.update( data );
            }

            {
                const double t0 = sutil::currentTime();
                denoiser.exec();
                const double t1 = sutil::currentTime();
                std::cout << "\tDenoise frame             :"
                          << std::fixed << std::setw( 8 ) << std::setprecision(2)
                          << ( t1-t0 )*1000.0 << " ms" << std::endl;
            }

            {
                const double t0 = sutil::currentTime();
                if( applyFlow )
                    denoiser.getFlowResults();
                else
                    denoiser.getResults();
                const double t1 = sutil::currentTime();
                std::cout << "\tCleanup state/copy to host:"
                          << std::fixed << std::setw( 8 ) << std::setprecision(2)
                          << ( t1-t0 )*1000.0 << " ms" << std::endl;
            }

            {
                const double t0 = sutil::currentTime();

                for( size_t i=0; i < 1 + aovs.size(); i++ )
                {
                    sutil::ImageBuffer output_image;
                    output_image.width        = color.width;
                    output_image.height       = color.height;
                    output_image.data         = data.outputs[i];
                    output_image.pixel_format = sutil::FLOAT4;

                    frame_filename = output_filename;
                    getFrameFilename( frame_filename, output_filename, frame );
                    if( i > 0 )
                    {
                        std::string basename = aov_filenames[i-1].substr( aov_filenames[i-1].find_last_of( "/\\" ) + 1 );
                        std::string::size_type const p( basename.find_last_of('.') );
                        std::string b = basename.substr( 0, p );
                        frame_filename.insert( frame_filename.rfind('.'), "_" + b + "_denoised" );
                    }
                    if( exposure != 0.f )
                    {
                        for( unsigned int p=0; p < color.width * color.height; p++ )
                        {
                            float * f = &((float*)output_image.data)[p * 4 + 0];
                            f[0] *= std::pow( 2.f, exposure );
                            f[1] *= std::pow( 2.f, exposure );
                            f[2] *= std::pow( 2.f, exposure );
                        }
                    }
                    std::cout << "Saving results to '" << frame_filename << "'..." << std::endl;
                    sutil::saveImage( frame_filename.c_str(), output_image, false );
                }

                const double t1 = sutil::currentTime();
                std::cout << "\tSave output to disk       :"
                          << std::fixed << std::setw( 8 ) << std::setprecision(2)
                          << ( t1-t0 )*1000.0 << " ms" << std::endl;
            }

            delete [] reinterpret_cast<float*>( color.data );
            delete [] reinterpret_cast<float*>( albedo.data );
            delete [] reinterpret_cast<float*>( normal.data );
            delete [] reinterpret_cast<float*>( flow.data );
            for( size_t i=0; i < 1 + aovs.size(); i++ )
                delete [] ( data.outputs[i] );
        }

        denoiser.finish();
    }
    catch( std::exception& e )
    {
        std::cerr << "ERROR: exception caught '" << e.what() << "'" << std::endl;
    }
}
