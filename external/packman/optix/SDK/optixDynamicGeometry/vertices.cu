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

#include "vertices.h"

__forceinline__ __device__ float triangle_wave( float x, float shift = 0.f, float period = 2.f * M_PI, float amplitude = 1.f )
{
    return fabsf( fmodf( ( 4.f / period ) * ( x - shift ), 4.f * amplitude ) - 2.f * amplitude ) - amplitude;
}

__forceinline__ __device__ void write_animated_triangle( float3* out_vertices, int tidx, float3 v0, float3 v1, float3 v2, AnimationMode mode, float time )
{
    float3 v = make_float3( 0 );

    if( mode == AnimationMode_Explode )
    {
        // Generate displacement vector from triangle index
        const float theta = ( ( float )M_PI * ( tidx * ( 13 / M_PI ) ) );
        const float phi = ( ( float )( 2.0 * M_PI ) * ( tidx * ( 97 / M_PI ) ) );

        // Apply displacement to the sphere triangles
        v = make_float3( triangle_wave( phi ) * triangle_wave( theta, M_PI / 2.f ),
                         triangle_wave( phi, M_PI / 2.f ) * triangle_wave( theta, M_PI / 2.f ), triangle_wave( theta ) )
            * triangle_wave( time, M_PI / 2.f ) * 2.f;
    }

    out_vertices[tidx * 3 + 0] = v0 + v;
    out_vertices[tidx * 3 + 1] = v1 + v;
    out_vertices[tidx * 3 + 2] = v2 + v;
}

__forceinline__ __device__ float3 deform_vertex( const float3& c, AnimationMode mode, float time )
{
    // Apply sine wave to the y coordinate of the sphere vertices
    if( mode == AnimationMode_Deform )
        return make_float3( c.x, c.y * ( 0.5f + 0.4f * cosf( 4 * ( c.x + time ) ) ), c.z );
    return c;
}

extern "C" __global__ void generate_vertices(float3* out_vertices, AnimationMode mode, float time, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < width * height )
    {
        // generate a single patch (two unindexed triangles) of a tessellated sphere

        int x = idx % width;
        int y = idx / width;

        const float theta0 = ( ( float )M_PI * ( y + 0 ) ) / height;
        const float theta1 = ( ( float )M_PI * ( y + 1 ) ) / height;
        const float phi0 = ( ( float )( 2.0 * M_PI ) * ( x + 0 ) ) / width;
        const float phi1 = ( ( float )( 2.0 * M_PI ) * ( x + 1 ) ) / width;

        const float ct0 = cosf( theta0 );
        const float st0 = sinf( theta0 );
        const float ct1 = cosf( theta1 );
        const float st1 = sinf( theta1 );

        const float cp0 = cosf( phi0 );
        const float sp0 = sinf( phi0 );
        const float cp1 = cosf( phi1 );
        const float sp1 = sinf( phi1 );

        const float3 v00 = deform_vertex( make_float3( cp0 * st0, sp0 * st0, ct0 ), mode, time );
        const float3 v10 = deform_vertex( make_float3( cp0 * st1, sp0 * st1, ct1 ), mode, time );
        const float3 v01 = deform_vertex( make_float3( cp1 * st0, sp1 * st0, ct0 ), mode, time );
        const float3 v11 = deform_vertex( make_float3( cp1 * st1, sp1 * st1, ct1 ), mode, time );

        write_animated_triangle( out_vertices, idx * 2 + 0, v00, v10, v11, mode, time );
        write_animated_triangle( out_vertices, idx * 2 + 1, v00, v11, v01, mode, time );
    }
}

extern "C" __host__ void
generateAnimatedVetrices( float3* out_vertices, AnimationMode animation_mode, float time, int width, int height )
{
    dim3 threadsPerBlock( 128, 1 );
    int numBlocks = ( width * height ) / threadsPerBlock.x;
    generate_vertices <<< numBlocks, threadsPerBlock >>> ( out_vertices, animation_mode, time, width, height );
}
