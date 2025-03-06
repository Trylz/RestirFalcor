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

#include <optix.h>

#include "optixMotionGeometry.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
    __constant__ Params params;
}

struct Onb
{
    __forceinline__ __device__ Onb( const float3& normal )
    {
        m_normal = normal;

        if( fabs( m_normal.x ) > fabs( m_normal.z ) )
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize( m_binormal );
        m_tangent  = cross( m_binormal, m_normal );
    }

    __forceinline__ __device__ void inverse_transform( float3& p ) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __forceinline__ __device__ void cosine_sample_hemisphere( const float u1, const float u2, float3& p )
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f * M_PIf * u2;
    p.x             = r * cosf( phi );
    p.y             = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x * p.x - p.y * p.y ) );
}


struct Payload {
    union {
        struct {
            unsigned int p0, p1, p2, p3;
        };
        struct {
            float3 c;
            unsigned int seed;
        };
    };
    __forceinline__ __device__ void setAll()
    {
        optixSetPayload_0( p0 );
        optixSetPayload_1( p1 );
        optixSetPayload_2( p2 );
        optixSetPayload_3( p3 );
    }
    __forceinline__ __device__ void getAll()
    {
        p0 = optixGetPayload_0();
        p1 = optixGetPayload_1();
        p2 = optixGetPayload_2();
        p3 = optixGetPayload_3();
    }
    __forceinline__ __device__ void setC()
    {
        optixSetPayload_0( p0 );
        optixSetPayload_1( p1 );
        optixSetPayload_2( p2 );
    }
    __forceinline__ __device__ void getC()
    {
        p0 = optixGetPayload_0();
        p1 = optixGetPayload_1();
        p2 = optixGetPayload_2();
    }
    __forceinline__ __device__ void getSeed()
    {
        p3 = optixGetPayload_3();
    }
};

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    float                  time,
    Payload&               prd
)
{
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        time,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset, first ray type (only one here)
        0,                   // SBT stride, forcing a single HitGroup!
        0,                   // missSBTIndex, matching first ray type
        prd.p0, prd.p1, prd.p2, prd.p3
    );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;

    Payload payload;
    payload.seed = tea<4>( idx.y * dim.x + idx.x, 12346789 + params.subframe_index );

    float3 final_c = make_float3( 0 );
#pragma unroll 1
    for( int x = 1; x <= params.spp; ++x )
    {
        const float2 d = 2.0f * make_float2(
            ( static_cast< float >( idx.x ) + rnd( payload.seed ) ) / static_cast< float >( dim.x ),
            ( static_cast< float >( idx.y ) + rnd( payload.seed ) ) / static_cast< float >( dim.y )
        ) - 1.0f;
        float3 direction = normalize( d.x * U + d.y * V + W );

        float time = rnd( payload.seed );

        payload.c = make_float3( 0.5f, 0.5f, 0.5f );
        trace( params.handle,
            eye,
            direction,
            0.00f,  // tmin
            1e16f,  // tmax
            time,
            payload );
        final_c += payload.c;
    }
    final_c /= params.spp;
    params.frame_buffer[idx.y * params.width + idx.x] = make_color( final_c );
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data = reinterpret_cast< MissData* >( optixGetSbtDataPointer() );
    Payload p;
    p.c = make_float3( rt_data->bg_color.x, rt_data->bg_color.y, rt_data->bg_color.z );
    p.setC();
}



extern "C" __global__ void __miss__occlusion()
{
    optixSetPayload_0( 0 );
}

extern "C" __global__ void __closesthit__ch()
{
    Payload p;
    p.getSeed();

    // fetch current triangle vertices
    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
        optixGetRayTime(), data );

    // compute triangle normal
    data[1] -= data[0];
    data[2] -= data[0];
    float3 normal = make_float3(
        data[1].y*data[2].z - data[1].z*data[2].y,
        data[1].z*data[2].x - data[1].x*data[2].z,
        data[1].x*data[2].y - data[1].y*data[2].x );
    const float s = 0.5f / sqrtf( normal.x*normal.x + normal.y*normal.y + normal.z*normal.z );

    float shade = 1.0f;
    if( params.ao )
    {
        const float z1 = rnd( p.seed );
        const float z2 = rnd( p.seed );

        unsigned int occluded = 1;
        float3 w_in;
        cosine_sample_hemisphere( z1, z2, w_in );
        float3 wn = normalize( optixTransformNormalFromObjectToWorldSpace( 2.f * s*normal ) );
        wn = faceforward( wn, -optixGetWorldRayDirection(), wn );
        Onb onb( wn );
        onb.inverse_transform( w_in );

        float3 pHit = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();

        optixTrace(
            params.handle,
            pHit + wn*0.001f, w_in,
            0.00f, 1e16f, optixGetRayTime(),  // tmin, tmax, time
            0xff,
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 0, // no hit group will even be executed (assuming no IS), hence, set stride and offset to 0
            1,    // select MS program
            occluded ); // this is inout here! If MS is called, it will override the payload

        if( occluded )
            shade = 0.f;
    }

    HitGroupData* rt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    // convert normal to color and store in payload
    p.c = shade * ( normal * s + make_float3( 0.5f ) ) * rt_data->color;

    p.setAll();
}
