#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;
import Scene.Raytracing;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    uint nbReservoirPerPixel;
};

RWStructuredBuffer<RestirReservoir> gReservoirs;

struct PrimaryRayData
{
    bool dummy;
};

struct ShadowRayData
{
    bool hit;
};

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
}

[shader("miss")]
void shadowMiss(inout ShadowRayData hitData)
{
    hitData.hit = false;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.hit = true;
}

[shader("raygeneration")]
void rayGen()
{
    uint3 threadId = DispatchRaysIndex();
    if (any(threadId.xy > (uint2)viewportDims))
        return;

    const uint pixelLinearIndex = threadId.y * viewportDims.x + threadId.x;
    const size_t reservoirsStart = pixelLinearIndex * nbReservoirPerPixel;

    for (uint i = 0; i <nbReservoirPerPixel; ++i)
    {
        const RestirReservoir r = gReservoirs[reservoirsStart + i] ;

        float3 L = r.mY.mLightSamplePosition - r.mY.mGeometryPos;
        const float Llen = length(L);
        L /= Llen;

        RayDesc ray;
        ray.Origin = r.mY.mGeometryPos;
        ray.Direction = L;
        ray.TMin = 0.001f;
        ray.TMax = Llen;

        ShadowRayData rayData;
        rayData.hit = true;
        TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, rayTypeCount, 1, ray, rayData);

        if (rayData.hit)
        {
            gReservoirs[reservoirsStart + i].m_W = 0.0f;
            gReservoirs[reservoirsStart + i].m_hitDistance = Llen;
        }
        else
        {
            gReservoirs[reservoirsStart + i].m_hitDistance = 1e8f;
        }
    }
}


