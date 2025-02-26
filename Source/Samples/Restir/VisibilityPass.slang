#include "Reservoir.slangh"

import Scene.Raytracing;

cbuffer PerFrameCB
{
    uint nbReservoirPerPixel;
    uint2 viewportDims;
};

RWStructuredBuffer<RestirReservoir> gReservoirs;

struct ShadowRayData
{
    bool hit;
};

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
        const float3 Llen = length(L);
        L /= Llen;

        const RayDesc shadowRay = { r.mY.mGeometryPos, 0.001f, L, Llen };

        ShadowRayData rayData;
        rayData.hit = true;
        TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, rayTypeCount, 1, shadowRay, rayData);

        if (rayData.hit)
            gReservoirs[reservoirsStart + i].m_W = 0.0f
    }


    uint3 launchIndex = DispatchRaysIndex();
    const RayDesc ray = gScene.camera.computeRayPinhole(launchIndex.xy, viewportDims).toRayDesc();

    PrimaryRayData hitData;
    TraceRay(gScene.rtAccel, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, rayTypeCount, 0, ray, hitData);
}
