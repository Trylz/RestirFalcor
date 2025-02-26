#include "Reservoir.slangh"

import Scene.Raytracing;
import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
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

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    const RayDesc ray = gScene.camera.computeRayPinhole(launchIndex.xy, viewportDims).toRayDesc();

    PrimaryRayData hitData;
    TraceRay(gScene.rtAccel, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, rayTypeCount, 0, ray, hitData);
}
