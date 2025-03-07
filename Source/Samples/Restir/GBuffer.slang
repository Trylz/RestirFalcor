import Scene.Raytracing;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

RWTexture2D<float4> gPositionWs;
RWTexture2D<float4> gNormalWs;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float4> gSpecular;

cbuffer PerFrameCB
{
    float2 viewportDims;
    uint sampleIndex;
};

struct PrimaryRayData
{
    bool dummy;
};

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    gPositionWs[launchIndex].w = 0.0f;
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Get the hit-point data.
    float3 rayDirW = WorldRayDirection();
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();

    // Prepare the shading data.
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, triangleIndex, attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW);

    // Create material instance and query its properties.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    let bsdfProperties = mi.getProperties(sd);

    // The launch index.
    const uint2 launchIndex = DispatchRaysIndex().xy;

    // Write the ws position.
    {
        gPositionWs[launchIndex] = float4(sd.posW, 1.f);
    }
    // Write the ws normal.
    {
        gNormalWs[launchIndex] = float4(sd.faceN, hitT);
    }

    // Write the albedo.
    {
        gAlbedo[launchIndex] = float4(bsdfProperties.diffuseReflectionAlbedo, materialID);
    }

    // Write the specular.
    {
        gSpecular[launchIndex] = float4(bsdfProperties.specularReflectionAlbedo, bsdfProperties.roughness);
    }
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    if (any(launchIndex.xy > (uint2)viewportDims))
        return;

    const RayDesc ray = gScene.camera.computeRayPinhole(launchIndex.xy, viewportDims).toRayDesc();

    PrimaryRayData hitData;
    TraceRay(gScene.rtAccel, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, rayTypeCount, 0, ray, hitData);
}

