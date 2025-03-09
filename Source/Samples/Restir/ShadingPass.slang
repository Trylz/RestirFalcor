#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float3 cameraPositionWs;
    uint nbReservoirPerPixel;
    float sceneShadingLightExponent;
    float3 sceneAmbientColor;
};

RWTexture2D<float4> gOutput;
StructuredBuffer<RestirReservoir> gReservoirs;

Texture2D<float4> gPositionWs;
Texture2D<float4> gNormalWs;
Texture2D<float4> gAlbedo;
Texture2D<float4> gSpecular;

Texture2D<float4> gBlueNoise;

float3 computeColor(RestirReservoir reservoir, uint2 pixelIdx, float3 P, float3 N, float3 V, float3 diffuse, float3 specular, float roughness)
{ 
    float3 L =  reservoir.mY.mLightSamplePosition - P;
    const float Llen = length(L); 
    L /= Llen;

    float3 shading = evaluateBRDF(N, L, V, diffuse, specular, roughness);
    shading *= max(0.0f, dot(L, N));

    shading *= reservoir.mY.mIncomingRadiance;
    shading *= min(reservoir.m_W * (4.0f * gBlueNoise[pixelIdx % 470].x), 10.0f);
    shading /= Llen * Llen;

    shading /= pow(Llen, sceneShadingLightExponent); // An additional tweak ;)

    return shading;
}

[numthreads(16, 16, 1)]
void ShadingPass(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    if (gPositionWs[pixel].w == 0.0f)
    {
       gOutput[pixel] = float4(1.0f, 1.0f,1.0f, 1e8f);
       return;
    }

    const uint pixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const size_t reservoirsStart = pixelLinearIndex * nbReservoirPerPixel;

    const float3 P = gPositionWs[pixel].xyz;
    const float3 N = gNormalWs[pixel].xyz;
    const float3 V = normalize(P - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    float3 outColor = float3(0.0f, 0.0f, 0.0f);

    float hitDistance = 1e8f;
    for (uint i = 0; i <nbReservoirPerPixel; ++i)
    {
        const RestirReservoir r = gReservoirs[reservoirsStart + i];
        outColor += computeColor(r, pixel, P, N, V, diffuse, specular, roughness);

        hitDistance = min(hitDistance, r.m_hitDistance);
    }

    outColor/= nbReservoirPerPixel;
    outColor += sceneAmbientColor * diffuse;

    gOutput[pixel] = float4(outColor, hitDistance);
}


