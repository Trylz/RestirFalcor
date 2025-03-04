#include "Light.slangh"
#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float3 cameraPositionWs;
    float4x4 previousFrameViewProjMat;
    uint nbReservoirPerPixel;
    uint sampleIndex;
    uint motion;// Maybe we wont need this is denoising does a great job
    float temporalWsRadiusThreshold;
    float temporalNormalThreshold;
};

RWStructuredBuffer<RestirReservoir> gCurrentFrameReservoirs;
StructuredBuffer<RestirReservoir> gPreviousFrameReservoirs;

Texture2D<float4> gCurrentPositionWs;
Texture2D<float4> gPreviousPositionWs;

Texture2D<float4> gCurrentNormalWs;
Texture2D<float4> gPreviousNormalWs;

Texture2D<float4> gAlbedo;
Texture2D<float4> gSpecular;

int2 getPreviousFramePixelPos(float4 P, float width, float height)
{
    float4 ndc = mul(P, previousFrameViewProjMat);
    ndc = ndc / ndc.w;
    float2 s = (ndc.xy + float2(1, 1)) * 0.5f;
    s = float2(s.x, 1.f - s.y);
    return (int2)(s * float2(width, height));
}

[numthreads(16, 16, 1)] void TemporalFilteringPass(uint3 threadId
                                        : SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    if (gCurrentPositionWs[pixel].w == 0.0f)
        return;

    const float3 currP = gCurrentPositionWs[pixel].xyz;

    int2 previousPixelPos = getPreviousFramePixelPos(float4(currP, 1.0f), (float)viewportDims.x, (float)viewportDims.y);
    if (previousPixelPos.x < 0 || previousPixelPos.x >= (int)viewportDims.x)
        return;
    if (previousPixelPos.y < 0 || previousPixelPos.y >= (int)viewportDims.y)
        return;
    if (gPreviousPositionWs[previousPixelPos].w == 0.0f)
        return;

    const float3 prevP = gPreviousPositionWs[previousPixelPos].xyz;
    if (length(prevP - currP) > temporalWsRadiusThreshold)
        return;

    const float3 currN = gCurrentNormalWs[pixel].xyz;
    const float3 prevN = gPreviousNormalWs[previousPixelPos].xyz;
    if (length(prevN - currN) > temporalNormalThreshold)
        return;

    const float3 V = normalize(currP - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    // Current pixel
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    // Previous pixel
    const uint previousPixelLinearIndex = previousPixelPos.y * viewportDims.x + previousPixelPos.x;
    const uint previousPixelReservoirsStart = previousPixelLinearIndex * nbReservoirPerPixel;

    // Init rng
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel.xy, sampleIndex);

    // Combine reservoirs
    for (uint i = 0; i < nbReservoirPerPixel; ++i)
    {
        RestirReservoir currentReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + i];

        RestirReservoir previousReservoir = gPreviousFrameReservoirs[previousPixelReservoirsStart + i];
        if (motion > 0)
        {
            // Clamp M according to paper.
            previousReservoir.mM = min(10 * currentReservoir.mM, previousReservoir.mM);
        }

        gCurrentFrameReservoirs[currentPixelReservoirsStart + i] =
            combineReservoirs(currentReservoir, previousReservoir, currP, currN, V, diffuse, specular, roughness, rng);
    }
}

