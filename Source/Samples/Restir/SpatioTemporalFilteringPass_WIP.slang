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
    uint motion;
    float temporalLinearDepthThreshold;
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

bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold)
{
    return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
}

bool RTXDI_IsValidNeighbor(float3 ourNorm, float3 theirNorm, float ourDepth, float theirDepth, float normalThreshold, float depthThreshold)
{
    return (dot(theirNorm.xyz, ourNorm.xyz) >= normalThreshold)
        && RTXDI_CompareRelativeDifference(ourDepth, theirDepth, depthThreshold);
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
    const float3 currN = gCurrentNormalWs[pixel].xyz;
    const float3 V = normalize(currP - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;
    const float currlinearDepth = gCurrentNormalWs[pixel].w;

    // The previous pixel cposition for this worldspace coordinate.
    const int2 previousPixelPos = getPreviousFramePixelPos(float4(currP, 1.0f), (float)viewportDims.x, (float)viewportDims.y);

    // Current pixel reservoir info.
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    // Init the random number generator.
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel.xy, sampleIndex);

    for (uint reservoirLocalIdx = 0u; reservoirLocalIdx < nbReservoirPerPixel; ++reservoirLocalIdx)
    {
        // Find a temporal surface
        RestirReservoir temporalNeighborReservoir;
        initReservoir(temporalNeighborReservoir);
        const float temporalSearchRadius = 32.0f; // Maybe use a % of viewportDims?

        // Try to find a matching surface in the neighborhood of the centrol reprojected pixel
        // See https://github.com/NVIDIA-RTX/RTXDI-Library/blob/26a6c715aefbc4056954bd1c7e537b8a9ef8984c/Include/Rtxdi/DI/SpatioTemporalResampling.hlsli
        int i;
        int2 centralIdx;

        for (i = 0; i < 9; i++)
        {
            int2 offset = int2(0, 0);
            offset.x = (i > 0) ? int((sampleNext1D(rng) - 0.5f) * temporalSearchRadius) : 0;
            offset.y = (i > 0) ? int((sampleNext1D(rng) - 0.5f) * temporalSearchRadius) : 0;

            centralIdx = previousPixelPos + offset;

            if (centralIdx.x < 0 || centralIdx.x >= (int)viewportDims.x)
                continue;
            if (centralIdx.y < 0 || centralIdx.y >= (int)viewportDims.y)
                continue;

            const float4 prevP = gPreviousPositionWs[centralIdx];
            if (prevP.w == 0.0f)
                continue;

            if (length(prevP.xyz - currP) > temporalWsRadiusThreshold)
                return;

            const float3 prevN = gPreviousNormalWs[centralIdx].xyz;
            if (length(prevN - currN) > temporalNormalThreshold)
                return;

            const float prevlinearDepth = gPreviousNormalWs[pixel].w;

            if (!RTXDI_IsValidNeighbor(
                currN, prevN,
                prevlinearDepth, currlinearDepth,
                temporalNormalThreshold, temporalLinearDepthThreshold))
                continue;

            const uint centralPixelLinearIndex = centralIdx.y * viewportDims.x + centralIdx.x;
            const uint centralPixelReservoirsStart = centralPixelLinearIndex * nbReservoirPerPixel;
            temporalNeighborReservoir = gPreviousFrameReservoirs[centralPixelReservoirsStart + reservoirLocalIdx];

            break;
        }

        if (temporalNeighborReservoir.m_W > 0.0f)
        {
            const RestirReservoir currentPixelReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx];
            if (motion > 0)
            {
                // Clamp M according to paper.
                temporalNeighborReservoir.mM = min(20 * currentPixelReservoir.mM, temporalNeighborReservoir.mM);
            }

            gCurrentFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx] = combineReservoirs(currentPixelReservoir, temporalNeighborReservoir, currP, currN, V, diffuse, specular, roughness, rng);;
        }
    }
}

