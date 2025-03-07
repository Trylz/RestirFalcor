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


// Compares two values and returns true if their relative difference is lower than the threshold.
// Zero or negative threshold makes test always succeed, not fail.
bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold)
{
    return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
}

// See if we will reuse this neighbor or history sample using
//    edge-stopping functions (e.g., per a bilateral filter).
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


    const int2 previousPixelPos = getPreviousFramePixelPos(float4(currP, 1.0f), (float)viewportDims.x, (float)viewportDims.y);
 
    if (gPreviousPositionWs[previousPixelPos].w == 0.0f)
        return;


    const float3 currN = gCurrentNormalWs[pixel].xyz;
    const float3 V = normalize(currP - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;
    const float currlinearDepth = gCurrentNormalWs[pixel].w;

    // Current pixel
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel.xy, sampleIndex);

    for (uint reservoirLocalIdx = 0u; reservoirLocalIdx < nbReservoirPerPixel; ++reservoirLocalIdx)
    {
        const RestirReservoir currentReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx];

        // Find a temporal surface
        RestirReservoir temporalNeighborReservoir;
        initReservoir(temporalNeighborReservoir);

        bool foundTemporalSurface = false;
        const float temporalSearchRadius = 16.0f; 

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

            // Should we just clamp this?--------------------------------------------------------------------------------------
            if (centralIdx.x < 0 || centralIdx.x >= (int)viewportDims.x)
                continue;
            if (centralIdx.y < 0 || centralIdx.y >= (int)viewportDims.y)
                continue;
            //---------------------------------------------------------------------------------------------------------------------

            const float3 prevP = gPreviousPositionWs[centralIdx].xyz;
            if (prevP.w == 0.0f)
                continue;

            const float3 prevN = gPreviousNormalWs[centralIdx].xyz;
            const float prevlinearDepth = currlinearDepth[pixel].w;

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
            // Maybe we dont need the check since reservoir weight will be zero.
            gCurrentFrameReservoirs[currentPixelReservoirsStart + i] = combineReservoirs(currentReservoir, temporalNeighborReservoir, currP, currN, V, diffuse, specular, roughness, rng);;
        }
/*




        RestirReservoir previousReservoir = gPreviousFrameReservoirs[previousPixelReservoirsStart + i];
        if (motion > 0)
        {
            // Clamp M according to paper.
            previousReservoir.mM = min(10 * currentReservoir.mM, previousReservoir.mM);
        }
*/
    }
}

