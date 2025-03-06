#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float3 cameraPositionWs;
    uint nbReservoirPerPixel;
    uint sampleIndex;
    float spatialWsRadiusThreshold;
    float spatialNormalThreshold;
};

StructuredBuffer<RestirReservoir> gCurrentFrameReservoirs;
RWStructuredBuffer<RestirReservoir> gStagingReservoirs;

RWTexture2D<float4> gPositionWs;
RWTexture2D<float4> gNormalWs;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float4> gSpecular;

[numthreads(16, 16, 1)] void SpatialFiltering(uint3 threadId
                                        : SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    if (gPositionWs[pixel].w == 0.0f)
        return;

    const float3 currP = gPositionWs[pixel].xyz;

    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel, sampleIndex);

    const float3 currN = gNormalWs[pixel].xyz;
    const float3 V = normalize(currP - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    for (uint reservoirLocalIdx = 0; reservoirLocalIdx < nbReservoirPerPixel; ++reservoirLocalIdx)
    {
        const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
        const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

        RestirReservoir combinedReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx];

        for (uint i = 0u; i < 5u; ++i)
        {
            // See https://github.com/lindayukeyi/ReSTIR_DX12/blob/main/ReSTIR/Data/shaders/spatialReuse.hlsl
            const float radius = sqrt(sampleNext1D(rng)) * 30.0f;
            const float theta = 2.0f * 3.14159265f * sampleNext1D(rng);
            float2 neighborf = pixel.xy;
            neighborf.x += radius * cos(theta);
            neighborf.y += radius * sin(theta);

            if (neighborf.x < 0.0f || neighborf.x >= (float)viewportDims.x || neighborf.y < 0.0f || neighborf.y >= (float)viewportDims.y)
            {
                continue;
            }

            const uint2 neighborPixelPos = (uint2)neighborf;
            const float3 neighborN = gNormalWs[neighborPixelPos].xyz;
            if (length(neighborN - currN) > spatialNormalThreshold)
            {
                continue;
            }

            const float3 neighborP = gPositionWs[neighborPixelPos].xyz;
            if (length(neighborP - currP) > spatialWsRadiusThreshold)
                return;

            const uint neighborPixelLinearIndex = neighborPixelPos.y * viewportDims.x + neighborPixelPos.x;
            const uint neighborPixelReservoirsStart = neighborPixelLinearIndex * nbReservoirPerPixel;

            const RestirReservoir neighborReservoir = gCurrentFrameReservoirs[neighborPixelReservoirsStart + reservoirLocalIdx];
            combinedReservoir = combineReservoirs(combinedReservoir, neighborReservoir, currP, currN, V, diffuse, specular, roughness, rng);
        }


        gStagingReservoirs[currentPixelReservoirsStart + reservoirLocalIdx] = combinedReservoir;
    }
}

