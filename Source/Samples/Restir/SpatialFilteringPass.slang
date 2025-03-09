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
        const float spatialSearchRadius = 8.0f;

        const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
        const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;
        const RestirReservoir currentPixelCombinedReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx];

        // Try to find a matching surface in the neighborhood of the centrol reprojected pixel
        // See
        // https://github.com/NVIDIA-RTX/RTXDI-Library/blob/26a6c715aefbc4056954bd1c7e537b8a9ef8984c/Include/Rtxdi/DI/SpatioTemporalResampling.hlsli
        int i;
        int2 centralIdx;

        for (i = 0; i < 9; i++)
        {
            int2 offset = int2(0, 0);
            offset.x = int((sampleNext1D(rng) - 0.5f) * spatialSearchRadius);
            offset.y = int((sampleNext1D(rng) - 0.5f) * spatialSearchRadius);

            centralIdx = (int2)pixel + offset;

            if (centralIdx.x < 0 || centralIdx.x >= (int)viewportDims.x)
                continue;
            if (centralIdx.y < 0 || centralIdx.y >= (int)viewportDims.y)
                continue;

            const float4 neighborP = gPositionWs[centralIdx];
            if (neighborP.w == 0.0f)
                continue;

            const float3 neighborN = gNormalWs[centralIdx].xyz;
            if (length(neighborN - currN) > spatialNormalThreshold)
                return;

            const uint centralPixelLinearIndex = centralIdx.y * viewportDims.x + centralIdx.x;
            const uint centralPixelReservoirsStart = centralPixelLinearIndex * nbReservoirPerPixel;
            const RestirReservoir spatialNeighborReservoir = gCurrentFrameReservoirs[centralPixelReservoirsStart + reservoirLocalIdx];

            combineReservoirs(currentPixelCombinedReservoir, spatialNeighborReservoir, currP, currN, V, diffuse, specular, roughness, rng);
        }

        gStagingReservoirs[currentPixelReservoirsStart + reservoirLocalIdx] = currentPixelCombinedReservoir;
    }
}

