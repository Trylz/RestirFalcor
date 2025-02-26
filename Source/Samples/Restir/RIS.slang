#include "Light.slangh"
#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    uint sampleIndex;
    uint nbRervoirPerPixel;
    uint lightCount;
};

StructuredBuffer<RestirReservoir> gReservoirs;
StructuredBuffer<RestirLight> gLights;

Texture2D<float4> gPositionWs;
Texture2D<float4> gNormalWs;
Texture2D<float4> gAlbedo;
Texture2D<float4> gSpecular;

RestirReservoir RIS(uint2 pixel, inout TinyUniformSampleGenerator rng)
{
}

[numthreads(16, 16, 1)]
void EntyPoint(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId.xy > viewportDims))
        return;

    const uint pixelLinearIndex = threadId.y * viewportDims.x + threadId.x;
    const size_t reservoirsStart = pixelLinearIndex * nbRervoirPerPixel;

    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(threadId.xy, sampleIndex);

    for (uint i = 0; i <nbRervoirPerPixel; ++i)
    {
        gReservoirs[reservoirsStart + i] = RIS(threadId.xy, rng);
    }
}

