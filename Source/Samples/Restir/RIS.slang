#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    uint sampleIndex;
    uint sampleIndex;
};

[numthreads(16, 16, 1)]
void EntyPoint(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId.xy > viewportDims))
        return;




    float metallic;
    if (metallicChannel < 0)
    {
        metallic = metallicValue;
    }
    else
    {
        metallic = sampleTexture(metallicTexture, sampler, threadId.xy, outDim)[metallicChannel];
    }

    float roughness;
    if (roughnessChannel < 0)
    {
        roughness = roughnessValue;
    }
    else
    {
        roughness = sampleTexture(roughnessTexture, sampler, threadId.xy, outDim)[roughnessChannel];
    }

    outputTexture[threadId.xy] = float4(0., saturate(roughness), saturate(metallic), 1.);
}

