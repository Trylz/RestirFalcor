#include "NRDEncoding.slangh"
#include "NRD.slangh"

cbuffer PerFrameCB
{
    uint2 viewportDims;
};

RWTexture2D<float4> gInOutOutput;
Texture2D<float4> gPositionWs;

[numthreads(16, 16, 1)] void UnpackNRD(uint3 threadId
                                       : SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    const float4 P = gPositionWs[pixel];
    if (P.w != 0.0f)
    {
        gInOutOutput[pixel] = RELAX_BackEnd_UnpackRadiance(gInOutOutput[pixel]);
    }
    else
    {
        gInOutOutput[pixel] = float4(1.0f, 1.0f, 1.0f, 0.0f);
    }
}


