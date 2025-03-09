#include "NRDEncoding.slangh"
#include "NRD.slangh"

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float4x4 viewMat;
    float4x4 previousFrameViewProjMat;
};

RWTexture2D<float4> gRadianceHit;
RWTexture2D<float4> gNormalLinearRoughness;
RWTexture2D<float> gViewZ;
RWTexture2D<float2> gMotionVector;

Texture2D<float4> gPositionWs;
Texture2D<float4> gAlbedo;
Texture2D<float4> gNormalWs;
Texture2D<float4> gSpecular;

int2 getPreviousFramePixelPos(float4 P, float width, float height)
{
    float4 ndc = mul(P, previousFrameViewProjMat);
    ndc = ndc / ndc.w;
    float2 s = (ndc.xy + float2(1, 1)) * 0.5f;
    s = float2(s.x, 1.f - s.y);
    return (int2)(s * float2(width, height));
}

[numthreads(16, 16, 1)]
void PackNRD(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    // Radiance hit
    gRadianceHit[pixel] = RELAX_FrontEnd_PackRadianceAndHitDist(gRadianceHit[pixel].xyz, gRadianceHit[pixel].w, true);

    // Normal roughness
    gNormalLinearRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(gNormalWs[pixel].xyz, gSpecular[pixel].w, gAlbedo[pixel].w);

    // View Z and MVs
    const float4 P = gPositionWs[pixel];
    if (P.w!= 0.0f)
    {
        gViewZ[pixel] = mul(P, viewMat).z;

        int2 prevPixel = getPreviousFramePixelPos(P, (float)viewportDims.x, (float)viewportDims.y);
        prevPixel = clamp(prevPixel, int2(0, 0), int2(viewportDims.x - 1, viewportDims.y - 1));
        gMotionVector[pixel] = (float2)(pixel - prevPixel);
    }
    else
    {
        gViewZ[pixel] = 1e8f; // or 0.0?
        gMotionVector[pixel] = 0.0f;
    }
}


