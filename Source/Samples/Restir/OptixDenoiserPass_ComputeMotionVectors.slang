cbuffer GlobalCB
{
    uint2 viewportDims;
    float4x4 previousFrameViewProjMat;
}

Texture2D<float4> gInPositionWs;
RWBuffer<float2> gOutBuf;

int2 getPreviousFramePixelPos(float4 P, float width, float height)
{
    float4 ndc = mul(P, previousFrameViewProjMat);
    ndc = ndc / ndc.w;
    float2 s = (ndc.xy + float2(1, 1)) * 0.5f;
    s = float2(s.x, 1.f - s.y);
    return (int2)(s * float2(width, height));
}

[numthreads(8, 8, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

	const float4 P = gInPositionWs[pixel];
	const uint bufIdx = threadId.y * viewportDims.x + pixel.x;

    if (P.w == 0.0f)
    {
    	gOutBuf[bufIdx] = float2(0.0f, 0.0f);
    	return;
    }

    int2 prevPixel = getPreviousFramePixelPos(P, (float)viewportDims.x, (float)viewportDims.y);
    prevPixel = clamp(prevPixel, int2(0, 0), int2(viewportDims.x - 1, viewportDims.y - 1));

	gOutBuf[bufIdx] = (float2)(pixel - prevPixel);
}

