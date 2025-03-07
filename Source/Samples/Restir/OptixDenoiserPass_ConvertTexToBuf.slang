cbuffer GlobalCB
{
    uint2 viewportDims;
}

Texture2D<float4> gInTex;
RWBuffer<float4> gOutBuf;

[numthreads(8, 8, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    const uint bufIdx = pixel.x + pixel.y * viewportDims.x;
    gOutBuf[bufIdx] = gInTex[pixel.xy];
}

