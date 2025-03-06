cbuffer GlobalCB
{
    uint2 viewportDims;
    float4x4 gViewIT;
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

    float3 normal = gInTex[pixel].xyz;
    if (length(normal) > 0.01f)
    {
        normal = mul(gViewIT, float4(normal, 0.0f)).xyz;
        normal = normalize(normal);
    }

    gOutBuf[bufIdx] = float4(normal, 0.0f);
}

