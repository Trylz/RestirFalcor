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
    float temporalWsRadiusThreshold;
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





//------------------------------------------------------------------------------------------------------------------------------------
//====================================================================================================================================

struct RTXDI_DISpatioTemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint numSamples;

    // Number of neighbor pixels considered when there is no temporal surface (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    uint numDisocclusionBoostSamples;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;

    // Enables the comparison of surface materials before taking a surface into resampling.
    bool enableMaterialSimilarityTest;

    // Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors
    bool discountNaiveSamples;

    // Random number for permutation sampling that is the same for all pixels in the frame
    uint uniformRandomNumber;
};







//====================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------





















[numthreads(16, 16, 1)] void TemporalFilteringPass(uint3 threadId
                                        : SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    if (gCurrentPositionWs[pixel].w == 0.0f)
        return;

    const float3 currP = gCurrentPositionWs[pixel].xyz;

    int2 previousPixelPos = getPreviousFramePixelPos(float4(currP, 1.0f), (float)viewportDims.x, (float)viewportDims.y);
    if (previousPixelPos.x < 0 || previousPixelPos.x >= (int)viewportDims.x)
        return;
    if (previousPixelPos.y < 0 || previousPixelPos.y >= (int)viewportDims.y)
        return;
    if (gPreviousPositionWs[previousPixelPos].w == 0.0f)
        return;

    const float3 prevP = gPreviousPositionWs[previousPixelPos].xyz;
    if (length(prevP - currP) > temporalWsRadiusThreshold)
        return;

    const float3 currN = gCurrentNormalWs[pixel].xyz;
    const float3 prevN = gPreviousNormalWs[previousPixelPos].xyz;
    if (length(prevN - currN) > temporalNormalThreshold)
        return;

    const float3 V = normalize(currP - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    // Current pixel
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    // Previous pixel
    const uint previousPixelLinearIndex = previousPixelPos.y * viewportDims.x + previousPixelPos.x;
    const uint previousPixelReservoirsStart = previousPixelLinearIndex * nbReservoirPerPixel;

    // Init rng
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel.xy, sampleIndex);

    // Combine reservoirs
    for (uint i = 0; i < nbReservoirPerPixel; ++i)
    {
        RestirReservoir currentReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + i];

        RestirReservoir previousReservoir = gPreviousFrameReservoirs[previousPixelReservoirsStart + i];
        if (motion > 0)
        {
            // Clamp M according to paper.
            previousReservoir.mM = min(10 * currentReservoir.mM, previousReservoir.mM);
        }

        gCurrentFrameReservoirs[currentPixelReservoirsStart + i] =
            combineReservoirs(currentReservoir, previousReservoir, currP, currN, V, diffuse, specular, roughness, rng);
    }
}

