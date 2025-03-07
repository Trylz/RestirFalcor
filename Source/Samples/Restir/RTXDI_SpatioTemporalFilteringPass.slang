#include "Light.slangh"
#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float3 cameraPositionWs;

    float4x4 previousFrameViewProjMat;
    float4x4 currentFrameViewProjMat;

    float4x4 previousFrameViewMat;
    float4x4 currentFrameViewMat;

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
   // uint sourceBufferIndex;

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
   // bool enableMaterialSimilarityTest;

    // Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors
    bool discountNaiveSamples;

    // Random number for permutation sampling that is the same for all pixels in the frame
   // uint uniformRandomNumber;
};

bool RTXDI_IsActiveCheckerboardPixel(uint2 pixelPosition, bool previousFrame, uint activeCheckerboardField)
{
    if (activeCheckerboardField == 0)
        return true;

    return ((pixelPosition.x + pixelPosition.y + int(previousFrame)) & 1) == (activeCheckerboardField & 1);
}

void RTXDI_ActivateCheckerboardPixel(inout uint2 pixelPosition, bool previousFrame, uint activeCheckerboardField)
{
    if (RTXDI_IsActiveCheckerboardPixel(pixelPosition, previousFrame, activeCheckerboardField))
        return;

    if (previousFrame)
        pixelPosition.x += int(activeCheckerboardField) * 2 - 3;
    else
        pixelPosition.x += (pixelPosition.y & 1) != 0 ? 1 : -1;
}

struct GBufferSurface
{
    float4 P;
    float3 N;
    float3 V;
    float3 diffuse;
    float3 specular;
    float roughness;
};

bool RAB_IsSurfaceValid(GBufferSurface surface)
{
    return surface.P.w != 0.0f;
}

GBufferSurface RAB_GetCurrentGBufferSurface(int2 pixel)
{
    GBufferSurface surface;
    surface.P = gCurrentPositionWs[pixel];
    surface.N = gCurrentNormalWs[pixel].xyz;
    surface.V = normalize(surface.P.xyz - cameraPositionWs);
    surface.diffuse = gAlbedo[pixel].xyz;
    surface.specular = gSpecular[pixel].xyz;
    data.roughness = gSpecular[pixel].w;

    return data;
}

GBufferSurface RAB_GetPreviousGBufferSurface(int2 pixel)
{
    GBufferSurface data;
    /*** TODOOO YANN*/
    return data;
}

float3 RAB_GetSurfaceNormal(GBufferSurface surface)
{
    return surface.N;
}

float RAB_GetCurrentSurfaceLinearDepth(GBufferSurface gData)
{
    return mul(gData.P, currentFrameViewMat).z;
}

float RAB_GetPreviousSurfaceLinearDepth(GBufferSurface gData)
{
    return mul(gData.P, currentFrameViewMat).z;
}

bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold)
{
    return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
}

bool RTXDI_IsValidNeighbor(float3 ourNorm, float3 theirNorm, float ourDepth, float theirDepth, float normalThreshold, float depthThreshold)
{
    return (dot(theirNorm.xyz, ourNorm.xyz) >= normalThreshold) && RTXDI_CompareRelativeDifference(ourDepth, theirDepth, depthThreshold);
}

RestirReservoir RTXDI_LoadCurrentFrameReservoir(int2 pixel, int reservoirLocalIdx)
{
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    return gCurrentFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx];
}

RestirReservoir RTXDI_LoadPreviousFrameReservoir(int2 pixel, int reservoirLocalIdx)
{
    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    return gPreviousFrameReservoirs[currentPixelReservoirsStart + reservoirLocalIdx];
}


RestirReservoir RTXDI_DISpatioTemporalResamplingWithPairwiseMIS(
    uint2 pixelPosition,
    RestirReservoir curSample,
    RTXDI_DISpatioTemporalResamplingParameters stparams,
    GBufferSurface surface,
    inout TinyUniformSampleGenerator rng
)
{
    const uint historyLimit = min(0x3fff, uint(stparams.maxHistoryLength * curSample.mM));

    float3 motion = stparams.screenSpaceMotion;
    //if (!stparams.enablePermutationSampling)
    {
        motion.xy += float2(sampleNext1D(rng), sampleNext1D(rng)) - 0.5f;
    }

    const int2 prevPos = int2(round(float2(pixelPosition) + motion.xy));

    const float expectedPrevLinearDepth = RAB_GetCurrentSurfaceLinearDepth(surface) + motion.z;

    const int activeCheckerboardField = 0;

    // Some default initializations
    temporalSamplePixelPos = int2(-1, -1);
    RAB_Surface temporalSurface;
    bool foundTemporalSurface = false;                                                // Found a valid backprojection?
    const float temporalSearchRadius = (activeCheckerboardField == 0) ? 4 : 8; // How far to search for a match when backprojecting
    int2 temporalSpatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the centrol reprojected pixel
    int i;
    int2 centralIdx;
    for (i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        offset.x = (i > 0) ? int((sampleNext1D(rng) - 0.5f) * temporalSearchRadius) : 0;
        offset.y = (i > 0) ? int((sampleNext1D(rng) - 0.5f) * temporalSearchRadius) : 0;

        centralIdx = prevPos + offset;
        /*
            if (stparams.enablePermutationSampling && i == 0)
        {
            RTXDI_ApplyPermutationSampling(centralIdx, stparams.uniformRandomNumber);
        }

        */
        RTXDI_ActivateCheckerboardPixel(centralIdx, true, activeCheckerboardField);

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetPreviousGBufferSurface(centralIdx);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;

        // Test surface similarity, discard the sample if the surface is too different.
        if (!RTXDI_IsValidNeighbor(
                RAB_GetSurfaceNormal(surface),
                RAB_GetSurfaceNormal(temporalSurface),
                expectedPrevLinearDepth,
                RAB_GetPreviousSurfaceLinearDepth(temporalSurface),
                stparams.normalThreshold,
                stparams.depthThreshold
            ))
            continue;

        temporalSpatialOffset = centralIdx - prevPos;
        foundTemporalSurface = true;
        break;
    }

    // How many spatial samples to use?  









//====================================================================================================================================
//------------------------------------------------------------------------------------------------------------------------------------


#define RTXDI_BIAS_CORRECTION_BASIC 1

[numthreads(16, 16, 1)] void TemporalFilteringPass(uint3 threadId
                                        : SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= viewportDims))
        return;

    if (gCurrentPositionWs[pixel].w == 0.0f)
        return;

    const float3 P = gCurrentPositionWs[pixel].xyz;
    const float3 N = gCurrentNormalWs[pixel].xyz;
    const float3 V = normalize(P - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    const uint currentPixelLinearIndex = pixel.y * viewportDims.x + pixel.x;
    const uint currentPixelReservoirsStart = currentPixelLinearIndex * nbReservoirPerPixel;

    // Init rng
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel.xy, sampleIndex);

    // Combine reservoirs
    for (uint i = 0; i < nbReservoirPerPixel; ++i)
    {
        RestirReservoir currentReservoir = gCurrentFrameReservoirs[currentPixelReservoirsStart + i];

        RTXDI_DISpatioTemporalResamplingParameters stparams;
       // stparams.screenSpaceMotion = primary.motionVector; YANN to compute
        stparams.maxHistoryLength = 20;
        stparams.biasCorrectionMode = RTXDI_BIAS_CORRECTION_BASIC;//  YANN really?
        stparams.depthThreshold = 0.1f;
        stparams.normalThreshold = 0.5f;
        stparams.numSamples = 32;// YANN restir M
        stparams.numDisocclusionBoostSamples = 0;
        stparams.samplingRadius = 32;
        stparams.enableVisibilityShortcut = true;
        stparams.enablePermutationSampling = false;//YANN
        stparams.discountNaiveSamples = false;

        gCurrentFrameReservoirs[currentPixelReservoirsStart + i] =
            RTXDI_DISpatioTemporalResamplingWithPairwiseMIS(pixel, currentReservoir, stparams, P, N, V, diffuse, specular, roughness, rng);
    }
}

