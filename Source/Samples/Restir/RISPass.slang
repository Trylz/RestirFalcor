#include "Light.slangh"
#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    float3 cameraPositionWs;
    uint sampleIndex;
    uint nbReservoirPerPixel;
    uint lightCount;
    uint RISSamplesCount;
};

RWStructuredBuffer<RestirReservoir> gReservoirs;
StructuredBuffer<RestirLight> gLights;
StructuredBuffer<float> gLightProbabilities;

Texture2D<float4> gPositionWs;
Texture2D<float4> gNormalWs;
Texture2D<float4> gAlbedo;
Texture2D<float4> gSpecular;

struct SampleToLight
{
	float3 L;
	float length;
	float intensityMultiplier;
};

//@ See : https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation.html
// http://mathworld.wolfram.com/SpherePointPicking.html
// r1 = [0.0 ; 1.0f] -> Hemisphere
// r1 = [-1.0 ; 1.0f] -> Sphere
// r2 = [0.0 ; 1.0f]
inline float3 uniformSphericalSample(float r1, float r2)
{
    float sinTheta = sqrt(1 - r1 * r1);
    float phi = 6.28318530f * r2;
    float x = sinTheta * cos(phi);
    float z = sinTheta * sin(phi);

    return float3(x, r1, z);
}

SampleToLight generateSampleTolight(float3 P, RestirLight light, inout TinyUniformSampleGenerator rng)
{
	const float3 lightToP = P - light.mWsPosition;
	const float distToLight = length(lightToP);

	const float r1 = sampleNext1D(rng) * 2.0f - 1.0f;
	const float r2 = sampleNext1D(rng);
	
	float3 lightToSample = uniformSphericalSample(r1, r2);
	if (dot(lightToP, lightToSample) / distToLight < 0.0f)
	{
		// Sample is on the back hemisphere. Take the opposite one.
		lightToSample *= -1;
	}

	SampleToLight sample;
	sample.L = -lightToP + (lightToSample * light.mRadius);
	sample.length = length(sample.L);
	sample.L /= sample.length;

	sample.intensityMultiplier = light.mfallOff / (sample.length * sample.length);

	return sample;
}

RestirReservoir RIS(uint2 pixel)
{
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(pixel, sampleIndex);

    RestirReservoir r;
    initReservoir(r);

    const float3 P = gPositionWs[pixel].xyz;
    const float3 N = gNormalWs[pixel].xyz;
    const float3 V = normalize(P - cameraPositionWs);
    const float3 diffuse = gAlbedo[pixel].xyz;
    const float3 specular = gSpecular[pixel].xyz;
    const float roughness = gSpecular[pixel].w;

    for (uint i = 0; i < RISSamplesCount; ++i)
	{
		// First randomly select a light.
        const float rand = sampleNext1D(rng);

        uint lightIndex = (uint)(rand * (float)lightCount);
        lightIndex = min(lightIndex,lightCount - 1u);

        // Read the light
        const RestirLight light = gLights[lightIndex];

		// Generate a random sample to light
		const SampleToLight sampleToLight = generateSampleTolight(P, light, rng);

		// Read light probability
		const float px = gLightProbabilities[lightIndex];

		// Compute sample pobability. According to paper BRDF * Le * G(x) 
		float3 ppxSpectrum = light.mColor;
		{
			ppxSpectrum *= sampleToLight.intensityMultiplier;//G(x)

			// Sample the brdf to get the scale
			const float3 sampledBrdf = evaluateBRDF(N, sampleToLight.L, V, diffuse, specular, roughness);
			ppxSpectrum *= sampledBrdf * max(0.0f, dot(sampleToLight.L, N));
		}

		// The sample probability.
		const float ppx = luma(ppxSpectrum);

		// Build the sample.
		RestirSample xi;
		xi.mGeometryPos = P;
		xi.mLightSamplePosition = xi.mGeometryPos + (sampleToLight.L * sampleToLight.length);
		xi.mIncomingRadiance = light.mColor;

		// Update the reservoir with brand new sample.
        updateReservoir(r, rng, xi, ppx / px);
	}

	// Compute and set RIS global weight.
	{
		float3 ppxSpectrum = r.mY.mIncomingRadiance;
		{
            const float3 L = normalize(r.mY.mLightSamplePosition - r.mY.mGeometryPos);
			const float3 sampledBrdf = evaluateBRDF(N, L, V, diffuse, specular, roughness);
			ppxSpectrum *= sampledBrdf * max(0.0f, dot(L, N));
		}

		const float ppx = luma(ppxSpectrum);
		if (ppx != 0.0f)
		{
			r.m_W = r.mWsum / ((float)r.mM * ppx);
		}
		else
		{
			r.m_W = 0.0f;
		}
	}

	return r;
}

[numthreads(16, 16, 1)]
void EntryPoint(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId.xy >= viewportDims))
        return;

    if (gPositionWs[threadId.xy].w == 0.0f)
        return;

    const uint pixelLinearIndex = threadId.y * viewportDims.x + threadId.x;
    const size_t reservoirsStart = pixelLinearIndex * nbReservoirPerPixel;

    for (uint i = 0; i <nbReservoirPerPixel; ++i)
    {
        gReservoirs[reservoirsStart + i] = RIS(threadId.xy);
    }
}


