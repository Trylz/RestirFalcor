#include "Light.slangh"
#include "Reservoir.slangh"

import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    uint2 viewportDims;
    uint sampleIndex;
    uint nbReservoirPerPixel;
    uint lightCount;
};

StructuredBuffer<RestirReservoir> gReservoirs;
StructuredBuffer<RestirLight> gLights;
StructuredBuffer<float> gLightProbabilities;

Texture2D<float4> gPositionWs;
Texture2D<float4> gNormalWs;
Texture2D<float4> gAlbedo;
Texture2D<float4> gSpecular;

struct SampleToLight
{
	Math::Vec3 L;
	nbFloat32 length;
	nbFloat32 intensityMultiplier;
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

    return Math::Vec3(x, r1, z);
}

SampleToLight generateSampleTolight(float3 P, RestirLight light, inout TinyUniformSampleGenerator rng)
{
		const Math::Vec3 lightToP = P - readPosition();

		const nbFloat32 distToLight = glm::length(lightToP);
		if (distToLight > m_range)
			return SampleToLight(false);

		const nbFloat32 r1 = randGenerator.generateSignedNormalized();
		const nbFloat32 r2 = randGenerator.generateUnsignedNormalized();
	
		Math::Vec3 lightToSample = Math::uniformSphericalSample(r1, r2);
		if (glm::dot(lightToP, lightToSample) / distToLight < 0.0f)
		{
			// Sample is on the back hemisphere. Take the opposite one.
			lightToSample *= -1;
		}

		SampleToLight sample;
		sample.L = -lightToP + (lightToSample * m_radius);
		sample.length = glm::length(sample.L);
		sample.L /= sample.length;

		sample.intensityMultiplier = m_fallOffMultiplier / (sample.length * sample.length);
		sample.intensityMultiplier = std::min(sample.intensityMultiplier, 1.0f);
}

RestirReservoir RIS(uint2 pixel)
{
    TinyUniformSampleGenerator rng = TinyUniformSampleGenerator(threadId.xy, sampleIndex);

    RestirReservoir r;
    initReservoir(r);

    for (uint i = 0; i < 32; ++i)
	{
		// First randomly select a light.
        const float rand = sampleNext1D(rng);

		uint lightIndex =  rand * (float)(lightCount - 1.0f));
        //lightIndex = min(lightIndex,lightCount - 1u);


		// Get light entity id
		const EntityIdentifier lightEntityId = lights[lightIndex];

		// Read light from memory.
		const auto light = Entity::EntityDatabaseSingleton::instance()->fastGetEntityRawPtr<Light::BaseLight>(lightEntityId);

		if (light->getType() == Graphics::Light::LightType::Directionnal)
			continue;//Not yet supported.

		// Generate a random sample to light
		const auto sampleToLight = light->generateSampleToLight(rng, isectProps.P);
		if (!sampleToLight.canProcess)
			continue;

		const nbFloat32 NoL = glm::dot(isectProps.BsdfProps.N, sampleToLight.L);
		if (NoL <= 0.0f)
			continue;

		// Compute light final color
		const Spectrum lightFinalColor = light->computeFinalColor();

		// Read light probability
		const nbFloat32 px = m_restirLightProbabilities[lightIndex];

		// Compute sample pobability. According to paper BRDF * Le * G(x) 
		Spectrum ppxSpectrum = lightFinalColor;
		{
			ppxSpectrum *= sampleToLight.intensityMultiplier;//G(x) without NoL

			// Build the bsdf sample struture
			Material::BaseMaterial::SampleBsdfData bsdfSample;
			bsdfSample.L = sampleToLight.L;
			bsdfSample.V = isectProps.V;
			bsdfSample.NoL = NoL;
			bsdfSample.geoProps = &isectProps.BsdfProps;

			// Sample the brdf to get the scale
			const RGBFColor sampledBrdf = material.sampleBsdf(materialColorCache, bsdfSample);
			ppxSpectrum *= sampledBrdf;//BRDF*NoL
		}

		// The sample probability.
		const nbFloat32 ppx = luma(ppxSpectrum);

		// Build the sample.
		RestirSample xi;
		xi.geometryPos = isectProps.deltaP;
		xi.lightSamplePosition = xi.geometryPos + (sampleToLight.L * sampleToLight.length);
		xi.incomingRadiance = lightFinalColor;

		// Update the reservoir with brand new sample.
		r.update(xi, ppx / px, rng);
	}

}

[numthreads(16, 16, 1)]
void EntyPoint(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId.xy > viewportDims))
        return;

    const uint pixelLinearIndex = threadId.y * viewportDims.x + threadId.x;
    const size_t reservoirsStart = pixelLinearIndex * nbRervoirPerPixel;

    for (uint i = 0; i <nbReservoirPerPixel; ++i)
    {
        gReservoirs[reservoirsStart + i] = RIS(threadId.xy);
    }
}

