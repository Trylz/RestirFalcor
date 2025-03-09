#include "LightManager.h"

namespace Restir
{
float luma(Falcor::float3 v)
{
    return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b;
}

LightManager::LightManager() {}

void LightManager::init(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene, SceneName sceneName)
{
    //------------------------------------------------------------------------------------------------------------
    //	Create lights
    //------------------------------------------------------------------------------------------------------------
    switch (sceneName)
    {
    case SceneName::Arcade:
        createArcadeSceneLights(pScene);
        break;

    case SceneName::DragonBuddha:
        createDragonBuddhaSceneLights(pScene);
        break;

    case SceneName::Sponza:
        createSponzaSceneLights(pScene);
        break;
    }

    //------------------------------------------------------------------------------------------------------------
    //	Compute light probabilities
    //------------------------------------------------------------------------------------------------------------
    mLightProbabilities.reserve(mLights.size());

    float totalWeight = 0.0f;
    for (const Light& light : mLights)
    {
        const float lightLuma = luma(light.mColor);

        mLightProbabilities.push_back(lightLuma);
        totalWeight += lightLuma;
    }

    for (float& prob : mLightProbabilities)
        prob /= totalWeight;

    //------------------------------------------------------------------------------------------------------------
    //	Create lights gpu buffer
    //------------------------------------------------------------------------------------------------------------

    mGpuLightBuffer = pDevice->createStructuredBuffer(
        sizeof(Light), mLights.size(), Falcor::ResourceBindFlags::ShaderResource, Falcor::MemoryType::DeviceLocal, mLights.data(), false
    );

    //------------------------------------------------------------------------------------------------------------
    //	Create light probalities gpu buffer
    //------------------------------------------------------------------------------------------------------------

    mGpuLightProbabilityBuffer = pDevice->createStructuredBuffer(
        sizeof(float),
        mLightProbabilities.size(),
        Falcor::ResourceBindFlags::ShaderResource,
        Falcor::MemoryType::DeviceLocal,
        mLightProbabilities.data(),
        false
    );
}

void LightManager::createArcadeSceneLights(Falcor::ref<Falcor::Scene> pScene)
{
    static const uint32_t nbLights = 2u;

    const Falcor::AABB& aabb = pScene->getSceneBounds();

    const float epsilon = aabb.radius() / 10.0f;

    const Falcor::float3 minPoint = aabb.minPoint + (aabb.maxPoint - aabb.minPoint) * epsilon;
    const Falcor::float3 maxPoint = aabb.maxPoint + (aabb.minPoint - aabb.maxPoint) * epsilon;
    const Falcor::float3 extent = maxPoint - minPoint;

    FloatRandomNumberGenerator rng(333);

    for (uint32_t i = 0u; i < nbLights; ++i)
    {
        Light light;

        // Init light.
        light.mRadius = 0.0001f;
        light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);

        // Color
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) * 50.0f /
            (float)nbLights;

        // Position
        const float dx = rng.generateUnsignedNormalized();
        const float dy = rng.generateUnsignedNormalized();
        const float dz = rng.generateUnsignedNormalized();
        light.mWsPosition = minPoint + extent * Falcor::float3(dx, dy, dz);

        mLights.push_back(light);
    }

     {
        Light light;
        light.mRadius = 0.0001f;
        light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);
        light.mColor =
            Falcor::float3(0.0f, 0.5f, 0.0f);
        light.mWsPosition = Falcor::float3(-0.335609f, 1.04073f, 0.507941f);
        mLights.push_back(light);
    }
}

void LightManager::createDragonBuddhaSceneLights(Falcor::ref<Falcor::Scene> pScene)
{
    {
        Light light;
        light.mRadius = 0.1f;
        light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);
        light.mColor = Falcor::float3(1.0f, 0.2f, 0.32f) * 16.0f;
        light.mWsPosition = pScene->getSceneBounds().center();
        light.mWsPosition.y += 1.0f;
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = 0.1f;
        light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);
        light.mColor = Falcor::float3(0.46f, 0.7f, 0.32f) * 16.0f;
        light.mWsPosition = Falcor::float3(-1.69987f, 1.27152f, 2.65488f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = 0.1f;
        light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);
        light.mColor = Falcor::float3(0.1f, 0.5f, 0.9f) * 16.0f;
        light.mWsPosition = Falcor::float3(1.63738f, 1.7456f, 2.72011f);
        mLights.push_back(light);
    }
}


void LightManager::spawnSponzaSceneLights(
    const Falcor::float3& startPt,
    const Falcor::float3& endPt,
    FloatRandomNumberGenerator& rng,
    Falcor::ref<Falcor::Scene> pScene
)
{
    const uint32_t nbLightsAlongSegment = 4;

    const float lightRadius = 0.001f;
    const float lightFalloff = std::min((lightRadius * lightRadius) * std::exp(1.0f / 0.0001f), 1.0f);

    const Falcor::float3 extents = endPt - startPt;
    const Falcor::float3 delta = extents / (float)nbLightsAlongSegment;

    for (uint32_t i = 0; i < nbLightsAlongSegment; ++i)
    {
        const Falcor::float3 posWs = startPt + (float)i * delta;

        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) * 100.0f /
            (float)nbLightsAlongSegment;
        light.mWsPosition = posWs;
        mLights.push_back(light);
    }
}

void LightManager::createSponzaSceneLights(Falcor::ref<Falcor::Scene> pScene)

{
    FloatRandomNumberGenerator rng(444);

    {
        const Falcor::float3 startPt(1.31626f, 1.86929f, 4.47785f);
        const Falcor::float3 endPt(-13.3487f, 2.38799f, 5.24492f);
        spawnSponzaSceneLights(startPt, endPt, rng, pScene);
    }

}

} // namespace Restir
