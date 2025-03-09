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

    case SceneName::BreakfastRoom:
        createBreakfastSceneLights(pScene);
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

void LightManager::createBreakfastSceneLights(Falcor::ref<Falcor::Scene> pScene)
{
    const float lightRadius = 0.1f;
    const float lightFalloff = std::min((lightRadius * lightRadius) * std::exp(1.0f / 0.0001f), 1.0f);
    const float lightIntensity = 40.0f;

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor = Falcor::float3(0.0, 1.f, 0.1f) * 20.0f* lightIntensity;
        light.mWsPosition = Falcor::float3(3.1884f, 2.68466f, -1.33084f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor = Falcor::float3(1.0f, 0.0f, 0.1f) * lightIntensity;
        light.mWsPosition = Falcor::float3(-3.80251f, 2.03401f, -1.50989f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor = Falcor::float3(0.0, 0.0f, 1.0f) * lightIntensity;
        light.mWsPosition = Falcor::float3(-0.497478f, 2.20476f, -0.623703f);
        mLights.push_back(light);
    }


    FloatRandomNumberGenerator rng(111);
    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(-1.86844f, -0.252511f, 2.93348f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(2.52178f, -0.0497445f, 4.24406f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(-4.84042f, 2.67538f, -2.06565f);
        mLights.push_back(light);
    }


    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(-2.74649f, 3.81714f, -3.12922f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(2.12604f, 3.8762f, -2.58422f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(0.988998f, 1.80315f, 1.62673f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(-2.24108f, 1.679448f, 2.02123f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(3.87917f, -0.0521774f, -1.03112f);
        mLights.push_back(light);
    }

    {
        Light light;
        light.mRadius = lightRadius;
        light.mfallOff = lightFalloff;
        light.mColor =
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity;
        light.mWsPosition = Falcor::float3(-5.01815f, -0.31212f, -1.68199f);
        mLights.push_back(light);
    }
}

void LightManager::spawnPointLightsAlongSegment(
    const Falcor::float3& startPt,
    const Falcor::float3& endPt,
    FloatRandomNumberGenerator& rng,
    float lightIntensity,
    uint32_t nbLightsAlongSegment
)
{
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
            Falcor::float3(rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized(), rng.generateUnsignedNormalized()) *
            lightIntensity /
            (float)nbLightsAlongSegment;
        light.mWsPosition = posWs;
        mLights.push_back(light);
    }
}

void LightManager::createSponzaSceneLights(Falcor::ref<Falcor::Scene> pScene)

{
    FloatRandomNumberGenerator rng(222);

    {
        const Falcor::float3 startPt(1.31626f, 1.86929f, 4.47785f);
        const Falcor::float3 endPt(-13.3487f, 2.38799f, 5.24492f);
        spawnPointLightsAlongSegment(startPt, endPt, rng, 1000.0f, 4);
    }


    {
        const Falcor::float3 startPt(13.5686f, 2.20822f, -4.80682f);
        const Falcor::float3 endPt(-13.3297f, 2.26712f, -4.96876f);
        spawnPointLightsAlongSegment(startPt, endPt, rng, 1000.0f, 4);
    }

    {
        const Falcor::float3 startPt(13.9416f, 2.45657f, 0.288835f);
        const Falcor::float3 endPt(-13.4072f, 2.36068f, 0.431062f);
        spawnPointLightsAlongSegment(startPt, endPt, rng, 1000.0f, 4);
    }

    {
        const Falcor::float3 startPt(14.4111f, 7.54439f, 5.19206f);
        const Falcor::float3 endPt(-12.7583f, 7.24114f, 5.15393f);
        spawnPointLightsAlongSegment(startPt, endPt, rng, 1000.0f, 4);
    }

    {
        const Falcor::float3 startPt(14.2571f, 7.36345f, -5.47451f);
        const Falcor::float3 endPt(-11.4741f, 7.57215f, -5.27168f);
        spawnPointLightsAlongSegment(startPt, endPt, rng, 1000.0f, 4);
    }
}

} // namespace Restir
