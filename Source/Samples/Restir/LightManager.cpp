#include "LightManager.h"

namespace Restir
{
float luma(Falcor::float3 v)
{
    return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b;
}

LightManager::LightManager()
{
}

void LightManager::init(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene)
{
    //------------------------------------------------------------------------------------------------------------
    //	Create a light at center of scene
    //------------------------------------------------------------------------------------------------------------
    {
        Light light;
        light.mWsPosition = pScene->getSceneBounds().center();
        light.mRadiance = Falcor::float3(0.8f, 0.8f, 0.8f) * 4.0f;
        light.mRadius = 0.0001f;
        light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);
        mLights.push_back(light);
    }
    //------------------------------------------------------------------------------------------------------------
    //	Compute light probabilities
    //------------------------------------------------------------------------------------------------------------
    mLightProbabilities.reserve(mLights.size());

    float totalWeight = 0.0f;
    for (const Light& light : mLights)
    {
        const float lightLuma = luma(light.mRadiance);

        mLightProbabilities.push_back(lightLuma);
        totalWeight += lightLuma;
    }

    for (float& prob : mLightProbabilities)
        prob /= totalWeight;

    //------------------------------------------------------------------------------------------------------------
    //	Create lights gpu buffer
    //------------------------------------------------------------------------------------------------------------

    mGpuLightBuffer = pDevice->createStructuredBuffer(
        sizeof(Light),
        mLights.size(),
        Falcor::ResourceBindFlags::ShaderResource, Falcor::MemoryType::DeviceLocal, mLights.data(),
        false
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

} // namespace Restir
