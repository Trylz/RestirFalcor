#include "LightManager.h"
namespace Restir
{
LightManager::LightManager(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene)
{
    //------------------------------------------------------------------------------------------------------------
    //	Create a light at center of scene
    //------------------------------------------------------------------------------------------------------------
    Light light;
    light.mWsPosition = pScene->getSceneBounds().center();
    light.mRadiance = Falcor::float3(0.8f, 0.8f, 0.8f) * 4.0f;
    light.mRadius = 0.0001f;
    light.mfallOff = std::min((light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f), 1.0f);
    mLights.push_back(light);

    // Light probabilities
    mGpuLightProbabilityBuffer.reserve(m_currentScene->getLights().size());

    float totalWeight = 0.0f;
    for (const auto& lightId : m_currentScene->getLights())
    {
        const auto light = Light::getLightPtr_FromEntity(lightId);
        const Spectrum lightFinalColor = light->computeFinalColor();
        const float lightLuma = luma(lightFinalColor);

        m_restirLightProbabilities.push_back(lightLuma);
        totalWeight += lightLuma;
    }

    for (float& prob : m_restirLightProbabilities)
        prob /= totalWeight;

    // Create GPU bufferS
    mpDevice->createStructuredBuffer(
        sizeof(float3),
        elemCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );
}

} // namespace Restir
