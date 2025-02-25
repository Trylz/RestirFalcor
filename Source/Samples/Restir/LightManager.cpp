#include "LightManager.h"
namespace Restir
{
LightManager::LightManager(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene)
{
    // Create a light at center of scene
    Light light;
    light.mWsPosition = pScene->getSceneBounds().center();
    light.mRadiance = Falcor::float3(0.8f, 0.8f, 0.8f) * 4.0f;
    light.mRadius = 0.0001f;
    light.mfallOff = (light.mRadius * light.mRadius) * std::exp(1.0f / 0.0001f);
    mLights.push_back(light);

    // Create GPU buffer
}

} // namespace Restir
