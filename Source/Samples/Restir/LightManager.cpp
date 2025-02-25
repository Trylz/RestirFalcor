#include "LightManager.h"
namespace Restir
{
LightManager::LightManager(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene)
{
    // Create a light at center of scene
    Light light;
    light.mRadiance = ;
    light.mRadiance = ;
    light.mRadiance = ;
    light.mRadiance = ;

    Falcor::float3 mWsPosition;
    float mRadius;
    float mRadiance;
    float mfallOff;

    Falcor::AABB sceneBounds = pScene->getSceneBounds();

    const Falcor::float3 minPoint = ;
    sceneBounds.minPoint
}

} // namespace Restir
