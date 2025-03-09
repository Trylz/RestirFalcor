#pragma once

#include "SceneName.h"
#include "Singleton.h"
#include "FloatRandomNumberGenerator.h"

namespace Restir
{
struct Light
{
    Falcor::float3 mWsPosition;
    Falcor::float3 mColor;
    float mRadius;
    float mfallOff;
};

struct LightManager
{
    LightManager();
    void init(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene, SceneName sceneName);

    inline const std::vector<Light>& getLights() const { return mLights; }
    inline const Falcor::ref<Falcor::Buffer>& getLightGpuBuffer() const { return mGpuLightBuffer; }

    inline const std::vector<float>& getLightProbabilities() const { return mLightProbabilities; }
    inline const Falcor::ref<Falcor::Buffer>& getLightProbabilitiesGpuBuffer() const { return mGpuLightProbabilityBuffer; }

private:
    void createArcadeSceneLights(Falcor::ref<Falcor::Scene> pScene);
    void createDragonBuddhaSceneLights(Falcor::ref<Falcor::Scene> pScene);
    void createBreakfastSceneLights(Falcor::ref<Falcor::Scene> pScene);

    void createSponzaSceneLights(Falcor::ref<Falcor::Scene> pScene);
    void spawnPointLightsAlongSegment(
        const Falcor::float3& startPt,
        const Falcor::float3& endPt,
        FloatRandomNumberGenerator& rng,
        float lightIntensity,
        uint32_t nbLightsAlongSegment
    );

    std::vector<Light> mLights;
    Falcor::ref<Falcor::Buffer> mGpuLightBuffer;

    std::vector<float> mLightProbabilities;
    Falcor::ref<Falcor::Buffer> mGpuLightProbabilityBuffer;
};

using LightManagerSingleton = Singleton<LightManager>;
} // namespace Restir

