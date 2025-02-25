#pragma once

#include "Singleton.h"

namespace Restir
{
struct Light
{
    Falcor::float3 mWsPosition;
    Falcor::float3 mRadiance;
    float mRadius;
    float mfallOff;
};

struct LightManager
{
    LightManager(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene);
    inline const std::vector<Light>& getLights() const { return mLights; }
    inline const Falcor::ref<Falcor::Buffer>& getLightGpuBuffer() const { return mGpuLightBuffer; }

private:
    std::vector<Light> mLights;
    Falcor::ref<Falcor::Buffer> mGpuLightBuffer;

    std::vector<float> mLightProbabilities;
    Falcor::ref<Falcor::Buffer> mGpuLightProbabilityBuffer;
};

using EntityDatabaseSingleton = Singleton<LightManager>;
} // namespace Restir
