#pragma once

#include "Singleton.h"

namespace Restir
{
struct Light
{
    Falcor::float3 mWsPosition;
    float mRadius;
    float mRadiance;
    float mfallOff;
};

struct LightManager
{
    LightManager();
    inline const std::vector<Light>& getLights() const { return mLights; }

private:
    std::vector<Light> mLights;
};

using EntityDatabaseSingleton = Singleton<LightManager>;
} // namespace Restir
