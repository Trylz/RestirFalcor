#pragma once

#include "Singleton.h"

namespace Restir
{
struct SceneSettings
{
    uint32_t RISSamplesCount = 32u;
    uint32_t nbReservoirPerPixel = 4u;

    // Temporal settings
    float temporalWsRadiusThreshold = 999999999.0f;
    float temporalLinearDepthThreshold = 0.4f;
    float temporalNormalThreshold = 0.1f;

    // Spatial settings
    float spatialWsRadiusThreshold = 999999999.0f;
    float spatialNormalThreshold = 0.12f;

    // Lighting settings
    float sceneShadingLightExponent = 1.0f;
    Falcor::float3 sceneAmbientColor = Falcor::float3(0.0f, 0.0f, 0.0f);
};

using SceneSettingsSingleton = Singleton<SceneSettings>;
} // namespace Restir
