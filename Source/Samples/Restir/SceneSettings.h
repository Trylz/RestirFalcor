#pragma once

#include "Singleton.h"

namespace Restir
{
struct SceneSettings
{
    uint32_t RISSamplesCount = 32u;
    uint32_t nbReservoirPerPixel = 4u;

    // Temporal settings
    // float temporalLinearDepthThreshold = 0.1f;
    float temporalWsRadiusThreshold = 999999999.0f;
    float temporalNormalThreshold = 0.5f;

    // Spatial settings
    float spatialWsRadiusThreshold = 999999999.0f;
    float spatialNormalThreshold = 0.12f;

    // Lightin settings
    float shadingLightExponent = 1.0f;
};

using SceneSettingsSingleton = Singleton<SceneSettings>;
} // namespace Restir
