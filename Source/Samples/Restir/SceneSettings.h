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

    // Lightin settings
    float shadingLightExponent = 1.0f;
};

using SceneSettingsSingleton = Singleton<SceneSettings>;
} // namespace Restir
