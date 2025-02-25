#pragma once

#include <random>

namespace Restir
{

struct FloatRandomNumberGenerator
{
    // Generate between [0...1]
    inline float generateUnsignedNormalized() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }

    // Generate between [-1...1]
    inline float generateSignedNormalized()
    {
        const float random = ((float)rand()) / (float)RAND_MAX;
        return (random * 2.0f) - 1.0f;
    }

    // Generate between [a...b]
    inline float generateBeetween(float a, float b)
    {
        const float random = ((float)rand()) / (float)RAND_MAX;
        const float range = b - a;
        return (random * range) + a;
    }
};

} // namespace Restir
