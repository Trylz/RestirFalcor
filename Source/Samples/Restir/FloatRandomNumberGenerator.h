#pragma once

#include <random>

namespace Restir
{

struct FloatRandomNumberGenerator
{
    inline FloatRandomNumberGenerator(int seed = -1)
    {
        if (seed < 0)
        {
            m_state = (uint32_t)time(0);
        }
        else
        {
            m_state = (uint32_t)seed;
        }
    }

    // Generate between [0...1]
    inline float generateUnsignedNormalized() { return static_cast<float>(xorshift32()) / static_cast<float>(s_uint32Max); }

    // Generate between [-1...1]
    inline float generateSignedNormalized() { return generateBeetween(-1.0f, 1.0f); }

    // Generate between [a...b]
    inline float generateBeetween(float a, float b) { return a + static_cast<float>(xorshift32()) / (static_cast<float>(s_uint32Max / (b - a))); }

    private:

    inline uint32_t xorshift32()
    {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        uint32_t x = m_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return m_state = x;
    }

    static const uint32_t s_uint32Max = std::numeric_limits<uint32_t>::max();
    uint32_t m_state;
  };
} // namespace Restir

