#pragma once

#include "Singleton.h"

namespace Restir
{
struct RestirSample
{
    Falcor::float3 mGeometryPos;
    Falcor::float3 mLightSamplePosition;
    Falcor::float3 mIncomingRadiance;
};

struct RestirReservoir
{
    RestirSample mY;
    float mWsum = 0.0f;
    uint32_t mM = 0;
    float mW = 0.0f;
    float mHitDistance = 1e8f;
};

struct ReservoirManager
{
    ReservoirManager();

    void init(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height);

    inline const Falcor::ref<Falcor::Buffer>& getCurrentFrameReservoirBuffer() const { return mCurrentFrameReservoir; }
    inline const Falcor::ref<Falcor::Buffer>& getPreviousFrameReservoirBuffer() const { return mPreviousFrameReservoir; }

    inline void setNextFrame() { std::swap(mCurrentFrameReservoir, mPreviousFrameReservoir); }

private:
    Falcor::ref<Falcor::Buffer> mCurrentFrameReservoir;
    Falcor::ref<Falcor::Buffer> mPreviousFrameReservoir;
};

using ReservoirManagerSingleton = Singleton<ReservoirManager>;
} // namespace Restir

