#pragma once

#include "Singleton.h"

namespace Restir
{
struct RestirSample
{
    Falcor::float3 geometryPos;
    Falcor::float3 lightSamplePosition;
    Falcor::float3 incomingRadiance;
};

struct RestirReservoir
{
    RestirSample m_y;
    float m_Wsum = 0.0f;
    uint32_t m_M = 0;
    float m_W = 0.0f;
};

struct ReservoirManager
{
    ReservoirManager();

    void init(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height);

    inline const Falcor::ref<Falcor::Buffer>& getCurrentFrameReservoirBuffer() const { return *mCurrentFrameReservoir; }
    inline const Falcor::ref<Falcor::Buffer>& getPreviousFrameReservoirBuffer() const { return *mPreviousFrameReservoir; }

    inline void setNextFrame() { std::swap(mCurrentFrameReservoir, mPreviousFrameReservoir);}

    static const uint32_t nbReservoirPerPixel = 4;

private:
    Falcor::ref<Falcor::Buffer> mReservoirBuffer1;
    Falcor::ref<Falcor::Buffer> mReservoirBuffer2;

    Falcor::ref<Falcor::Buffer>* mCurrentFrameReservoir;
    Falcor::ref<Falcor::Buffer>* mPreviousFrameReservoir;
};

using ReservoirManagerSingleton = Singleton<ReservoirManager>;
} // namespace Restir
