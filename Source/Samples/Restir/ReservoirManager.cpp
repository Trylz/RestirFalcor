#include "ReservoirManager.h"
namespace Restir
{

ReservoirManager::ReservoirManager()
{
}


void ReservoirManager::init(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height)
{
    //------------------------------------------------------------------------------------------------------------
    //	Init reservoirs
    //------------------------------------------------------------------------------------------------------------
    const uint32_t nbPixels = width * height;
    const uint32_t nbReservoirs = nbPixels * nbReservoirPerPixel;
    std::vector<RestirReservoir> reservoirs(nbReservoirs);

    //------------------------------------------------------------------------------------------------------------
    //	Create GPU reservoirs
    //------------------------------------------------------------------------------------------------------------

    mReservoirBuffer1 = pDevice->createStructuredBuffer(
        sizeof(RestirReservoir),
        reservoirs.size(),
        Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess,
        Falcor::MemoryType::DeviceLocal,
        reservoirs.data(),
        false
    );

    mReservoirBuffer2 = pDevice->createStructuredBuffer(
        sizeof(RestirReservoir),
        reservoirs.size(),
        Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess,
        Falcor::MemoryType::DeviceLocal,
        reservoirs.data(),
        false
    );

    //------------------------------------------------------------------------------------------------------------
    //	Init current and previous frame reservoirs
    //------------------------------------------------------------------------------------------------------------
    mCurrentFrameReservoir = &mReservoirBuffer1;
    mPreviousFrameReservoir = &mReservoirBuffer2;
}

} // namespace Restir
