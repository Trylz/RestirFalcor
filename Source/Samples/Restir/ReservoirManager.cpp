#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{

ReservoirManager::ReservoirManager() {}

void ReservoirManager::init(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height)
{
    //------------------------------------------------------------------------------------------------------------
    //	Init reservoirs
    //------------------------------------------------------------------------------------------------------------
    const uint32_t nbPixels = width * height;
    const uint32_t nbReservoirs = nbPixels * SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    std::vector<RestirReservoir> reservoirs(nbReservoirs);

    //------------------------------------------------------------------------------------------------------------
    //	Create GPU reservoirs
    //------------------------------------------------------------------------------------------------------------

    mCurrentFrameReservoir = pDevice->createStructuredBuffer(
        sizeof(RestirReservoir),
        reservoirs.size(),
        Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess,
        Falcor::MemoryType::DeviceLocal,
        reservoirs.data(),
        false
    );

    mPreviousFrameReservoir = pDevice->createStructuredBuffer(
        sizeof(RestirReservoir),
        reservoirs.size(),
        Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess,
        Falcor::MemoryType::DeviceLocal,
        reservoirs.data(),
        false
    );
}

} // namespace Restir

