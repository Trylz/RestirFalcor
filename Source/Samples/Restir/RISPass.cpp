#include "RISPass.h"
#include "GBuffer.h"
#include "LightManager.h"
#include "ReservoirManager.h"

namespace Restir
{
    using namespace Falcor;

    RISPass::RISPass(ref<Device> pDevice, uint32_t width, uint32_t height) : mWidth(width), mHeight(height)
    {
        mpRISPass = ComputePass::create(pDevice, "Samples/Restir/RISPass.slang", "EntryPoint");
    }

    void RISPass::render(Falcor::RenderContext* pRenderContext, ref<Camera> pCamera)
    {
        auto var = mpRISPass->getRootVar();

        var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
        var["PerFrameCB"]["cameraPositionWs"] = pCamera->getPosition();
        var["PerFrameCB"]["sampleIndex"] = ++mSampleIndex;
        var["PerFrameCB"]["nbReservoirPerPixel"] = ReservoirManager::nbReservoirPerPixel;
        var["PerFrameCB"]["lightCount"] = (uint32_t)LightManagerSingleton::instance()->getLights().size();

        var["gReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();
        var["gLights"] = LightManagerSingleton::instance()->getLightGpuBuffer();
        var["gLightProbabilities"] = LightManagerSingleton::instance()->getLightProbabilitiesGpuBuffer();

        var["gPositionWs"] = GBufferSingleton::instance()->getPositionWsTexture();
        var["gNormalWs"] = GBufferSingleton::instance()->getPositionWsTexture();
        var["gAlbedo"] = GBufferSingleton::instance()->getPositionWsTexture();
        var["gSpecular"] = GBufferSingleton::instance()->getPositionWsTexture();

        mpRISPass->execute(pRenderContext, mWidth, mHeight);
    }
} // namespace Restir
