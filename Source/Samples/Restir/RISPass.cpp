#include "RISPass.h"
#include "GBuffer.h"
#include "LightManager.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{
using namespace Falcor;

RISPass::RISPass(ref<Device> pDevice, uint32_t width, uint32_t height) : mWidth(width), mHeight(height)
{
    mpRISPass = ComputePass::create(pDevice, "Samples/Restir/RISPass.slang", "EntryPoint");
}

void RISPass::render(Falcor::RenderContext* pRenderContext, ref<Camera> pCamera)
{
    FALCOR_PROFILE(pRenderContext, "RISPass::render");

    auto var = mpRISPass->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["cameraPositionWs"] = pCamera->getPosition();
    var["PerFrameCB"]["sampleIndex"] = ++mSampleIndex;
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["lightCount"] = (uint32_t)LightManagerSingleton::instance()->getLights().size();
    var["PerFrameCB"]["RISSamplesCount"] = SceneSettingsSingleton::instance()->RISSamplesCount;

    var["gReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();
    var["gLights"] = LightManagerSingleton::instance()->getLightGpuBuffer();
    var["gLightProbabilities"] = LightManagerSingleton::instance()->getLightProbabilitiesGpuBuffer();

    var["gPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();
    var["gNormalWs"] = GBufferSingleton::instance()->getCurrentNormalWsTexture();
    var["gAlbedo"] = GBufferSingleton::instance()->getAlbedoTexture();
    var["gSpecular"] = GBufferSingleton::instance()->getSpecularTexture();

    mpRISPass->execute(pRenderContext, mWidth, mHeight);
}
} // namespace Restir

