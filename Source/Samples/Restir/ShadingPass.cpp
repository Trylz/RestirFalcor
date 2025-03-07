#include "ShadingPass.h"
#include "GBuffer.h"
#include "LightManager.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{
using namespace Falcor;

ShadingPass::ShadingPass(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height) : mWidth(width), mHeight(height)
{
    mpShadingPass = ComputePass::create(pDevice, "Samples/Restir/ShadingPass.slang", "ShadingPass");

    mpOuputTexture = pDevice->createTexture2D(
        width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );

    mpOuputTexture->setName("ShadingPass ouput texture");
}

void ShadingPass::render(Falcor::RenderContext* pRenderContext, ref<Camera> pCamera)
{
    FALCOR_PROFILE(pRenderContext, "ShadingPass::render");

    auto var = mpShadingPass->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["cameraPositionWs"] = pCamera->getPosition();
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["shadingLightExponent"] = SceneSettingsSingleton::instance()->shadingLightExponent;

    var["gOutput"] = mpOuputTexture;
    var["gReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();

    var["gPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();
    var["gNormalWs"] = GBufferSingleton::instance()->getCurrentNormalWsTexture();
    var["gAlbedo"] = GBufferSingleton::instance()->getAlbedoTexture();
    var["gSpecular"] = GBufferSingleton::instance()->getSpecularTexture();

    mpShadingPass->execute(pRenderContext, mWidth, mHeight);
}
} // namespace Restir
