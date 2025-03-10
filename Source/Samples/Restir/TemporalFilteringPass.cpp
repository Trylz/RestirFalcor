#include "TemporalFilteringPass.h"
#include "GBuffer.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{
using namespace Falcor;

TemporalFilteringPass::TemporalFilteringPass(
    ref<Device> pDevice,
    Falcor::ref<Falcor::Scene> pScene,
    uint32_t width,
    uint32_t height
)
    : mpScene(pScene), mWidth(width), mHeight(height)
{
    mpTemporalFilteringPass = ComputePass::create(pDevice, "Samples/Restir/TemporalFilteringPass.slang", "TemporalFilteringPass");
}

void TemporalFilteringPass::render(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "TemporalFilteringPass::render");

    auto var = mpTemporalFilteringPass->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["cameraPositionWs"] = mpScene->getCamera()->getPosition();
    var["PerFrameCB"]["previousFrameViewProjMat"] = transpose(mPreviousFrameViewProjMat);
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["sampleIndex"] = ++mSampleIndex;
    var["PerFrameCB"]["motion"] = (uint)(mPreviousFrameViewProjMat != mpScene->getCamera()->getViewProjMatrix());

    var["PerFrameCB"]["temporalLinearDepthThreshold"] = SceneSettingsSingleton::instance()->temporalLinearDepthThreshold;
    var["PerFrameCB"]["temporalWsRadiusThreshold"] = SceneSettingsSingleton::instance()->temporalWsRadiusThreshold;
    var["PerFrameCB"]["temporalNormalThreshold"] = SceneSettingsSingleton::instance()->temporalNormalThreshold;

    var["gCurrentFrameReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();
    var["gPreviousFrameReservoirs"] = ReservoirManagerSingleton::instance()->getPreviousFrameReservoirBuffer();

    var["gCurrentPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();
    var["gPreviousPositionWs"] = GBufferSingleton::instance()->getPreviousPositionWsTexture();

    var["gCurrentNormalWs"] = GBufferSingleton::instance()->getCurrentNormalWsTexture();
    var["gPreviousNormalWs"] = GBufferSingleton::instance()->getPreviousNormalWsTexture();
    var["gAlbedo"] = GBufferSingleton::instance()->getAlbedoTexture();
    var["gSpecular"] = GBufferSingleton::instance()->getSpecularTexture();

    mpTemporalFilteringPass->execute(pRenderContext, mWidth, mHeight);
    mPreviousFrameViewProjMat = mpScene->getCamera()->getViewProjMatrix();
}
} // namespace Restir
