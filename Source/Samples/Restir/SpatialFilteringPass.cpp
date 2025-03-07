#include "SpatialFilteringPass.h"
#include "GBuffer.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{
using namespace Falcor;

SpatialFilteringPass::SpatialFilteringPass(ref<Device> pDevice, Falcor::ref<Falcor::Scene> pScene, uint32_t width, uint32_t height)
    : mpScene(pScene), mWidth(width), mHeight(height)
{
    mpSpatialFilteringPass = ComputePass::create(pDevice, "Samples/Restir/SpatialFilteringPass.slang", "SpatialFiltering");

    const uint32_t nbPixels = width * height;
    const uint32_t nbReservoirs = nbPixels * SceneSettingsSingleton::instance()->nbReservoirPerPixel;

    mpStagingReservoirs = pDevice->createStructuredBuffer(
        sizeof(RestirReservoir),
        nbReservoirs,
        Falcor::ResourceBindFlags::ShaderResource | Falcor::ResourceBindFlags::UnorderedAccess,
        Falcor::MemoryType::DeviceLocal
    );
}

void SpatialFilteringPass::render(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SpatialFilteringPass::render");

    performSpatialFiltering(pRenderContext);
    performReservoirCopy(pRenderContext);
}

void SpatialFilteringPass::performSpatialFiltering(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SpatialFilteringPass::performSpatialFiltering");

    auto var = mpSpatialFilteringPass->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["cameraPositionWs"] = mpScene->getCamera()->getPosition();
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["sampleIndex"] = ++mSampleIndex;

    var["PerFrameCB"]["spatialWsRadiusThreshold"] = SceneSettingsSingleton::instance()->spatialWsRadiusThreshold;
    var["PerFrameCB"]["spatialNormalThreshold"] = SceneSettingsSingleton::instance()->spatialNormalThreshold;

    var["gCurrentFrameReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();
    var["gStagingReservoirs"] = mpStagingReservoirs;

    var["gPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();
    var["gNormalWs"] = GBufferSingleton::instance()->getCurrentNormalWsTexture();
    var["gAlbedo"] = GBufferSingleton::instance()->getAlbedoTexture();
    var["gSpecular"] = GBufferSingleton::instance()->getSpecularTexture();

    mpSpatialFilteringPass->execute(pRenderContext, mWidth, mHeight);
}

void SpatialFilteringPass::performReservoirCopy(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SpatialFilteringPass::performReservoirCopy");
    pRenderContext->copyBufferRegion(
        ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer().get(),
        0u,
        mpStagingReservoirs.get(),
        0u,
        mpStagingReservoirs->getSize()
    );
}

} // namespace Restir

