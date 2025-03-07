#include "VisibilityPass.h"

#include "ReservoirManager.h"
#include "SceneSettings.h"

namespace Restir
{
using namespace Falcor;

VisibilityPass::VisibilityPass(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene, uint32_t width, uint32_t height)
    : mpScene(pScene), mWidth(width), mHeight(height)
{
    compileProgram(pDevice);
}

void VisibilityPass::compileProgram(Falcor::ref<Falcor::Device> pDevice)
{
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    auto defines = mpScene->getSceneDefines();

    ProgramDesc rtProgDesc;
    rtProgDesc.addShaderModules(shaderModules);
    rtProgDesc.addShaderLibrary("Samples/Restir/VisibilityPass.slang");
    rtProgDesc.addTypeConformances(typeConformances);
    rtProgDesc.setMaxTraceRecursionDepth(1);

    rtProgDesc.setMaxPayloadSize(24);

    ref<RtBindingTable> sbt = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
    sbt->setRayGen(rtProgDesc.addRayGen("rayGen"));
    sbt->setMiss(0, rtProgDesc.addMiss("primaryMiss"));
    sbt->setMiss(1, rtProgDesc.addMiss("shadowMiss"));
    auto primary = rtProgDesc.addHitGroup("primaryClosestHit", "primaryAnyHit");
    auto shadow = rtProgDesc.addHitGroup("", "shadowAnyHit");
    sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), primary);
    sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), shadow);

    mpRaytraceProgram = Program::create(pDevice, rtProgDesc, defines);
    mpRtVars = RtProgramVars::create(pDevice, mpRaytraceProgram, sbt);
}

void VisibilityPass::render(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "VisibilityPass::render");

    auto var = mpRtVars->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;

    var["gReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();

    mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, uint3(mWidth, mHeight, 1));
}

} // namespace Restir

