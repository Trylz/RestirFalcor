#include "GBuffer.h"

namespace Restir
{
using namespace Falcor;

GBuffer::GBuffer() {}

void GBuffer::init(ref<Device> pDevice, ref<Scene> pScene, uint32_t width, uint32_t height)
{
    mpDevice = pDevice;
    mpScene = pScene;
    mWidth = width;
    mHeight = height;

    createTextures();
    compilePrograms();
}

void GBuffer::createTextures()
{
    mCurrentPositionWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mPreviousPositionWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mCurrentNormalWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mPreviousNormalWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mAlbedoTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mSpecularTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
}

void GBuffer::compilePrograms()
{
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    auto defines = mpScene->getSceneDefines();

    ProgramDesc rtProgDesc;
    rtProgDesc.addShaderModules(shaderModules);
    rtProgDesc.addShaderLibrary("Samples/Restir/GBuffer.slang");
    rtProgDesc.addTypeConformances(typeConformances);
    rtProgDesc.setMaxTraceRecursionDepth(1);

    rtProgDesc.setMaxPayloadSize(24);

    ref<RtBindingTable> sbt = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
    sbt->setRayGen(rtProgDesc.addRayGen("rayGen"));
    sbt->setMiss(0, rtProgDesc.addMiss("primaryMiss"));
    auto primary = rtProgDesc.addHitGroup("primaryClosestHit", "primaryAnyHit");

    sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), primary);

    mpRaytraceProgram = Program::create(mpDevice, rtProgDesc, defines);
    mpRtVars = RtProgramVars::create(mpDevice, mpRaytraceProgram, sbt);
}

void GBuffer::render(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "GBuffer::render");

    auto var = mpRtVars->getRootVar();

    var["PerFrameCB"]["viewportDims"] = float2(mWidth, mHeight);
    var["PerFrameCB"]["sampleIndex"] = mSampleIndex++;

    var["gPositionWs"] = mCurrentPositionWsTexture;
    var["gNormalWs"] = mCurrentNormalWsTexture;
    var["gAlbedo"] = mAlbedoTexture;
    var["gSpecular"] = mSpecularTexture;

    mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, uint3(mWidth, mHeight, 1));
}

} // namespace Restir

