#include "GBuffer.h"
#include "Core/Pass/RasterPass.h"

namespace Restir
{

GBuffer::GBuffer(ref<Device> pDevice, uint32_t width, uint32_t height) : mpDevice(pDevice), mWidth(width), mHeight(height)
{
    createTextures();
}

void GBuffer::createTextures()
{
    // Create gbuffer textures.
    mPositionWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );

    mNormalWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );

    mAlbedoTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );

    mSpecularTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );

    mMotionVectorTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );

    mDepthTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::D32Float, 1, 1, nullptr, ResourceBindFlags::DepthStencil
    );  
}

void GBuffer::compilePrograms()
{
    // We'll now create a raytracing program. To do that we need to setup two things:
    // - A program description (ProgramDesc). This holds all shader entry points, compiler flags, macro defintions,
    // etc.
    // - A binding table (RtBindingTable). This maps shaders to geometries in the scene, and sets the ray generation and
    // miss shaders.
    //
    // After setting up these, we can create the Program and associated RtProgramVars that holds the variable/resource
    // bindings. The Program can be reused for different scenes, but RtProgramVars needs to binding table which is
    // Scene-specific and needs to be re-created when switching scene. In this example, we re-create both the program
    // and vars when a scene is loaded.

    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    // Get scene defines. These need to be set on any program using the scene.
    auto defines = mpScene->getSceneDefines();

    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(shaderModules);
    rasterProgDesc.addShaderLibrary("Samples/Restir/GBuffer.slang").vsEntry("vsMain").psEntry("psMain");
    rasterProgDesc.addTypeConformances(typeConformances);

    mpFbo = Fbo::create(mpDevice);
    mpRasterPass = RasterPass::create(mpDevice, rasterProgDesc, defines);
}

void GBuffer::render(RenderContext* pRenderContext, ref<Scene> pScene)
{
    if (pScene != mpScene)
    {
        mpScene = pScene.get();
        compilePrograms();
    }

    auto var = mpRtVars->getRootVar();

    var["PerFrameCB"]["viewportDims"] = float2(mWidth, mHeight);
    var["PerFrameCB"]["sampleIndex"] = mSampleIndex++;

    var["gPositionWs"] = mPositionWsTexture;
    var["gNormalWs"] = mNormalWsTexture;
    var["gAlbedo"] = mAlbedoTexture;
    var["gSpecular"] = mSpecularTexture;
    var["gMotionVector"] = mMotionVectorTexture;

    mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, uint3(mWidth, mHeight, 1));
}

} // namespace Restir
