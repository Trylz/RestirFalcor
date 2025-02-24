#include "GBuffer.h"

namespace Restir
{
const std::string kDepthName = "depth";

const ChannelList kGBufferChannels = {
    // clang-format off
    { "posW",           "gPosW",            "Position in world space",                           true /* optional */, ResourceFormat::RGBA32Float },
    { "normW",          "gNormW",           "Shading normal in world space",                     true /* optional */, ResourceFormat::RGBA32Float },
    { "tangentW",       "gTangentW",        "Shading tangent in world space (xyz) and sign (w)", true /* optional */, ResourceFormat::RGBA32Float },
    { "faceNormalW",    "gFaceNormalW",     "Face normal in world space",                        true /* optional */, ResourceFormat::RGBA32Float },
    { "texC",           "gTexC",            "Texture coordinate",                                true /* optional */, ResourceFormat::RG32Float   },
    { "texGrads",       "gTexGrads",        "Texture gradients (ddx, ddy)",                      true /* optional */, ResourceFormat::RGBA16Float },
    { "mvec",           "gMotionVector",    "Motion vector",                                     true /* optional */, ResourceFormat::RG32Float   },
    { "mtlData",        "gMaterialData",    "Material data (ID, header.x, header.y, lobes)",     true /* optional */, ResourceFormat::RGBA32Uint  },
    // clang-format on
};

GBuffer::GBuffer(ref<Device> pDevice, uint32_t width, uint32_t height) : mpDevice(pDevice), mWidth(width), mHeight(height)
{
    createTextures();
}

void GBuffer::createTextures()
{
    // Create gbuffer textures.
    mPositionWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mNormalWsTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mAlbedoTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mSpecularTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mMotionVectorTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
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

    ProgramDesc rtProgDesc;
    rtProgDesc.addShaderModules(shaderModules);
    rtProgDesc.addShaderLibrary("Samples/Restir/GBuffer.slang");
    rtProgDesc.addTypeConformances(typeConformances);
    rtProgDesc.setMaxTraceRecursionDepth(1); // 1 for calling TraceRay from RayGen, 1 for calling it from the
    // primary-ray ClosestHit shader for reflections, 1 for reflection ray
    // tracing a shadow ray
    rtProgDesc.setMaxPayloadSize(24); // The largest ray payload struct (PrimaryRayData) is 24 bytes. The payload size
    // should be set as small as possible for maximum performance.

    ref<RtBindingTable> sbt = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
    sbt->setRayGen(rtProgDesc.addRayGen("rayGen"));
    sbt->setMiss(0, rtProgDesc.addMiss("primaryMiss"));
    sbt->setMiss(1, rtProgDesc.addMiss("shadowMiss"));
    auto primary = rtProgDesc.addHitGroup("primaryClosestHit", "primaryAnyHit");
    auto shadow = rtProgDesc.addHitGroup("", "shadowAnyHit");
    sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), primary);
    sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), shadow);

    mpRaytraceProgram = Program::create(mpDevice, rtProgDesc, defines);
    mpRtVars = RtProgramVars::create(mpDevice, mpRaytraceProgram, sbt);
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
    var["gPositionWs"] = mPositionWsTexture;
    var["gNormalWs"] = mNormalWsTexture;
    var["gAlbedo"] = mAlbedoTexture;
    var["gSpecular"] = mSpecularTexture;
    var["gMotionVector"] = mMotionVectorTexture;

    mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, uint3(mWidth, mHeight, 1));
}

} // namespace Restir
