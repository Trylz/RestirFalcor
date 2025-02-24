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

GBuffer::GBuffer(ref<Device> pDevice, ref<Scene> pScene, uint32_t width, uint32_t height) : RenderPass(pDevice), mpScene(pScene)
{
    createTextures(pDevice, width, height);
    initializeGraphicStates(pDevice);
}

void GBuffer::createTextures(ref<Device> pDevice, uint32_t width, uint32_t height)
{
    // Set witdh and height
    mWidth = width;
    mHeight = height;

    // Create gbuffer textures.
    mPositionWsTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mNormalWsTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mTangentWsTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    FaceNormalWsTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mTextureCoordTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RG32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );
    mTextureGradientsTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA16Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mMotionVectorTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RG32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mMaterialDataTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA32Uint,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mDepthTexture = pDevice->createTexture2D(
        width,
        height,
        ResourceFormat::D32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );
}

void GBuffer::initializeGraphicStates(ref<Device> pDevice)
{
    if (!pDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("GBuffer requires Shader Model 6.2 support.");
    if (!pDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("GBuffer requires pixel shader barycentrics support.");
    if (!pDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
        FALCOR_THROW("GBuffer requires rasterizer ordered views (ROVs) support.");

    mDepthPass.pState = GraphicsState::create(pDevice);
    mGBufferPass.pState = GraphicsState::create(pDevice);

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::Equal).setDepthWriteMask(false);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mGBufferPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(pDevice);
}

RenderPassReflection GBuffer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Add the required depth output. This always exists.
    reflector.addOutput(kDepthName, "Depth buffer")
        .format(ResourceFormat::D32Float)
        .bindFlags(ResourceBindFlags::DepthStencil)
        .texture2D(mWidth, mHeight);

    // Add all the other outputs.
    // The default channels are written as render targets, the rest as UAVs as there is way to assign/pack render targets yet.
    addRenderPassOutputs(reflector, kGBufferChannels, ResourceBindFlags::RenderTarget, sz);
   // addRenderPassOutputs(reflector, kGBufferExtraChannels, ResourceBindFlags::UnorderedAccess, sz);
    //reflector.getField(kVBufferName)->format(mVBufferFormat);

    return reflector;
}

void GBuffer::compilePrograms()
{
    // Depth pass..
    {
        if (!mDepthPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kDepthPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mDepthPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mDepthPass.pState->setProgram(mDepthPass.pProgram);
        }

        // Set program defines.
        mDepthPass.pState->getProgram()->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        // Create program vars.
        if (!mDepthPass.pVars)
            mDepthPass.pVars = ProgramVars::create(mpDevice, mDepthPass.pProgram.get());

        mpFbo->attachDepthStencilTarget(pDepth);
        mDepthPass.pState->setFbo(mpFbo);

        mpScene->rasterize(pRenderContext, mDepthPass.pState.get(), mDepthPass.pVars.get(), cullMode);
    }

    // GBuffer pass.
    {
        // Create GBuffer pass program.
        if (!mGBufferPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kGBufferPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mGBufferPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mGBufferPass.pState->setProgram(mGBufferPass.pProgram);
        }

        // Set program defines.
        mGBufferPass.pProgram->addDefine("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
        mGBufferPass.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        // TODO: This should be moved to a more general mechanism using Slang.
        mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferChannels, renderData));
        mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferExtraChannels, renderData));

        // Create program vars.
        if (!mGBufferPass.pVars)
            mGBufferPass.pVars = ProgramVars::create(mpDevice, mGBufferPass.pProgram.get());

        auto var = mGBufferPass.pVars->getRootVar();

        // Bind extra channels as UAV buffers.
        for (const auto& channel : kGBufferExtraChannels)
        {
            ref<Texture> pTex = getOutput(renderData, channel.name);
            var[channel.texname] = pTex;
        }

        var["PerFrameCB"]["gFrameDim"] = mFrameDim;
        mGBufferPass.pState->setFbo(mpFbo); // Sets the viewport

        // Rasterize the scene.
        mpScene->rasterize(pRenderContext, mGBufferPass.pState.get(), mGBufferPass.pVars.get(), cullMode);
    }
}

void GBuffer::render(RenderContext* pRenderContext)
{

}

} // namespace Restir
