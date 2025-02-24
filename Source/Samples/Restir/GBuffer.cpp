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

GBuffer::GBuffer(ref<Device> pDevice, ref<Scene> pScene, uint32_t width, uint32_t height)
   : mpDevice(pDevice)
    ,mpScene(pScene)
    ,mWidth(width)
    ,mHeight(height)
{
    createTextures();
    initializeGraphicStates();
}

void GBuffer::createTextures()
{
    // Create gbuffer textures.
    mPositionWsTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mNormalWsTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mTangentWsTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mFaceNormalWsTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mTextureCoordTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RG32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );
    mTextureGradientsTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA16Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mMotionVectorTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RG32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mMaterialDataTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA32Uint,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );

    mDepthTexture = mpDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::D32Float,
        1,
        1,
        nullptr,
        /* ResourceBindFlags::UnorderedAccess | */ ResourceBindFlags::ShaderResource
    );
}

void GBuffer::initializeGraphicStates()
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("GBuffer requires Shader Model 6.2 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("GBuffer requires pixel shader barycentrics support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
        FALCOR_THROW("GBuffer requires rasterizer ordered views (ROVs) support.");

    mDepthPass.pState = GraphicsState::create(mpDevice);
    mGBufferPass.pState = GraphicsState::create(mpDevice);

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::Equal).setDepthWriteMask(false);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mGBufferPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(mpDevice);
}

/*
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
*/

void GBuffer::compilePrograms()
{
    // Some of the stuff here are recreated every frame when i check GBufferRaster. Why?
    // Depth pass..
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary("Samples/Restir/DepthPass.slang").vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mDepthPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mDepthPass.pState->setProgram(mDepthPass.pProgram);
        mDepthPass.pVars = ProgramVars::create(mpDevice, mDepthPass.pProgram.get());

        mpFbo->attachDepthStencilTarget(mDepthTexture);
        mDepthPass.pState->setFbo(mpFbo);
    }

    // GBuffer pass.
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary("Samples/Restir/GBuffer.slang").vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());

        mGBufferPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mGBufferPass.pState->setProgram(mGBufferPass.pProgram);

        // Set program defines.
        mGBufferPass.pProgram->addDefine("ADJUST_SHADING_NORMALS", "0");
        mGBufferPass.pProgram->addDefine("USE_ALPHA_TEST", "0");

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        // TODO: This should be moved to a more general mechanism using Slang.
        // mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferChannels, renderData));
        //mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferExtraChannels, renderData));

        // Create program vars.
        mGBufferPass.pVars = ProgramVars::create(mpDevice, mGBufferPass.pProgram.get());
        mGBufferPass.pState->setFbo(mpFbo); // Sets the viewport

    }
}

void GBuffer::render(RenderContext* pRenderContext)
{
    const RasterizerState::CullMode cullMode = RasterizerState::CullMode::None;

    // Depth pass.
    {
        FALCOR_PROFILE(pRenderContext, "GBuffer::render Depth");
        pRenderContext->clearDsv(mDepthTexture->getDSV().get(), 1.f, 0);
        mpScene->rasterize(pRenderContext, mDepthPass.pState.get(), mDepthPass.pVars.get(), cullMode);
    }

    // GBuffer pass.
    {
        FALCOR_PROFILE(pRenderContext, "GBuffer::render Gbuffer");
        mpFbo->attachColorTarget(mPositionWsTexture, 0u);
        mpFbo->attachColorTarget(mNormalWsTexture, 1u);
        mpFbo->attachColorTarget(mTangentWsTexture, 2u);
        mpFbo->attachColorTarget(mFaceNormalWsTexture, 3u);
        mpFbo->attachColorTarget(mTextureCoordTexture, 4u);
        mpFbo->attachColorTarget(mTextureGradientsTexture, 5u);
        mpFbo->attachColorTarget(mMotionVectorTexture, 6u);
        mpFbo->attachColorTarget(mMaterialDataTexture, 7u);

        pRenderContext->clearFbo(mpFbo.get(), float4(0), 1.f, 0, FboAttachmentType::Color);

        auto var = mGBufferPass.pVars->getRootVar();
        var["PerFrameCB"]["gFrameDim"] = uint2(mWidth, mHeight);
        mGBufferPass.pState->setFbo(mpFbo);
        mpScene->rasterize(pRenderContext, mGBufferPass.pState.get(), mGBufferPass.pVars.get(), cullMode);
    }
}

} // namespace Restir
