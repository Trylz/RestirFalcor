#include "GBuffer.h"

namespace Restir
{
GBuffer::GBuffer(ref<Device> pDevice, uint32_t width, uint32_t height)
{
    // Create native GBuffer pass.
    Properties props;
    mpNativeGBufferPass = GBufferRaster::create(pDevice, props);

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

    // Create Gbuffer pass render data
    const std::string& passName = "Gbuffer";

    ResourceCache resources;
    resources.registerExternalResource("posW", mPositionWsTexture);
    resources.registerExternalResource("normW", mPositionWsTexture);
    resources.registerExternalResource("tangentW", mPositionWsTexture);
    resources.registerExternalResource("faceNormalW", mPositionWsTexture);
    resources.registerExternalResource("texC", mPositionWsTexture);
    resources.registerExternalResource("texGrads", mPositionWsTexture);
    resources.registerExternalResource("mvec", mPositionWsTexture);
    resources.registerExternalResource("mtlData", mPositionWsTexture);

    Dictionary dic; // Why this??

    const uint2 defaultTexDims(width, height);
    ResourceFormat defaultTexFormat = ResourceFormat::RGBA32Float; // Why this??

    mpRenderData.reset(new Falcor::RenderData("GbufferRenderData", resources, dic, defaultTexDims, defaultTexFormat));
}

void GBuffer::render(RenderContext* pRenderContext)
{
    mpNativeGBufferPass->execute(pRenderContext, *mpRenderData);
}

} // namespace Restir
