#include "GBuffer.h"

namespace Restir
{
GBuffer::GBuffer(ref<Device> pDevice, uint32_t width, uint32_t height)
{
    createTextures(pDevice, width, height);
    initializeGraphicStates(pDevice);
}

void GBuffer::createTextures(ref<Device> pDevice, uint32_t width, uint32_t height)
{
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
    mDepthPass.pState = GraphicsState::create(pDevice);
    mGBufferPass.pState = GraphicsState::create(pDevice);

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::Equal).setDepthWriteMask(false);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mGBufferPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(pDevice);
}

void GBuffer::render(RenderContext* pRenderContext)
{

}

} // namespace Restir
