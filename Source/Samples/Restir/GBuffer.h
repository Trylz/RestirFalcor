#include "../RenderPasses/GBuffer/GBuffer/GBufferRaster.h"

namespace Restir
{
class GBuffer
{
public:
    GBuffer(ref<Device> pDevice, uint32_t width, uint32_t height);
    void render(RenderContext* pRenderContext);

    inline const ref<Texture>& getPositionWsTexture() const { return mPositionWsTexture; }
    inline const ref<Texture>& getNormalWsTexture() const { return mNormalWsTexture; }
    inline const ref<Texture>& getTangentWsTexture() const { return mTangentWsTexture; }
    inline const ref<Texture>& getFaceNormalWsTexture() const { return FaceNormalWsTexture; }
    inline const ref<Texture>& getTextureCoordTexture() const { return mTextureCoordTexture; }
    inline const ref<Texture>& getTextureGradientsTexture() const { return mTextureGradientsTexture; }
    inline const ref<Texture>& getMotionVectorTexture() const { return mMotionVectorTexture; }
    inline const ref<Texture>& getMaterialDataTexture() const { return mMaterialDataTexture; }

private:
    ref<GBufferRaster> mpNativeGBufferPass;
    ref<Falcor::RenderData> mpRenderData;

    ref<Texture> mPositionWsTexture;
    ref<Texture> mNormalWsTexture;
    ref<Texture> mTangentWsTexture;
    ref<Texture> FaceNormalWsTexture;
    ref<Texture> mTextureCoordTexture;
    ref<Texture> mTextureGradientsTexture;
    ref<Texture> mMotionVectorTexture;
    ref<Texture> mMaterialDataTexture;

    ref<Texture> mDepthTexture;
};
} // namespace Restir
