#include "../RenderPasses/GBuffer/GBuffer/GBufferRaster.h"

namespace Restir
{
class GBuffer /*: public RenderPass*/
{
public:
    GBuffer(ref<Device> pDevice, uint32_t width, uint32_t height);
    void render(RenderContext* pRenderContext, ref<Scene> pScene);

    inline const ref<Texture>& getPositionWsTexture() const { return mPositionWsTexture; }
    inline const ref<Texture>& getNormalWsTexture() const { return mNormalWsTexture; }
    inline const ref<Texture>& getAlbedoTexture() const { return mAlbedoTexture; }
    inline const ref<Texture>& getSpecularTexture() const { return mSpecularTexture; }
    inline const ref<Texture>& getMotionVectorTexture() const { return mMotionVectorTexture; }

private:
    void createTextures();
    void compilePrograms();

    ref<Device> mpDevice;
    Scene* mpScene = nullptr; // TODO use smart pointer

    uint32_t mWidth;
    uint32_t mHeight;

    ref<Fbo> mpFbo;

    ref<Texture> mPositionWsTexture;
    ref<Texture> mNormalWsTexture;
    ref<Texture> mAlbedoTexture;
    ref<Texture> mSpecularTexture;
    ref<Texture> mMotionVectorTexture;
    ref<Texture> mDepthTexture;

    ref<RasterPass> mpRasterPass;
};
} // namespace Restir
