#include "../RenderPasses/GBuffer/GBuffer/GBufferRaster.h"

namespace Restir
{
class GBuffer : public RenderPass
{
public:
    GBuffer(ref<Device> pDevice, ref<Scene> pScene, uint32_t width, uint32_t height);
    void render(RenderContext* pRenderContext);

    RenderPassReflection reflect(const CompileData& compileData) override;

    inline const ref<Texture>& getPositionWsTexture() const { return mPositionWsTexture; }
    inline const ref<Texture>& getNormalWsTexture() const { return mNormalWsTexture; }
    inline const ref<Texture>& getTangentWsTexture() const { return mTangentWsTexture; }
    inline const ref<Texture>& getFaceNormalWsTexture() const { return FaceNormalWsTexture; }
    inline const ref<Texture>& getTextureCoordTexture() const { return mTextureCoordTexture; }
    inline const ref<Texture>& getTextureGradientsTexture() const { return mTextureGradientsTexture; }
    inline const ref<Texture>& getMotionVectorTexture() const { return mMotionVectorTexture; }
    inline const ref<Texture>& getMaterialDataTexture() const { return mMaterialDataTexture; }

private:
    void createTextures(ref<Device> pDevice, uint32_t width, uint32_t height);
    void initializeGraphicStates(ref<Device> pDevice);
    void compilePrograms();

    uint32_t mWidth;
    uint32_t mHeight;

    ref<Scene> mpScene;

    ref<Texture> mPositionWsTexture;
    ref<Texture> mNormalWsTexture;
    ref<Texture> mTangentWsTexture;
    ref<Texture> FaceNormalWsTexture;
    ref<Texture> mTextureCoordTexture;
    ref<Texture> mTextureGradientsTexture;
    ref<Texture> mMotionVectorTexture;
    ref<Texture> mMaterialDataTexture;
    ref<Texture> mDepthTexture;

    ref<Fbo> mpFbo;

    struct
    {
        ref<GraphicsState> pState;
        ref<Program> pProgram;
        ref<ProgramVars> pVars;
    } mDepthPass;

    struct
    {
        ref<GraphicsState> pState;
        ref<Program> pProgram;
        ref<ProgramVars> pVars;
    } mGBufferPass;
};
} // namespace Restir
