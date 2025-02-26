#pragma once
#include "../RenderPasses/GBuffer/GBuffer/GBufferRaster.h"
#include "Singleton.h"

namespace Restir
{
class GBuffer 
{
public:
    GBuffer();

    void init(ref<Device> pDevice, ref<Scene> pScene, uint32_t width, uint32_t height);

    void render(RenderContext* pRenderContext);

    inline const ref<Texture>& getPositionWsTexture() const { return mPositionWsTexture; }
    inline const ref<Texture>& getNormalWsTexture() const { return mNormalWsTexture; }
    inline const ref<Texture>& getAlbedoTexture() const { return mAlbedoTexture; }
    inline const ref<Texture>& getSpecularTexture() const { return mSpecularTexture; }
    inline const ref<Texture>& getMotionVectorTexture() const { return mMotionVectorTexture; }

private:
    void createTextures();
    void compilePrograms();

    ref<Device> mpDevice;
    ref<Scene> mpScene;

    uint32_t mWidth;
    uint32_t mHeight;

    ref<Texture> mPositionWsTexture;
    ref<Texture> mNormalWsTexture;
    ref<Texture> mAlbedoTexture;
    ref<Texture> mSpecularTexture;
    ref<Texture> mMotionVectorTexture;

    ref<Program> mpRaytraceProgram;
    ref<RtProgramVars> mpRtVars;

    uint32_t mSampleIndex = 0u;
};

using GBufferSingleton = Singleton<GBuffer>;

} // namespace Restir
