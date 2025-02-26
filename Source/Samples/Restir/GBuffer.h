#pragma once

#include "Singleton.h"

namespace Restir
{
class GBuffer 
{
public:
    GBuffer();

    void init(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene, uint32_t width, uint32_t height);

    void render(Falcor::RenderContext* pRenderContext);

    inline const Falcor::ref<Falcor::Texture>& getPositionWsTexture() const { return mPositionWsTexture; }
    inline const Falcor::ref<Falcor::Texture>& getNormalWsTexture() const { return mNormalWsTexture; }
    inline const Falcor::ref<Falcor::Texture>& getAlbedoTexture() const { return mAlbedoTexture; }
    inline const Falcor::ref<Falcor::Texture>& getSpecularTexture() const { return mSpecularTexture; }
    inline const Falcor::ref<Falcor::Texture>& getMotionVectorTexture() const { return mMotionVectorTexture; }

private:
    void createTextures();
    void compilePrograms();

    Falcor::ref<Falcor::Device> mpDevice;
    Falcor::ref<Falcor::Scene> mpScene;

    uint32_t mWidth;
    uint32_t mHeight;

    Falcor::ref<Falcor::Texture> mPositionWsTexture;
    Falcor::ref<Falcor::Texture> mNormalWsTexture;
    Falcor::ref<Falcor::Texture> mAlbedoTexture;
    Falcor::ref<Falcor::Texture> mSpecularTexture;
    Falcor::ref<Falcor::Texture> mMotionVectorTexture;

    Falcor::ref<Falcor::Program> mpRaytraceProgram;
    Falcor::ref<Falcor::RtProgramVars> mpRtVars;

    uint32_t mSampleIndex = 0u;
};

using GBufferSingleton = Singleton<GBuffer>;

} // namespace Restir
