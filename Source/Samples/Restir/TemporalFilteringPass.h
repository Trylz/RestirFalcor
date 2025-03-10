#pragma once
#include "Falcor.h"
#include "SceneName.h"

namespace Restir
{
class TemporalFilteringPass
{
public:
    TemporalFilteringPass(
        Falcor::ref<Falcor::Device> pDevice,
        Falcor::ref<Falcor::Scene> pScene,
        uint32_t width,
        uint32_t height
    );

    void render(Falcor::RenderContext* pRenderContext);

private:
    Falcor::ref<Falcor::Scene> mpScene;

    uint32_t mWidth;
    uint32_t mHeight;
    Falcor::float4x4 mPreviousFrameViewProjMat;
    uint32_t mSampleIndex = 0u;
    Falcor::ref<Falcor::ComputePass> mpTemporalFilteringPass;
};
} // namespace Restir
