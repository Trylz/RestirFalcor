#pragma once
#include "Falcor.h"

namespace Restir
{
class RISPass
{
public:
    RISPass(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height);

    void render(Falcor::RenderContext* pRenderContext, Falcor::ref<Falcor::Camera> pCamera);

private:
    uint32_t mWidth;
    uint32_t mHeight;

    uint32_t mSampleIndex = 0u;

    Falcor::ref<Falcor::ComputePass> mpRISPass;
};
} // namespace Restir

