#pragma once
#include "Falcor.h"

namespace Restir
{
class ShadingPass
{
public:
    ShadingPass(Falcor::ref<Falcor::Device> pDevice, uint32_t width, uint32_t height);

    void render(Falcor::RenderContext* pRenderContext, Falcor::ref<Falcor::Camera> pCamera);

    inline Falcor::ref<Falcor::Texture>& getOuputTexture() { return mpOuputTexture; };

private:
    uint32_t mWidth;
    uint32_t mHeight;

    Falcor::ref<Falcor::ComputePass> mpShadingPass;
    Falcor::ref<Falcor::Texture> mpOuputTexture;
 
    Falcor::ref<Falcor::Texture> mpBlueNoiseTexture;
};
} // namespace Restir
