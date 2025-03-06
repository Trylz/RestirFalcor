#pragma once
#include "Falcor.h"

namespace Restir
{
class SpatialFilteringPass
{
public:
    SpatialFilteringPass(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene, uint32_t width, uint32_t height);

    void render(Falcor::RenderContext* pRenderContext);

    void performSpatialFiltering(Falcor::RenderContext* pRenderContext);
    void performReservoirCopy(Falcor::RenderContext* pRenderContext);

private:
    Falcor::ref<Falcor::Scene> mpScene;

    uint32_t mWidth;
    uint32_t mHeight;

    Falcor::ref<Falcor::ComputePass> mpSpatialFilteringPass;
    Falcor::ref<Falcor::Buffer> mpStagingReservoirs;

    uint32_t mSampleIndex = 0u;
};
} // namespace Restir

