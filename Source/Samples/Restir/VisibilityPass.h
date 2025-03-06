#pragma once

#include "Falcor.h"

namespace Restir
{
class VisibilityPass
{
public:
    VisibilityPass(Falcor::ref<Falcor::Device> pDevice, Falcor::ref<Falcor::Scene> pScene, uint32_t width, uint32_t height);

    void render(Falcor::RenderContext* pRenderContext);

private:
    void compileProgram(Falcor::ref<Falcor::Device> pDevice);

    Falcor::ref<Falcor::Scene> mpScene;

    uint32_t mWidth;
    uint32_t mHeight;

    Falcor::ref<Falcor::Program> mpRaytraceProgram;
    Falcor::ref<Falcor::RtProgramVars> mpRtVars;
};
} // namespace Restir

