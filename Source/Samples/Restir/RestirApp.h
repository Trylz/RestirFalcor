#pragma once

#include "Falcor.h"
#include "NRDDenoiserPass.h"
#include "GBuffer.h"
#include "OptixDenoiserPass.h"
#include "RISPass.h"
#include "ShadingPass.h"
#include "SpatialFilteringPass.h"
#include "TemporalFilteringPass.h"
#include "VisibilityPass.h"
#include "Core/SampleApp.h"

using namespace Falcor;

#define DENOISING_NRD 0

class RestirApp : public SampleApp
{
public:
    RestirApp(const SampleAppConfig& config);
    ~RestirApp();

    void onLoad(RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void loadScene(const Fbo* pTargetFbo, RenderContext* pRenderContext);
    void render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

    ref<Scene> mpScene;
    ref<Camera> mpCamera;

    Restir::RISPass* mpRISPass = nullptr;
    Restir::VisibilityPass* mpVisibilityPass = nullptr;
    Restir::ShadingPass* mpShadingPass = nullptr;
    Restir::TemporalFilteringPass* mpTemporalFilteringPass = nullptr;
    Restir::SpatialFilteringPass* mpSpatialFilteringPass = nullptr;

#if DENOISING_NRD
    Restir::NRDDenoiserPass* mpDenoisingPass = nullptr;
#else
    Restir::OptixDenoiserPass* mpDenoisingPass = nullptr;
#endif
};
