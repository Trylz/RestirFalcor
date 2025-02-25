#pragma once

#include "Falcor.h"
#include "GBuffer.h"
#include "Core/SampleApp.h"

using namespace Falcor;

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
    void loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo);
    void setPerFrameVars(const Fbo* pTargetFbo);
    void render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

    ref<Scene> mpScene;
    ref<Camera> mpCamera;

    // The restir passes.
    ref<Restir::GBuffer> mpGufferPass;
};
