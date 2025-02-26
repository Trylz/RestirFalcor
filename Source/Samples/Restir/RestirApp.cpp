
#include "RestirApp.h"
#include "LightManager.h"
#include "ReservoirManager.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"

FALCOR_EXPORT_D3D12_AGILITY_SDK

static const float4 kClearColor(0.38f, 0.52f, 0.10f, 1);
static const std::string kDefaultScene = "Arcade/Arcade.pyscene";

RestirApp::RestirApp(const SampleAppConfig& config) : SampleApp(config) {}

RestirApp::~RestirApp() {}

void RestirApp::onLoad(RenderContext* pRenderContext)
{
    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        FALCOR_THROW("Device does not support raytracing!");
    }

    loadScene(kDefaultScene, getTargetFbo().get());
}

void RestirApp::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    if (mpCamera)
    {
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }
}

void RestirApp::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpScene)
    {
        IScene::UpdateFlags updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
        if (is_set(updates, IScene::UpdateFlags::GeometryChanged))
            FALCOR_THROW("This sample does not support scene geometry changes.");
        if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
            FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

        render(pRenderContext, pTargetFbo);
    }

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void RestirApp::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Hello DXR Settings", {300, 400}, {10, 80});

    // w.checkbox("Ray Trace", mRayTrace);
    // w.checkbox("Use Depth of Field", mUseDOF);
    if (w.button("Load Scene"))
    {
        std::filesystem::path path;
        if (openFileDialog(Scene::getFileExtensionFilters(), path))
        {
            loadScene(path, getTargetFbo().get());
        }
    }

    mpScene->renderUI(w);
}

bool RestirApp::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == Input::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        // mRayTrace = !mRayTrace;
        return true;
    }

    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return true;

    return false;
}

bool RestirApp::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent);
}

void RestirApp::loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo)
{
    mpScene = Scene::create(getDevice(), path);
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    // Create the singletons.
    Restir::GBufferSingleton::create();
    Restir::GBufferSingleton::instance()->init(getDevice(), mpScene, pTargetFbo->getWidth(), pTargetFbo->getHeight());

    Restir::LightManagerSingleton::create();
    Restir::LightManagerSingleton::instance()->init(getDevice(), mpScene);

    Restir::ReservoirManagerSingleton::create();
    Restir::ReservoirManagerSingleton::instance()->init(getDevice(), pTargetFbo->getWidth(), pTargetFbo->getHeight());

    // Create the render passes.
    mpRISPass.reset(new Restir::RISPass(getDevice(), pTargetFbo->getWidth(), pTargetFbo->getHeight()));
}

void RestirApp::render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "RestirApp::render");

    Restir::GBufferSingleton::instance()->render(pRenderContext);
    mpRISPass->render(pRenderContext, mpCamera);

    pRenderContext->blit(Restir::GBufferSingleton::instance()->getAlbedoTexture()->getSRV(), pTargetFbo->getRenderTargetView(0));
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "HelloDXR";
    config.windowDesc.resizableWindow = true;

    RestirApp helloDXR(config);
    return helloDXR.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
