
#include "RestirApp.h"

#include "ApplicationPathsManager.h"
#include "LightManager.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"
#include <sstream>
#include <windows.h>
#include <iostream>

FALCOR_EXPORT_D3D12_AGILITY_SDK

// THE SCENE WE USE.
#define SCENE_NAME 1

// WE WANT TO USE TEMPORAL FILTERING
#define USE_TEMPORAL_FILTERING 1

// WE DONT WANT TO USE SPATIAL FILTERING
#define USE_SPATIAL_FILTERING 0

// WE WANT TO USE DENOISING
#define USE_DENOISING 1

#if SCENE_NAME == 0
static const Restir::SceneName kSceneName = Restir::SceneName::Arcade;
#elif SCENE_NAME == 1
// To work model is required. READ TestScenes\DragonBuddha\README.txt
static const Restir::SceneName kSceneName = Restir::SceneName::DragonBuddha;
#elif SCENE_NAME == 2
static const Restir::SceneName kSceneName = Restir::SceneName::BreakfastRoom;
#else
static const Restir::SceneName kSceneName = Restir::SceneName::Sponza;
#endif

// https://stackoverflow.com/questions/4804298/how-to-convert-wstring-into-string
std::wstring ExePath()
{
    TCHAR buffer[MAX_PATH] = {0};
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::wstring::size_type pos = std::wstring(buffer).find_last_of(L"\\/");
    return std::wstring(buffer).substr(0, pos);
}

RestirApp::RestirApp(const SampleAppConfig& config) : SampleApp(config) {}

RestirApp::~RestirApp() {}

void RestirApp::onLoad(RenderContext* pRenderContext)
{
    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        FALCOR_THROW("Device does not support raytracing!");
    }

    Restir::ApplicationPathsManagerSingleton::create();

    std::wstring exePathW = ExePath();
    std::string exePath(exePathW.begin(), exePathW.end());
    Restir::ApplicationPathsManagerSingleton::instance()->setExePath(exePath);

    if (kSceneName == Restir::SceneName::DragonBuddha)
    {
        const std::string str = exePath + "/../../../../TestScenes/DragonBuddha/dragonbuddha.pyscene";
        Restir::ApplicationPathsManagerSingleton::instance()->setScenePath(str);
    }
    else if (kSceneName == Restir::SceneName::BreakfastRoom)
    {
        const std::string str = exePath + "/../../../../TestScenes/BreakfastRoom/breakfast_room.pyscene";
        Restir::ApplicationPathsManagerSingleton::instance()->setScenePath(str);
    }
    else if (kSceneName == Restir::SceneName::Sponza)
    {
        const std::string str = exePath + "/../../../../TestScenes/Bugged/Sponza/sponza.pyscene";
        Restir::ApplicationPathsManagerSingleton::instance()->setScenePath(str);
    }
    else
    {
        Restir::ApplicationPathsManagerSingleton::instance()->setScenePath("Arcade/Arcade.pyscene");
    }

    const std::string str = exePath + "/../../../../TestScenes/Shared/";
    Restir::ApplicationPathsManagerSingleton::instance()->setSharedDataPath(str);

    loadScene(getTargetFbo().get(), pRenderContext);
}

void RestirApp::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    if (mpCamera)
    {
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }
}

void RestirApp::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
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

void RestirApp::onGuiRender(Gui* pGui) {}

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

void RestirApp::loadScene(const Fbo* pTargetFbo, RenderContext* pRenderContext)
{
    mpScene = Scene::create(getDevice(), Restir::ApplicationPathsManagerSingleton::instance()->getScenePath());
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    // Create scene settings singleton.
    Restir::SceneSettingsSingleton::create();
    switch (kSceneName)
    {
    case Restir::SceneName::Arcade:
        Restir::SceneSettingsSingleton::instance()->RISSamplesCount = 16;
        Restir::SceneSettingsSingleton::instance()->nbReservoirPerPixel = 3;
        Restir::SceneSettingsSingleton::instance()->shadingLightExponent = 8.0f;
        break;

    case Restir::SceneName::DragonBuddha:
        Restir::SceneSettingsSingleton::instance()->RISSamplesCount = 32;
        Restir::SceneSettingsSingleton::instance()->nbReservoirPerPixel = 4;
        break;

    case Restir::SceneName::BreakfastRoom:
        Restir::SceneSettingsSingleton::instance()->RISSamplesCount = 32;
        Restir::SceneSettingsSingleton::instance()->nbReservoirPerPixel = 4;
        Restir::SceneSettingsSingleton::instance()->shadingLightExponent = 4.0f;
        break;

    case Restir::SceneName::Sponza:
        Restir::SceneSettingsSingleton::instance()->RISSamplesCount = 32;
        Restir::SceneSettingsSingleton::instance()->nbReservoirPerPixel = 8;
        break;
    }

    // Create the remaining singletons.
    Restir::GBufferSingleton::create();
    Restir::GBufferSingleton::instance()->init(getDevice(), mpScene, pTargetFbo->getWidth(), pTargetFbo->getHeight());

    Restir::LightManagerSingleton::create();
    Restir::LightManagerSingleton::instance()->init(getDevice(), mpScene, kSceneName);

    Restir::ReservoirManagerSingleton::create();
    Restir::ReservoirManagerSingleton::instance()->init(getDevice(), pTargetFbo->getWidth(), pTargetFbo->getHeight());

    // Create the render passes.
    mpRISPass = new Restir::RISPass(getDevice(), pTargetFbo->getWidth(), pTargetFbo->getHeight());
    mpVisibilityPass = new Restir::VisibilityPass(getDevice(), mpScene, pTargetFbo->getWidth(), pTargetFbo->getHeight());

#if USE_TEMPORAL_FILTERING
    mpTemporalFilteringPass =
        new Restir::TemporalFilteringPass(getDevice(), mpScene, kSceneName, pTargetFbo->getWidth(), pTargetFbo->getHeight());
#endif

#if USE_SPATIAL_FILTERING
    mpSpatialFilteringPass = new Restir::SpatialFilteringPass(getDevice(), mpScene, pTargetFbo->getWidth(), pTargetFbo->getHeight());
#endif

    mpShadingPass = new Restir::ShadingPass(getDevice(), pTargetFbo->getWidth(), pTargetFbo->getHeight());

#if USE_DENOISING
#if DENOISING_NRD
    mpDenoisingPass = new Restir::NRDDenoiserPass(
        getDevice(), pRenderContext, mpScene, mpShadingPass->getOuputTexture(), pTargetFbo->getWidth(), pTargetFbo->getHeight()
    );
#else
    mpDenoisingPass = new Restir::OptixDenoiserPass(
        getDevice(), mpScene, pRenderContext, mpShadingPass->getOuputTexture(), pTargetFbo->getWidth(), pTargetFbo->getHeight()
    );
#endif
#endif
}

void RestirApp::render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "RestirApp::render");

    /*
    std::cout << mpCamera->getPosition().x << " " << mpCamera->getPosition().y << " " << mpCamera->getPosition().z << std::endl;
    std::cout << mpCamera->getTarget().x << " " << mpCamera->getTarget().y << " " << mpCamera->getTarget().z << std::endl;
    std::cout << mpCamera->getUpVector().x << " " << mpCamera->getUpVector().y << " " << mpCamera->getUpVector().z << std::endl;
    std::cout << mpCamera->getFocalLength() << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;
    */
    Restir::GBufferSingleton::instance()->render(pRenderContext);
    mpRISPass->render(pRenderContext, mpCamera);
    mpVisibilityPass->render(pRenderContext);

#if USE_TEMPORAL_FILTERING
    mpTemporalFilteringPass->render(pRenderContext);
#endif

#if USE_SPATIAL_FILTERING
    mpSpatialFilteringPass->render(pRenderContext);
#endif

    mpShadingPass->render(pRenderContext, mpCamera);

#if USE_DENOISING
    mpDenoisingPass->render(pRenderContext);
    pRenderContext->blit(mpDenoisingPass->getOuputTexture()->getSRV(), pTargetFbo->getRenderTargetView(0));
#else
    pRenderContext->blit(mpShadingPass->getOuputTexture()->getSRV(), pTargetFbo->getRenderTargetView(0));
#endif

    Restir::GBufferSingleton::instance()->setNextFrame();
    Restir::ReservoirManagerSingleton::instance()->setNextFrame();
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "HelloRestir";
    config.windowDesc.resizableWindow = true;

    RestirApp helloRestir(config);
    return helloRestir.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
