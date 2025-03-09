#include "OptixDenoiserPass.h"
#include "GBuffer.h"

#include <optix_function_table_definition.h>

// Some debug macros
#define OPTIX_CHECK(call)                                                                                                            \
    {                                                                                                                                \
        OptixResult result = call;                                                                                                   \
        if (result != OPTIX_SUCCESS)                                                                                                 \
        {                                                                                                                            \
            FALCOR_THROW("Optix call {} failed with error {} ({}).", #call, optixGetErrorName(result), optixGetErrorString(result)); \
        }                                                                                                                            \
    }

#define CUDA_CHECK(call)                                                                                                          \
    {                                                                                                                             \
        cudaError_t result = call;                                                                                                \
        if (result != cudaSuccess)                                                                                                \
        {                                                                                                                         \
            FALCOR_THROW("CUDA call {} failed with error {} ({}).", #call, cudaGetErrorName(result), cudaGetErrorString(result)); \
        }                                                                                                                         \
    }

namespace Restir
{
void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    Falcor::logWarning("[Optix][{:2}][{:12}]: {}", level, tag, message);
}

OptixDeviceContext initOptix(Falcor::Device* pDevice)
{
    FALCOR_CHECK(pDevice->initCudaDevice(), "Failed to initialize CUDA device.");

    OPTIX_CHECK(optixInit());

    FALCOR_CHECK(g_optixFunctionTable.optixDeviceContextCreate, "OptiX function table not initialized.");

    // Build our OptiX context
    OptixDeviceContext optixContext;
    OPTIX_CHECK(optixDeviceContextCreate(pDevice->getCudaDevice()->getContext(), 0, &optixContext));

    // Tell Optix how to write to our Falcor log.
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, optixLogCallback, nullptr, 4));

    return optixContext;
}

FALCOR_ENUM_INFO(
    OptixDenoiserModelKind,
    {
        {OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_LDR, "LDR"},
        {OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR, "HDR"},
        {OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_AOV, "AOV"},
        {OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL, "Temporal"},
    }
);
FALCOR_ENUM_REGISTER(OptixDenoiserModelKind);

OptixDenoiserPass::OptixDenoiserPass(
    Falcor::ref<Falcor::Device> pDevice,
    Falcor::ref<Falcor::Scene> pScene,
    RenderContext* pRenderContext,
    Falcor::ref<Falcor::Texture>& inColorTexture,
    uint32_t width,
    uint32_t height
)
    : mpDevice(pDevice), mpScene(pScene), mInColorFromShadingPassTexture(inColorTexture), mWidth(width), mHeight(height)
{
    mpConvertTexToBuf = ComputePass::create(mpDevice, "Samples/Restir/OptixDenoiserPass_ConvertTexToBuf.slang", "main");
    mpConvertNormalsToBuf = ComputePass::create(mpDevice, "Samples/Restir/OptixDenoiserPass_ConvertNormalsToBuf.slang", "main");
    mpComputeMotionVectors = ComputePass::create(mpDevice, "Samples/Restir/OptixDenoiserPass_ComputeMotionVectors.slang", "main");
    mpConvertBufToTex = FullScreenPass::create(mpDevice, "Samples/Restir/OptixDenoiserPass_ConvertBufToTex.slang");
    mpFbo = Fbo::create(mpDevice);

    mOutputTexture = pDevice->createTexture2D(
        mWidth,
        mHeight,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget |
            ResourceBindFlags::Shared
    );
    mOutputTexture->setName("OptixDenoiserPass_Output");

    compile(pRenderContext);
    setupDenoiser();
}

OptixDenoiserPass::~OptixDenoiserPass()
{
    cuda_utils::freeSharedDevicePtr((void*)mDenoiser.interop.denoiserInput.devicePtr);
    cuda_utils::freeSharedDevicePtr((void*)mDenoiser.interop.denoiserOutput.devicePtr);
    cuda_utils::freeSharedDevicePtr((void*)mDenoiser.interop.normal.devicePtr);
    cuda_utils::freeSharedDevicePtr((void*)mDenoiser.interop.albedo.devicePtr);
    cuda_utils::freeSharedDevicePtr((void*)mDenoiser.interop.motionVec.devicePtr);

    optixDenoiserDestroy(mDenoiser.denoiser);
}

void OptixDenoiserPass::compile(RenderContext* pRenderContext)
{
    // Initialize OptiX context.
    mOptixContext = initOptix(mpDevice.get());

    // Set correct parameters for the provided inputs.
    mDenoiser.options.guideNormal = 1u;
    mDenoiser.options.guideAlbedo = 1u;

    // If the user specified a denoiser on initialization, respect that.  Otherwise, choose the "best"
    mSelectedModel = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL;
    mDenoiser.modelKind = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL;

    // (Re-)allocate temporary buffers when render resolution changes
    uint2 newSize = uint2(mWidth, mHeight);

    // If allowing tiled denoising, these may be smaller than the window size (TODO; not currently handled)
    mDenoiser.tileWidth = newSize.x;
    mDenoiser.tileHeight = newSize.y;

    // Reallocate / reszize our staging buffers for transferring data to and from OptiX / CUDA / DXR
    allocateStagingBuffers(pRenderContext);

    // Size intensity and hdrAverage buffers correctly.  Only one at a time is used, but these are small, so create them both
    if (mDenoiser.intensityBuffer.getSize() != (1 * sizeof(float)))
        mDenoiser.intensityBuffer.resize(1 * sizeof(float));
    if (mDenoiser.hdrAverageBuffer.getSize() != (3 * sizeof(float)))
        mDenoiser.hdrAverageBuffer.resize(3 * sizeof(float));

    // Create an intensity GPU buffer to pass to OptiX when appropriate
    if (!mDenoiser.kernelPredictionMode || !mDenoiser.useAOVs)
    {
        mDenoiser.params.hdrIntensity = mDenoiser.intensityBuffer.getDevicePtr();
        mDenoiser.params.hdrAverageColor = static_cast<CUdeviceptr>(0);
    }
    else // Create an HDR average color GPU buffer to pass to OptiX when appropriate
    {
        mDenoiser.params.hdrIntensity = static_cast<CUdeviceptr>(0);
        mDenoiser.params.hdrAverageColor = mDenoiser.hdrAverageBuffer.getDevicePtr();
    }
}

void OptixDenoiserPass::allocateStagingBuffers(RenderContext* pRenderContext)
{
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.denoiserInput, mDenoiser.layer.input);
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.denoiserOutput, mDenoiser.layer.output);
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.normal, mDenoiser.guideLayer.normal);
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.albedo, mDenoiser.guideLayer.albedo);
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.motionVec, mDenoiser.guideLayer.flow, OPTIX_PIXEL_FORMAT_FLOAT2);
}

void OptixDenoiserPass::allocateStagingBuffer(RenderContext* pRenderContext, Interop& interop, OptixImage2D& image, OptixPixelFormat format)
{
    uint32_t elemSize = 4 * sizeof(float);
    ResourceFormat falcorFormat = ResourceFormat::RGBA32Float;
    switch (format)
    {
    case OPTIX_PIXEL_FORMAT_FLOAT4:
        elemSize = 4 * sizeof(float);
        falcorFormat = ResourceFormat::RGBA32Float;
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT3:
        elemSize = 3 * sizeof(float);
        falcorFormat = ResourceFormat::RGB32Float;
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT2:
        elemSize = 2 * sizeof(float);
        falcorFormat = ResourceFormat::RG32Float;
        break;
    default:
        FALCOR_THROW("OptixDenoiser called allocateStagingBuffer() with unsupported format");
    }

    // Create a new DX <-> CUDA shared buffer using the Falcor API to create, then find its CUDA pointer.
    interop.buffer = mpDevice->createTypedBuffer(
        falcorFormat,
        mWidth * mHeight,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget | ResourceBindFlags::Shared
    );
    interop.devicePtr = (CUdeviceptr)exportBufferToCudaDevice(interop.buffer);

    // Setup an OptiXImage2D structure so OptiX will used this new buffer for image data
    image.width = mWidth;
    image.height = mHeight;
    image.rowStrideInBytes = mWidth * elemSize;
    image.pixelStrideInBytes = elemSize;
    image.format = format;
    image.data = interop.devicePtr;
}

void OptixDenoiserPass::freeStagingBuffer(Interop& interop, OptixImage2D& image)
{
    cuda_utils::freeSharedDevicePtr((void*)interop.devicePtr);
    interop.buffer = nullptr;
    image.data = static_cast<CUdeviceptr>(0);
}

void OptixDenoiserPass::render(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "OptixDenoiserPass::render");

    if (mFirstFrame)
        mPreviousFrameViewProjMat = mpScene->getCamera()->getViewProjMatrix();

    const uint2 bufferSize = uint2(mWidth, mHeight);

    convertTexToBuf(pRenderContext, mInColorFromShadingPassTexture, mDenoiser.interop.denoiserInput.buffer, bufferSize);
    convertTexToBuf(pRenderContext, GBufferSingleton::instance()->getAlbedoTexture(), mDenoiser.interop.albedo.buffer, bufferSize);

    convertNormalsToBuf(
        pRenderContext,
        GBufferSingleton::instance()->getCurrentNormalWsTexture(),
        mDenoiser.interop.normal.buffer,
        bufferSize,
        transpose(inverse(mpScene->getCamera()->getViewMatrix()))
    );

    computeMotionVectors(pRenderContext, mDenoiser.interop.motionVec.buffer, bufferSize);

    pRenderContext->waitForFalcor();

    // Compute average intensity, if needed
    if (mDenoiser.params.hdrIntensity)
    {
        optixDenoiserComputeIntensity(
            mDenoiser.denoiser,
            nullptr, // CUDA stream
            &mDenoiser.layer.input,
            mDenoiser.params.hdrIntensity,
            mDenoiser.scratchBuffer.getDevicePtr(),
            mDenoiser.scratchBuffer.getSize()
        );
    }

    // Compute average color, if needed
    if (mDenoiser.params.hdrAverageColor)
    {
        optixDenoiserComputeAverageColor(
            mDenoiser.denoiser,
            nullptr, // CUDA stream
            &mDenoiser.layer.input,
            mDenoiser.params.hdrAverageColor,
            mDenoiser.scratchBuffer.getDevicePtr(),
            mDenoiser.scratchBuffer.getSize()
        );
    }

    // On the first frame with a new denoiser, we have no prior input for temporal denoising.
    //    In this case, pass in our current frame as both the current and prior frame.
    if (mFirstFrame)
    {
        mDenoiser.layer.previousOutput = mDenoiser.layer.input;
    }

    // Run denoiser
    optixDenoiserInvoke(
        mDenoiser.denoiser,
        nullptr, // CUDA stream
        &mDenoiser.params,
        mDenoiser.stateBuffer.getDevicePtr(),
        mDenoiser.stateBuffer.getSize(),
        &mDenoiser.guideLayer, // Our set of normal / albedo / motion vector guides
        &mDenoiser.layer,      // Array of input or AOV layers (also contains denoised per-layer outputs)
        1u,                    // Nuumber of layers in the above array
        0u,                    // (Tile) Input offset X
        0u,                    // (Tile) Input offset Y
        mDenoiser.scratchBuffer.getDevicePtr(),
        mDenoiser.scratchBuffer.getSize()
    );

    pRenderContext->waitForCuda();

    // Copy denoised output buffer to texture for Falcor to consume
    convertBufToTex(pRenderContext, mDenoiser.interop.denoiserOutput.buffer, mOutputTexture, bufferSize);

    // Make sure we set the previous frame output to the correct location for future frames.
    // Everything in this if() cluase could happen every frame, but is redundant after the first frame.
    if (mFirstFrame)
    {
        // Note: This is a deep copy that can dangerously point to deallocated memory when resetting denoiser settings.
        // This is (partly) why in the first frame, the layer.previousOutput is set to layer.input, above.
        mDenoiser.layer.previousOutput = mDenoiser.layer.output;

        // We're no longer in the first frame of denoising; no special processing needed now.
        mFirstFrame = false;
    }

    mPreviousFrameViewProjMat = mpScene->getCamera()->getViewProjMatrix();
}

void* OptixDenoiserPass::exportBufferToCudaDevice(ref<Buffer>& buf)
{
    if (buf == nullptr)
        return nullptr;
    return cuda_utils::getSharedDevicePtr(buf->getDevice()->getType(), buf->getSharedApiHandle(), (uint32_t)buf->getSize());
}

void OptixDenoiserPass::setupDenoiser()
{
    // Create the denoiser
    optixDenoiserCreate(mOptixContext, mDenoiser.modelKind, &mDenoiser.options, &mDenoiser.denoiser);

    // Find out how much memory is needed for the requested denoiser
    optixDenoiserComputeMemoryResources(mDenoiser.denoiser, mDenoiser.tileWidth, mDenoiser.tileHeight, &mDenoiser.sizes);

    // Allocate/resize some temporary CUDA buffers for internal OptiX processing/state
    mDenoiser.scratchBuffer.resize(mDenoiser.sizes.withoutOverlapScratchSizeInBytes);
    mDenoiser.stateBuffer.resize(mDenoiser.sizes.stateSizeInBytes);

    // Finish setup of the denoiser
    optixDenoiserSetup(
        mDenoiser.denoiser,
        nullptr,
        mDenoiser.tileWidth + 2 * mDenoiser.tileOverlap,  // Should work with tiling if parameters set appropriately
        mDenoiser.tileHeight + 2 * mDenoiser.tileOverlap, // Should work with tiling if parameters set appropriately
        mDenoiser.stateBuffer.getDevicePtr(),
        mDenoiser.stateBuffer.getSize(),
        mDenoiser.scratchBuffer.getDevicePtr(),
        mDenoiser.scratchBuffer.getSize()
    );
}

void OptixDenoiserPass::computeMotionVectors(RenderContext* pRenderContext, const ref<Buffer>& buf, const uint2& size)
{
    FALCOR_PROFILE(pRenderContext, "OptixDenoiserPass::computeMotionVectors");

    auto var = mpComputeMotionVectors->getRootVar();

    var["GlobalCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["GlobalCB"]["previousFrameViewProjMat"] = transpose(mPreviousFrameViewProjMat);

    var["gInPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();

    var["gOutBuf"] = buf;

    mpComputeMotionVectors->execute(pRenderContext, size.x, size.y);
}

void OptixDenoiserPass::convertTexToBuf(RenderContext* pRenderContext, const ref<Texture>& tex, const ref<Buffer>& buf, const uint2& size)
{
    FALCOR_PROFILE(pRenderContext, "OptixDenoiserPass::convertTexToBuf");

    auto var = mpConvertTexToBuf->getRootVar();
    var["GlobalCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["gInTex"] = tex;
    var["gOutBuf"] = buf;
    mpConvertTexToBuf->execute(pRenderContext, size.x, size.y);
}

void OptixDenoiserPass::convertNormalsToBuf(
    RenderContext* pRenderContext,
    const ref<Texture>& tex,
    const ref<Buffer>& buf,
    const uint2& size,
    float4x4 viewIT
)
{
    FALCOR_PROFILE(pRenderContext, "OptixDenoiserPass::convertNormalsToBuf");

    auto var = mpConvertNormalsToBuf->getRootVar();
    var["GlobalCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["GlobalCB"]["gViewIT"] = viewIT;
    var["gInTex"] = tex;
    var["gOutBuf"] = buf;
    mpConvertNormalsToBuf->execute(pRenderContext, size.x, size.y); // mpConvertTexToBuf->execute(pRenderContext, size.x, size.y);
}

void OptixDenoiserPass::convertBufToTex(RenderContext* pRenderContext, const ref<Buffer>& buf, const ref<Texture>& tex, const uint2& size)
{
    FALCOR_PROFILE(pRenderContext, "OptixDenoiserPass::convertBufToTex");

    auto var = mpConvertBufToTex->getRootVar();
    var["GlobalCB"]["gStride"] = size.x;
    var["gInBuf"] = buf;
    mpFbo->attachColorTarget(tex, 0);
    mpConvertBufToTex->execute(pRenderContext, mpFbo);
}
} // namespace Restir

