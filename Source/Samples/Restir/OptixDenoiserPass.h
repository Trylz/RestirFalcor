#pragma once

#include "Falcor.h"
#include "Core/Enum.h"
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPass.h"

#include "Utils/CudaUtils.h"

#include <optix.h>
#include <optix_stubs.h>

using namespace Falcor;

namespace Restir
{
class CudaBuffer
{
public:
    CudaBuffer() {}

    CUdeviceptr getDevicePtr() { return (CUdeviceptr)mpDevicePtr; }
    size_t getSize() { return mSizeBytes; }

    void allocate(size_t size)
    {
        if (mpDevicePtr)
            free();
        mSizeBytes = size;
        FALCOR_CUDA_CHECK(cudaMalloc((void**)&mpDevicePtr, mSizeBytes));
    }

    void resize(size_t size) { allocate(size); }

    void free()
    {
        FALCOR_CUDA_CHECK(cudaFree(mpDevicePtr));
        mpDevicePtr = nullptr;
        mSizeBytes = 0;
    }

private:
    size_t mSizeBytes = 0;
    void* mpDevicePtr = nullptr;
};

OptixDeviceContext initOptix(Falcor::Device* pDevice);

// Based on RenderPasses\OptixDenoiser.
class OptixDenoiserPass
{
public:
    OptixDenoiserPass(
        Falcor::ref<Falcor::Device> pDevice,
        Falcor::ref<Falcor::Scene> pScene,
        RenderContext* pRenderContext,
        Falcor::ref<Falcor::Texture>& inColorTexture,
        uint32_t width,
        uint32_t height
    );

    ~OptixDenoiserPass();

    void render(RenderContext* pRenderContext);
    inline Falcor::ref<Falcor::Texture>& getOuputTexture() { return mOutputTexture; };


private:
    ref<Device> mpDevice;
    ref<Scene> mpScene;

    Falcor::ref<Falcor::Texture> mInColorFromShadingPassTexture;
    Falcor::ref<Falcor::Texture> mOutputTexture;

    void compile(RenderContext* pRenderContext);

    void setupDenoiser();

    void convertTexToBuf(RenderContext* pRenderContext, const ref<Texture>& tex, const ref<Buffer>& buf, const uint2& size);
    void convertNormalsToBuf(
        RenderContext* pRenderContext,
        const ref<Texture>& tex,
        const ref<Buffer>& buf,
        const uint2& size,
        float4x4 viewIT
    );
    void convertBufToTex(RenderContext* pRenderContext, const ref<Buffer>& buf, const ref<Texture>& tex, const uint2& size);
    void computeMotionVectors(RenderContext* pRenderContext, const ref<Buffer>& buf, const uint2& size);

    uint32_t mWidth;
    uint32_t mHeight;

    uint32_t mSelectedModel = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;

    bool mFirstFrame = true;

    Falcor::float4x4 mPreviousFrameViewProjMat;

    OptixDeviceContext mOptixContext = nullptr;

    struct Interop
    {
        ref<Buffer> buffer;                     ///< Falcor buffer
        CUdeviceptr devicePtr = (CUdeviceptr)0; ///< CUDA pointer to buffer
    };

    struct
    {
        OptixDenoiserOptions options = {0u, 0u};
        OptixDenoiserModelKind modelKind = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;
        OptixDenoiser denoiser = nullptr;

        // OptixDenoiserParams params = {static_cast<CUdeviceptr>(0), 0.0f, static_cast<CUdeviceptr>(0), 0u};
        OptixDenoiserParams params = {0u, static_cast<CUdeviceptr>(0), 0.0f, static_cast<CUdeviceptr>(0)};

        OptixDenoiserSizes sizes = {};

        bool kernelPredictionMode = false;
        bool useAOVs = false;
        uint32_t tileOverlap = 0u;

        uint32_t tileWidth = 0u;
        uint32_t tileHeight = 0u;

        OptixDenoiserGuideLayer guideLayer = {};

        OptixDenoiserLayer layer = {};

        struct Intermediates
        {
            Interop normal;
            Interop albedo;
            Interop motionVec;
            Interop denoiserInput;
            Interop denoiserOutput;
        } interop;

        CudaBuffer scratchBuffer, stateBuffer, intensityBuffer, hdrAverageBuffer;

    } mDenoiser;


    ref<ComputePass> mpConvertTexToBuf;
    ref<ComputePass> mpConvertNormalsToBuf;
    ref<ComputePass> mpComputeMotionVectors;
    ref<FullScreenPass> mpConvertBufToTex;
    ref<Fbo> mpFbo;

    void allocateStagingBuffer(
        RenderContext* pRenderContext,
        Interop& interop,
        OptixImage2D& image,
        OptixPixelFormat format = OPTIX_PIXEL_FORMAT_FLOAT4
    );

    void freeStagingBuffer(Interop& interop, OptixImage2D& image);

    void allocateStagingBuffers(RenderContext* pRenderContext);

    void* exportBufferToCudaDevice(ref<Buffer>& buf);
};
} // namespace Restir

