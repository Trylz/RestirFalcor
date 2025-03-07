#pragma once

#include "Falcor.h"

#include "Core/Enum.h"
#include "Core/API/Shared/D3D12DescriptorSet.h"
#include "Core/API/Shared/D3D12RootSignature.h"
#include "Core/API/Shared/D3D12ConstantBufferView.h"
#include "RenderGraph/RenderPassHelpers.h"

#include "Dependencies/NvidiaNRD/Include/NRD.h"

namespace Restir
{
using namespace Falcor;

// Addapted from RenderPasses\NRDPass
class NRDDenoiserPass
{
public:
    NRDDenoiserPass(
        Falcor::ref<Falcor::Device> pDevice,
        Falcor::RenderContext* pRenderContext,
        Falcor::ref<Falcor::Scene> pScene,
        Falcor::ref<Falcor::Texture>& inColor,
        uint32_t width,
        uint32_t height
    );
    ~NRDDenoiserPass();

    void render(Falcor::RenderContext* pRenderContext);
    inline Falcor::ref<Falcor::Texture>& getOuputTexture() { return mOuputTexture; };

private:
    void initNRD();
    void createPipelines();
    void createResources();

    void createFalcorTextures(Falcor::ref<Falcor::Device> pDevice);

    void packNRD(Falcor::RenderContext* pRenderContext);
    void dipatchNRD(Falcor::RenderContext* pRenderContext);
    void unpackNRD(Falcor::RenderContext* pRenderContext);

    void dispatch(Falcor::RenderContext* pRenderContext, const nrd::DispatchDesc& dispatchDesc);

    void populateCommonSettings(nrd::CommonSettings& settings);
    void populateDenoiserSettings(nrd::RelaxDiffuseSettings& settings);

    Falcor::ref<Falcor::Device> mpDevice;
    Falcor::ref<Falcor::Scene> mpScene;

    Falcor::ref<Falcor::ComputePass> mpPackNRDPass;
    Falcor::ref<Falcor::ComputePass> mpUnpackNRDPass;

    uint32_t mWidth;
    uint32_t mHeight;

    uint32_t mFrameIndex = 0u;

    nrd::Denoiser* mpDenoiser = nullptr;

    Falcor::ref<Falcor::Texture> mViewZTexture;
    Falcor::ref<Falcor::Texture> mMotionVectorTexture;
    Falcor::ref<Falcor::Texture> mNormalLinearRoughnessTexture;
    Falcor::ref<Falcor::Texture> mOuputTexture;

    Falcor::ref<Falcor::Texture> m_InColorTexture; // From Shading pass.

    Falcor::float4x4 mPreviousFrameViewMat;
    Falcor::float4x4 mPreviousFrameProjMat;
    Falcor::float4x4 mPreviousFrameViewProjMat;

    std::vector<ref<Sampler>> mpSamplers;
    std::vector<D3D12DescriptorSetLayout> mCBVSRVUAVdescriptorSetLayouts;
    ref<D3D12DescriptorSet> mpSamplersDescriptorSet;
    std::vector<ref<D3D12RootSignature>> mpRootSignatures;
    std::vector<ref<ComputePass>> mpPasses;
    std::vector<ref<const ProgramKernels>> mpCachedProgramKernels;
    std::vector<ref<ComputeStateObject>> mpCSOs;
    std::vector<ref<Texture>> mpPermanentTextures;
    std::vector<ref<Texture>> mpTransientTextures;
    ref<D3D12ConstantBufferView> mpCBV;
};
} // namespace Restir
