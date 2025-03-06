#include "DenoisingPass.h"
#include "GBuffer.h"
#include "Core/API/NativeHandleTraits.h"
#include "ReservoirManager.h"
#include "SceneSettings.h"

#include <slang-gfx.h>
#include <d3d12.h>

#if defined(_DEBUG)
#pragma comment(lib, __FILE__ "\\..\\Dependencies\\NvidiaNRD\\lib\\Debug\\NRD.lib")
#else
#pragma comment(lib, __FILE__ "\\..\\Dependencies\\NvidiaNRD\\lib\\Release\\NRD.lib")
#endif

#if defined _DEBUG
#include <cassert>
#define NRD_ASSERT(x) assert(x)
#else
#define NRD_ASSERT(x) (x)
#endif

namespace Restir
{
using namespace Falcor;

NRDPass::NRDPass(
    Falcor::ref<Falcor::Device> pDevice,
    Falcor::RenderContext* pRenderContext,
    Falcor::ref<Falcor::Scene> pScene,
    uint32_t width,
    uint32_t height
)
    : mpDevice(pDevice), mpScene(pScene), mpRenderContext(pRenderContext), mWidth(width), mHeight(height)
{
    mpDevice->requireD3D12();

    mpPackNRDPass = ComputePass::create(pDevice, "Samples/Restir/DenoisingPass_PackNRD.slang", "PackNRD");
    mpUnpackNRDPass = ComputePass::create(pDevice, "Samples/Restir/DenoisingPass_UnpackNRD.slang", "UnpackNRD");

    createFalcorTextures(pDevice);
    initNRD();
}

static void* nrdAllocate(void* userArg, size_t size, size_t alignment)
{
    return malloc(size);
}

static void* nrdReallocate(void* userArg, void* memory, size_t size, size_t alignment)
{
    return realloc(memory, size);
}

static void nrdFree(void* userArg, void* memory)
{
    free(memory);
}

static ResourceFormat getFalcorFormat(nrd::Format format)
{
    switch (format)
    {
    case nrd::Format::R8_UNORM:
        return ResourceFormat::R8Unorm;
    case nrd::Format::R8_SNORM:
        return ResourceFormat::R8Snorm;
    case nrd::Format::R8_UINT:
        return ResourceFormat::R8Uint;
    case nrd::Format::R8_SINT:
        return ResourceFormat::R8Int;
    case nrd::Format::RG8_UNORM:
        return ResourceFormat::RG8Unorm;
    case nrd::Format::RG8_SNORM:
        return ResourceFormat::RG8Snorm;
    case nrd::Format::RG8_UINT:
        return ResourceFormat::RG8Uint;
    case nrd::Format::RG8_SINT:
        return ResourceFormat::RG8Int;
    case nrd::Format::RGBA8_UNORM:
        return ResourceFormat::RGBA8Unorm;
    case nrd::Format::RGBA8_SNORM:
        return ResourceFormat::RGBA8Snorm;
    case nrd::Format::RGBA8_UINT:
        return ResourceFormat::RGBA8Uint;
    case nrd::Format::RGBA8_SINT:
        return ResourceFormat::RGBA8Int;
    case nrd::Format::RGBA8_SRGB:
        return ResourceFormat::RGBA8UnormSrgb;
    case nrd::Format::R16_UNORM:
        return ResourceFormat::R16Unorm;
    case nrd::Format::R16_SNORM:
        return ResourceFormat::R16Snorm;
    case nrd::Format::R16_UINT:
        return ResourceFormat::R16Uint;
    case nrd::Format::R16_SINT:
        return ResourceFormat::R16Int;
    case nrd::Format::R16_SFLOAT:
        return ResourceFormat::R16Float;
    case nrd::Format::RG16_UNORM:
        return ResourceFormat::RG16Unorm;
    case nrd::Format::RG16_SNORM:
        return ResourceFormat::RG16Snorm;
    case nrd::Format::RG16_UINT:
        return ResourceFormat::RG16Uint;
    case nrd::Format::RG16_SINT:
        return ResourceFormat::RG16Int;
    case nrd::Format::RG16_SFLOAT:
        return ResourceFormat::RG16Float;
    case nrd::Format::RGBA16_UNORM:
        return ResourceFormat::RGBA16Unorm;
    case nrd::Format::RGBA16_SNORM:
        return ResourceFormat::Unknown; // Not defined in Falcor
    case nrd::Format::RGBA16_UINT:
        return ResourceFormat::RGBA16Uint;
    case nrd::Format::RGBA16_SINT:
        return ResourceFormat::RGBA16Int;
    case nrd::Format::RGBA16_SFLOAT:
        return ResourceFormat::RGBA16Float;
    case nrd::Format::R32_UINT:
        return ResourceFormat::R32Uint;
    case nrd::Format::R32_SINT:
        return ResourceFormat::R32Int;
    case nrd::Format::R32_SFLOAT:
        return ResourceFormat::R32Float;
    case nrd::Format::RG32_UINT:
        return ResourceFormat::RG32Uint;
    case nrd::Format::RG32_SINT:
        return ResourceFormat::RG32Int;
    case nrd::Format::RG32_SFLOAT:
        return ResourceFormat::RG32Float;
    case nrd::Format::RGB32_UINT:
        return ResourceFormat::RGB32Uint;
    case nrd::Format::RGB32_SINT:
        return ResourceFormat::RGB32Int;
    case nrd::Format::RGB32_SFLOAT:
        return ResourceFormat::RGB32Float;
    case nrd::Format::RGBA32_UINT:
        return ResourceFormat::RGBA32Uint;
    case nrd::Format::RGBA32_SINT:
        return ResourceFormat::RGBA32Int;
    case nrd::Format::RGBA32_SFLOAT:
        return ResourceFormat::RGBA32Float;
    case nrd::Format::R10_G10_B10_A2_UNORM:
        return ResourceFormat::RGB10A2Unorm;
    case nrd::Format::R10_G10_B10_A2_UINT:
        return ResourceFormat::RGB10A2Uint;
    case nrd::Format::R11_G11_B10_UFLOAT:
        return ResourceFormat::R11G11B10Float;
    case nrd::Format::R9_G9_B9_E5_UFLOAT:
        return ResourceFormat::RGB9E5Float;
    default:
        FALCOR_THROW("Unsupported NRD format.");
    }
}

static void copyMatrix(float* dstMatrix, const float4x4& srcMatrix)
{
    float4x4 col_major = transpose(srcMatrix);
    memcpy(dstMatrix, static_cast<const float*>(col_major.data()), sizeof(float4x4));
}

void NRDPass::createFalcorTextures(Falcor::ref<Falcor::Device> pDevice)
{
    mViewZTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::R32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mViewZTexture->setName("NRD_ViewZ");

    mMotionVectorTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RG32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mMotionVectorTexture->setName("NRD_MotionVectorTexture");

    mNormalLinearRoughnessTexture = mpDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mNormalLinearRoughnessTexture->setName("NRD_NormalLinearRoughness");

    mInputTexture = pDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );
    mInputTexture->setName("NRD_InputTexture");

    mOuputTexture = pDevice->createTexture2D(
        mWidth, mHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );
    mOuputTexture->setName("NRD_OutputTexture");
}

void NRDPass::initNRD()
{
    mpDenoiser = nullptr;

    const nrd::LibraryDesc& libraryDesc = nrd::GetLibraryDesc();

    const nrd::MethodDesc methods[] = {{nrd::Method::RELAX_DIFFUSE, uint16_t(mWidth), uint16_t(mHeight)}};

    nrd::DenoiserCreationDesc denoiserCreationDesc;
    denoiserCreationDesc.memoryAllocatorInterface.Allocate = nrdAllocate;
    denoiserCreationDesc.memoryAllocatorInterface.Reallocate = nrdReallocate;
    denoiserCreationDesc.memoryAllocatorInterface.Free = nrdFree;
    denoiserCreationDesc.requestedMethodNum = 1;
    denoiserCreationDesc.requestedMethods = methods;

    nrd::Result res = nrd::CreateDenoiser(denoiserCreationDesc, mpDenoiser);

    if (res != nrd::Result::SUCCESS)
        FALCOR_THROW("NRDPass: Failed to create NRD denoiser");

    createResources();
    createPipelines();
}

void NRDPass::createPipelines()
{
    mpPasses.clear();
    mpCachedProgramKernels.clear();
    mpCSOs.clear();
    mCBVSRVUAVdescriptorSetLayouts.clear();
    mpRootSignatures.clear();

    // Get denoiser desc for currently initialized denoiser implementation.
    const nrd::DenoiserDesc& denoiserDesc = nrd::GetDenoiserDesc(*mpDenoiser);

    // Create samplers descriptor layout and set.
    D3D12DescriptorSetLayout SamplersDescriptorSetLayout;

    for (uint32_t j = 0; j < denoiserDesc.staticSamplerNum; j++)
    {
        SamplersDescriptorSetLayout.addRange(ShaderResourceType::Sampler, denoiserDesc.staticSamplers[j].registerIndex, 1);
    }
    mpSamplersDescriptorSet =
        D3D12DescriptorSet::create(mpDevice, SamplersDescriptorSetLayout, D3D12DescriptorSetBindingUsage::ExplicitBind);

    // Set sampler descriptors right away.
    for (uint32_t j = 0; j < denoiserDesc.staticSamplerNum; j++)
    {
        mpSamplersDescriptorSet->setSampler(0, j, mpSamplers[j].get());
    }

    // Go over NRD passes and creating descriptor sets, root signatures and PSOs for each.
    for (uint32_t i = 0; i < denoiserDesc.pipelineNum; i++)
    {
        const nrd::PipelineDesc& nrdPipelineDesc = denoiserDesc.pipelines[i];
        const nrd::ComputeShader& nrdComputeShader = nrdPipelineDesc.computeShaderDXIL;

        // Initialize descriptor set.
        D3D12DescriptorSetLayout CBVSRVUAVdescriptorSetLayout;

        // Add constant buffer to descriptor set.
        CBVSRVUAVdescriptorSetLayout.addRange(ShaderResourceType::Cbv, denoiserDesc.constantBufferDesc.registerIndex, 1);

        for (uint32_t j = 0; j < nrdPipelineDesc.descriptorRangeNum; j++)
        {
            const nrd::DescriptorRangeDesc& nrdDescriptorRange = nrdPipelineDesc.descriptorRanges[j];

            ShaderResourceType descriptorType = nrdDescriptorRange.descriptorType == nrd::DescriptorType::TEXTURE
                                                    ? ShaderResourceType::TextureSrv
                                                    : ShaderResourceType::TextureUav;

            CBVSRVUAVdescriptorSetLayout.addRange(descriptorType, nrdDescriptorRange.baseRegisterIndex, nrdDescriptorRange.descriptorNum);
        }

        mCBVSRVUAVdescriptorSetLayouts.push_back(CBVSRVUAVdescriptorSetLayout);

        // Create root signature for the NRD pass.
        D3D12RootSignature::Desc rootSignatureDesc;
        rootSignatureDesc.addDescriptorSet(SamplersDescriptorSetLayout);
        rootSignatureDesc.addDescriptorSet(CBVSRVUAVdescriptorSetLayout);

        const D3D12RootSignature::Desc& desc = rootSignatureDesc;

        ref<D3D12RootSignature> pRootSig = D3D12RootSignature::create(mpDevice, desc);

        mpRootSignatures.push_back(pRootSig);

        // Create Compute PSO for the NRD pass.
        {
            std::string shaderFileName = "nrd/Shaders/Source/" + std::string(nrdPipelineDesc.shaderFileName) + ".hlsl";

            ProgramDesc programDesc;
            programDesc.addShaderLibrary(shaderFileName).csEntry(nrdPipelineDesc.shaderEntryPointName);
            programDesc.setCompilerFlags(SlangCompilerFlags::MatrixLayoutColumnMajor);
            // Disable warning 30056: non-short-circuiting `?:` operator is deprecated, use 'select' instead.
            programDesc.setCompilerArguments({"-Wno-30056"});
            DefineList defines;
            defines.add("NRD_COMPILER_DXC");
            defines.add("NRD_USE_OCT_NORMAL_ENCODING", "1");
            defines.add("NRD_USE_MATERIAL_ID", "0");
            ref<ComputePass> pPass = ComputePass::create(mpDevice, programDesc, defines);

            ref<Program> pProgram = pPass->getProgram();
            ref<const ProgramKernels> pProgramKernels = pProgram->getActiveVersion()->getKernels(mpDevice.get(), pPass->getVars().get());

            ComputeStateObjectDesc csoDesc;
            csoDesc.pProgramKernels = pProgramKernels;
            csoDesc.pD3D12RootSignatureOverride = pRootSig;

            ref<ComputeStateObject> pCSO = mpDevice->createComputeStateObject(csoDesc);

            mpPasses.push_back(pPass);
            mpCachedProgramKernels.push_back(pProgramKernels);
            mpCSOs.push_back(pCSO);
        }
    }
}

void NRDPass::createResources()
{
    // Destroy previously created resources.
    mpSamplers.clear();
    mpPermanentTextures.clear();
    mpTransientTextures.clear();

    const nrd::DenoiserDesc& denoiserDesc = nrd::GetDenoiserDesc(*mpDenoiser);
    const uint32_t poolSize = denoiserDesc.permanentPoolSize + denoiserDesc.transientPoolSize;

    // Create samplers.
    for (uint32_t i = 0; i < denoiserDesc.staticSamplerNum; i++)
    {
        const nrd::StaticSamplerDesc& nrdStaticsampler = denoiserDesc.staticSamplers[i];
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);

        if (nrdStaticsampler.sampler == nrd::Sampler::NEAREST_CLAMP || nrdStaticsampler.sampler == nrd::Sampler::LINEAR_CLAMP)
        {
            samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        }
        else
        {
            samplerDesc.setAddressingMode(TextureAddressingMode::Mirror, TextureAddressingMode::Mirror, TextureAddressingMode::Mirror);
        }

        if (nrdStaticsampler.sampler == nrd::Sampler::NEAREST_CLAMP || nrdStaticsampler.sampler == nrd::Sampler::NEAREST_MIRRORED_REPEAT)
        {
            samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
        }
        else
        {
            samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
        }

        mpSamplers.push_back(mpDevice->createSampler(samplerDesc));
    }

    // Texture pool.
    for (uint32_t i = 0; i < poolSize; i++)
    {
        const bool isPermanent = (i < denoiserDesc.permanentPoolSize);

        // Get texture desc.
        const nrd::TextureDesc& nrdTextureDesc =
            isPermanent ? denoiserDesc.permanentPool[i] : denoiserDesc.transientPool[i - denoiserDesc.permanentPoolSize];

        // Create texture.
        ResourceFormat textureFormat = getFalcorFormat(nrdTextureDesc.format);
        ref<Texture> pTexture = mpDevice->createTexture2D(
            nrdTextureDesc.width,
            nrdTextureDesc.height,
            textureFormat,
            1u,
            nrdTextureDesc.mipNum,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        if (isPermanent)
            mpPermanentTextures.push_back(pTexture);
        else
            mpTransientTextures.push_back(pTexture);
    }
}

void NRDPass::packNRD(Falcor::RenderContext* pRenderContext, uint32_t ReservoirIdx)
{
    FALCOR_PROFILE(pRenderContext, "DenoisingPass::packNRD");

    auto var = mpPackNRDPass->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["viewMat"] = transpose(mpScene->getCamera()->getViewMatrix());
    var["PerFrameCB"]["previousFrameViewProjMat"] = transpose(mPreviousFrameViewProjMat);
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["reservoirIndex"] = ReservoirIdx;

    var["gReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();

    var["gRadianceHit"] = mInputTexture;
    var["gNormalLinearRoughness"] = mNormalLinearRoughnessTexture;
    var["gViewZ"] = mViewZTexture;
    var["gMotionVector"] = mMotionVectorTexture;

    var["gPositionWs"] = GBufferSingleton::instance()->getCurrentPositionWsTexture();
    var["gAlbedo"] = GBufferSingleton::instance()->getAlbedoTexture();
    var["gNormalWs"] = GBufferSingleton::instance()->getCurrentNormalWsTexture();

    mpPackNRDPass->execute(pRenderContext, mWidth, mHeight);
}

void NRDPass::unpackNRD(Falcor::RenderContext* pRenderContext, uint32_t ReservoirIdx)
{
    FALCOR_PROFILE(pRenderContext, "DenoisingPass::unpackNRD");

    auto var = mpUnpackNRDPass->getRootVar();

    var["PerFrameCB"]["viewportDims"] = uint2(mWidth, mHeight);
    var["PerFrameCB"]["nbReservoirPerPixel"] = SceneSettingsSingleton::instance()->nbReservoirPerPixel;
    var["PerFrameCB"]["reservoirIndex"] = ReservoirIdx;

    var["gReservoirs"] = ReservoirManagerSingleton::instance()->getCurrentFrameReservoirBuffer();
    var["gNRDOuputTexture"] = mOuputTexture;

    mpUnpackNRDPass->execute(pRenderContext, mWidth, mHeight);
}

void NRDPass::render(Falcor::RenderContext* pRenderContext, uint32_t ReservoirIdx)
{
    NRD_ASSERT(pRenderContext == mpRenderContext);

    FALCOR_PROFILE(pRenderContext, "DenoisingPass::render");

    packNRD(pRenderContext, ReservoirIdx);
    dipatchNRD(pRenderContext);
    unpackNRD(pRenderContext, ReservoirIdx);

    mPreviousFrameViewMat = mpScene->getCamera()->getViewMatrix();
    mPreviousFrameProjMat = mpScene->getCamera()->getProjMatrix();
    mPreviousFrameViewProjMat = mpScene->getCamera()->getViewProjMatrix();

    ++mFrameIndex;
}

void NRDPass::populateCommonSettings(nrd::CommonSettings& settings)
{
    const auto& camera = mpScene->getCamera();

    const Falcor::float4x4 currProjMatrix = camera->getProjMatrix();
    copyMatrix(settings.viewToClipMatrix, currProjMatrix);
    copyMatrix(settings.viewToClipMatrixPrev, mPreviousFrameProjMat);

    const Falcor::float4x4 currViewMatrix = camera->getViewMatrix();
    copyMatrix(settings.worldToViewMatrix, currViewMatrix);
    copyMatrix(settings.worldToViewMatrixPrev, mPreviousFrameViewMat);

    settings.motionVectorScale[0] = 1.0f;
    settings.motionVectorScale[1] = 1.0f;

    settings.cameraJitter[0] = 0.0f;
    settings.cameraJitter[1] = 0.0f;

    settings.isMotionVectorInWorldSpace = false;

    settings.denoisingRange = 4.0f * mpScene->getSceneBounds().radius();

    settings.disocclusionThreshold = 0.01f;
    settings.debug = false;
    settings.frameIndex = mFrameIndex;

    settings.accumulationMode = mFrameIndex ? nrd::AccumulationMode::CONTINUE : nrd::AccumulationMode::RESTART;
}

void NRDPass::populateDenoiserSettings(nrd::RelaxDiffuseSettings& settings)
{
    settings.prepassBlurRadius = 16.0f;
    settings.diffuseMaxFastAccumulatedFrameNum = 2;
    settings.diffuseLobeAngleFraction = 0.8f;
    settings.disocclusionFixMaxRadius = 32.0f;
    settings.disocclusionFixNumFramesToFix = 4;
    settings.spatialVarianceEstimationHistoryThreshold = 4;
    settings.atrousIterationNum = 6;
    settings.depthThreshold = 0.02f;
}

void NRDPass::dipatchNRD(Falcor::RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "DenoisingPass::dipatchNRD");

    nrd::CommonSettings commonSettings = {};
    populateCommonSettings(commonSettings);

    nrd::RelaxDiffuseSettings denoiserSettings{};
    populateDenoiserSettings(denoiserSettings);
    nrd::SetMethodSettings(*mpDenoiser, nrd::Method::RELAX_DIFFUSE, static_cast<void*>(&denoiserSettings));

    const nrd::DispatchDesc* dispatchDescs = nullptr;
    uint32_t dispatchDescNum = 0;
    nrd::Result result = nrd::GetComputeDispatches(*mpDenoiser, commonSettings, dispatchDescs, dispatchDescNum);
    FALCOR_ASSERT(result == nrd::Result::SUCCESS);

    for (uint32_t i = 0; i < dispatchDescNum; i++)
    {
        const nrd::DispatchDesc& dispatchDesc = dispatchDescs[i];
        FALCOR_PROFILE(pRenderContext, dispatchDesc.name);
        dispatch(pRenderContext, dispatchDesc);
    }

    // Submit the existing command list and start a new one.
    pRenderContext->submit();
}

void NRDPass::dispatch(RenderContext* pRenderContext, const nrd::DispatchDesc& dispatchDesc)
{
    const nrd::DenoiserDesc& denoiserDesc = nrd::GetDenoiserDesc(*mpDenoiser);
    const nrd::PipelineDesc& pipelineDesc = denoiserDesc.pipelines[dispatchDesc.pipelineIndex];

    // Set root signature.
    mpRootSignatures[dispatchDesc.pipelineIndex]->bindForCompute(pRenderContext);

    // Upload constants.
    auto cbAllocation = mpDevice->getUploadHeap()->allocate(dispatchDesc.constantBufferDataSize, ResourceBindFlags::Constant);
    std::memcpy(cbAllocation.pData, dispatchDesc.constantBufferData, dispatchDesc.constantBufferDataSize);

    // Create descriptor set for the NRD pass.
    ref<D3D12DescriptorSet> CBVSRVUAVDescriptorSet = D3D12DescriptorSet::create(
        mpDevice, mCBVSRVUAVdescriptorSetLayouts[dispatchDesc.pipelineIndex], D3D12DescriptorSetBindingUsage::ExplicitBind
    );

    // Set CBV.
    mpCBV = D3D12ConstantBufferView::create(mpDevice, cbAllocation.getGpuAddress(), cbAllocation.size);
    CBVSRVUAVDescriptorSet->setCbv(0 /* NB: range #0 is CBV range */, denoiserDesc.constantBufferDesc.registerIndex, mpCBV.get());

    uint32_t resourceIndex = 0;
    for (uint32_t descriptorRangeIndex = 0; descriptorRangeIndex < pipelineDesc.descriptorRangeNum; descriptorRangeIndex++)
    {
        const nrd::DescriptorRangeDesc& nrdDescriptorRange = pipelineDesc.descriptorRanges[descriptorRangeIndex];

        for (uint32_t descriptorOffset = 0; descriptorOffset < nrdDescriptorRange.descriptorNum; descriptorOffset++)
        {
            FALCOR_ASSERT(resourceIndex < dispatchDesc.resourceNum);
            const nrd::Resource& resource = dispatchDesc.resources[resourceIndex];

            FALCOR_ASSERT(resource.stateNeeded == nrdDescriptorRange.descriptorType);

            ref<Texture> texture;

            switch (resource.type)
            {
            case nrd::ResourceType::IN_MV:
                texture = mMotionVectorTexture;
                break;
            case nrd::ResourceType::IN_NORMAL_ROUGHNESS:
                texture = mNormalLinearRoughnessTexture;
                break;
            case nrd::ResourceType::IN_VIEWZ:
                texture = mViewZTexture;
                break;
            case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:
                texture = mInputTexture;
                break;
            case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:
                texture = mOuputTexture;
                break;
            case nrd::ResourceType::TRANSIENT_POOL:
                texture = mpTransientTextures[resource.indexInPool];
                break;
            case nrd::ResourceType::PERMANENT_POOL:
                texture = mpPermanentTextures[resource.indexInPool];
                break;
            default:
                FALCOR_ASSERT(!"Unavailable resource type");
                break;
            }

            FALCOR_ASSERT(texture);

            // Set up resource barriers.
            Resource::State newState =
                resource.stateNeeded == nrd::DescriptorType::TEXTURE ? Resource::State::ShaderResource : Resource::State::UnorderedAccess;
            for (uint16_t mip = 0; mip < resource.mipNum; mip++)
            {
                const ResourceViewInfo viewInfo = ResourceViewInfo(resource.mipOffset + mip, 1, 0, 1);
                pRenderContext->resourceBarrier(texture.get(), newState, &viewInfo);
            }

            // Set the SRV and UAV descriptors.
            if (nrdDescriptorRange.descriptorType == nrd::DescriptorType::TEXTURE)
            {
                ref<ShaderResourceView> pSRV = texture->getSRV(resource.mipOffset, resource.mipNum, 0, 1);
                CBVSRVUAVDescriptorSet->setSrv(
                    descriptorRangeIndex + 1 /* NB: range #0 is CBV range */,
                    nrdDescriptorRange.baseRegisterIndex + descriptorOffset,
                    pSRV.get()
                );
            }
            else
            {
                ref<UnorderedAccessView> pUAV = texture->getUAV(resource.mipOffset, 0, 1);
                CBVSRVUAVDescriptorSet->setUav(
                    descriptorRangeIndex + 1 /* NB: range #0 is CBV range */,
                    nrdDescriptorRange.baseRegisterIndex + descriptorOffset,
                    pUAV.get()
                );
            }

            resourceIndex++;
        }
    }

    FALCOR_ASSERT(resourceIndex == dispatchDesc.resourceNum);

    // Set descriptor sets.
    mpSamplersDescriptorSet->bindForCompute(pRenderContext, mpRootSignatures[dispatchDesc.pipelineIndex].get(), 0);
    CBVSRVUAVDescriptorSet->bindForCompute(pRenderContext, mpRootSignatures[dispatchDesc.pipelineIndex].get(), 1);

    // Set pipeline state.
    ref<ComputePass> pPass = mpPasses[dispatchDesc.pipelineIndex];
    ref<Program> pProgram = pPass->getProgram();
    ref<const ProgramKernels> pProgramKernels = pProgram->getActiveVersion()->getKernels(mpDevice.get(), pPass->getVars().get());

    // Check if anything changed.
    bool newProgram = (pProgramKernels.get() != mpCachedProgramKernels[dispatchDesc.pipelineIndex].get());
    if (newProgram)
    {
        mpCachedProgramKernels[dispatchDesc.pipelineIndex] = pProgramKernels;

        ComputeStateObjectDesc desc;
        desc.pProgramKernels = pProgramKernels;
        desc.pD3D12RootSignatureOverride = mpRootSignatures[dispatchDesc.pipelineIndex];

        ref<ComputeStateObject> pCSO = mpDevice->createComputeStateObject(desc);
        mpCSOs[dispatchDesc.pipelineIndex] = pCSO;
    }
    ID3D12GraphicsCommandList* pCommandList =
        pRenderContext->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>();
    ID3D12PipelineState* pPipelineState = mpCSOs[dispatchDesc.pipelineIndex]->getNativeHandle().as<ID3D12PipelineState*>();

    pCommandList->SetPipelineState(pPipelineState);

    // Dispatch.
    pCommandList->Dispatch(dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);

    mpDevice->getUploadHeap()->release(cbAllocation);
}

DenoisingPass::DenoisingPass(
    Falcor::ref<Falcor::Device> pDevice,
    Falcor::RenderContext* pRenderContext,
    Falcor::ref<Falcor::Scene> pScene,
    uint32_t width,
    uint32_t height
)
{
    mNRDPass = new NRDPass(pDevice, pRenderContext, pScene, width, height);
}

DenoisingPass::~DenoisingPass()
{
    delete mNRDPass;
}

void DenoisingPass::render(Falcor::RenderContext* pRenderContext)
{
    for (uint32_t i = 0u; i < SceneSettingsSingleton::instance()->nbReservoirPerPixel; ++i)
        mNRDPass->render(pRenderContext, i);
}
} // namespace Restir

