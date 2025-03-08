if(NOT FALCOR_HAS_CUDA)
    return()
endif()

add_falcor_executable(Restir)

target_sources(Restir PRIVATE
    ApplicationPathsManager.h
    NRDDenoiserPass.h
    GBuffer.h
    LightManager.h
    OptixDenoiserPass.h
    ReservoirManager.h
    RestirApp.h
    RISPass.h
    SceneName.h
    SceneSettings.h
    Singleton.h
    ShadingPass.h
    SpatialFilteringPass.h
    TemporalFilteringPass.h    
    VisibilityPass.h

    NRDDenoiserPass.cpp
    FloatRandomNumberGenerator.h
    GBuffer.cpp
    LightManager.cpp
    OptixDenoiserPass.cpp
    ReservoirManager.cpp
    RestirApp.cpp
    RISPass.cpp
    ShadingPass.cpp
    SpatialFilteringPass.cpp

    TemporalFilteringPass.cpp
    VisibilityPass.cpp

    NRDDenoiserPass_PackNRD.slang
    NRDDenoiserPass_UnpackNRD.slang

    OptixDenoiserPass_ConvertBufToTex.slang
    OptixDenoiserPass_ComputeMotionVectors.slang
    OptixDenoiserPass_ConvertNormalsToBuf.slang
    OptixDenoiserPass_ConvertTexToBuf.slang

    GBuffer.slang
    RISPass.slang
    ShadingPass.slang
    TemporalFilteringPass.slang
    VisibilityPass.slang

    SpatialFilteringPass.slang

    NRD.slangh
    NRDEncoding.slangh
    Light.slangh
    Reservoir.slangh
)

set_target_properties(Restir PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_copy_shaders(Restir Samples/Restir)

target_link_libraries(Restir PRIVATE optix)

target_source_group(Restir "Samples")
