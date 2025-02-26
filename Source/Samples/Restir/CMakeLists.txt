add_falcor_executable(Restir)

target_sources(Restir PRIVATE
	DenoisingPass.h
    GBuffer.h
    LightManager.h
    ReservoirManager.h
    RestirApp.h
    RISPass.h
    Singleton.h
    TemporalFilteringPass.h    
    VisibilityPass.h

	DenoisingPass.cpp
    FloatRandomNumberGenerator.h
    GBuffer.cpp
    LightManager.cpp
    ReservoirManager.cpp
	RestirApp.cpp
	RISPass.cpp
	TemporalFilteringPass.cpp
	VisibilityPass.cpp

    GBuffer.slang
    RISPass.slang
    TemporalFilteringPass.slang
    VisibilityPass.slang

    BRDF.slangh
    Light.slangh
    Reservoir.slangh
)

target_copy_shaders(Restir Samples/Restir)

target_source_group(Restir "Samples")
