add_falcor_executable(Restir)

target_sources(Restir PRIVATE
	Denoising.h
    GBuffer.h
    LightManager.h
    ReservoirManager.h
    RestirApp.h
    RIS.h
    Singleton.h
    TemporalFiltering.h    
    Visibility.h

    BRDF.slangh
	Denoising.cpp
    FloatRandomNumberGenerator.h
    GBuffer.cpp
    LightManager.cpp
    ReservoirManager.cpp
	RestirApp.cpp
	RIS.cpp
	TemporalFiltering.cpp
	Visibility.cpp

    GBuffer.slang
    RIS.slang
    TemporalFiltering.slang
    Visibility.slang

    Light.slangh
    Reservoir.slangh
)

target_copy_shaders(Restir Samples/Restir)

target_source_group(Restir "Samples")
