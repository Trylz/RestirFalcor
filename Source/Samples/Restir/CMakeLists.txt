add_falcor_executable(Restir)

target_sources(Restir PRIVATE
	Denoising.h
    GBuffer.h
    LightManager.h
    RestirApp.h
    RIS.h
    Singleton.h
    TemporalFiltering.h    
    Visibility.h

	Denoising.cpp
    FloatRandomNumberGenerator.h
    GBuffer.cpp
    LightManager.cpp
	RestirApp.cpp
	RIS.cpp
	TemporalFiltering.cpp
	Visibility.cpp

    GBuffer.slang
    Reservoir.slang
    RIS.slang
    TemporalFiltering.slang
    Visibility.slang
)

target_copy_shaders(Restir Samples/Restir)

target_source_group(Restir "Samples")
