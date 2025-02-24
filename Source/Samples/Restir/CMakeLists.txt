add_falcor_executable(Restir)

target_sources(Restir PRIVATE
	Denoising.h
    GBuffer.h
    RestirApp.h
    RIS.h
    TemporalFiltering.h    
    Visibility.h

	Denoising.cpp
    GBuffer.cpp
	RestirApp.cpp
	RIS.cpp
	TemporalFiltering.cpp
	Visibility.cpp

    GBuffer.slang
    RIS.slang
    TemporalFiltering.slang
    Visibility.slang
)

target_copy_shaders(Restir Samples/Restir)

target_source_group(Restir "Samples")
