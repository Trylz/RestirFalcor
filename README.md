# Restir Falcor

This repository host an implementation of the Restir algorithm inside Nvidia Falcor framework.
![restir_teaser](https://github.com/user-attachments/assets/e2757806-ed91-4eab-91fd-16b60cdad4f5)
Showcase: https://www.youtube.com/watch?v=jiP2zfzdDog

Here the original paper:  
https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf

# How to test

## Grab necessary assets
First grab the dragon buddha model has explained here:
https://github.com/Trylz/RestirFalcor/blob/master/TestScenes/DragonBuddha/README.txt  
It is necessary since git doesnt allow me to push file that are greater than 100 MB.

## Building prerequisites
- Windows 10 version 20H2 (October 2020 Update) or newer, OS build revision .789 or newer
- Visual Studio 2022 or newer
- Cmake
- [Windows 10 SDK (10.0.19041.0) for Windows 10, version 2004](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
- Cuda toolkit installed. Right now 12.6(https://developer.nvidia.com/cuda-12-6-0-download-archive) is hardcoded
**set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin")** in root **CMakeLists.txt**

- An Nvidia GPU since Cuda is used
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX
- NVIDIA driver 466.11 or newer

## Building Restir Falcor
Run the **setup_vs2022.bat** script after cloning this repository. The solution files are written to build/windows-vs2022 and the binary output is located in build/windows-vs2022/bin.

## Test scenes
Two test scenes are provided. The Arcade scene that ships with Falcor and the DragonBuddha scene.  
To switch scene simply change the **SCENE_NAME** define  inside RestirApp.cpp file  
0: Arcade scene  
1: Dragon buddha scene.


# Code overview
Restir source code is located here  
https://github.com/Trylz/RestirFalcor/tree/master/Source/Samples/Restir

## Application entry point
The application entry point is located in the RestirApp.cpp file

## Reservoirs
Reservoirs are managed by the reservoir manager. See ReservoirManager.cpp.

## Lights
Lights are managed by the light manager. See LightManager.cpp.  
This is where lights are created for each scene.  
Right now only spherical(so area) lights are supported.
![AeraLightsPNG](https://github.com/user-attachments/assets/092bba21-114f-438b-9f6b-09b36b451a47)
![AeraLights_CloseUp](https://github.com/user-attachments/assets/d50d26fb-47a8-40f8-bf72-5d98b735f511)



## Settings
Scenes specific tweaks are available stored in the *SceneSettings* struct.  

## Render passes
### RIS
Resampled Importance Sampling is performed as described in paper.

### Visibility test
For each reservoir its weight is set to zero if occluded. So raytracing is performed here.

### Temporal filtering 
We combine the current frame reservoirs with thoses of the previous frame

Here the dragon buddha scene with temporal filtering ON and OFF

ON:
https://youtu.be/8Q-eVLcEUPY

OFF:
https://youtu.be/iBL5fuxIujA

### Spatial filtering 
Combine each pixel reservoirs with thoses of its neighboords. This is disabled by default since it give me strange resulsts.  
#define SPATIAL_FILTERING 0

### Denoising

Denoising can be done using NRD or Optix denoiser  
#define DENOISING_USE_NRD 0  

Note that NRD is currently performed on the final image(same for Optix).
It is not recommended when i read the doc. So we may want to denoise the reservoirs instead. This is work in progress see: NRDDenoiserPass_MultipleNrd_WIP.cpp

# ISSUES
This is work in progress and some artefacts are still visible.

### Noise on the Arcade scene floor
![Arcade_Noise](https://github.com/user-attachments/assets/f9cee9fe-40d5-47cb-bc63-540f0a099dab)


### Flickering on the buddha of the dragon buddha scene when camera motion.
https://youtu.be/wGGd2VHuTtU  
Increasing the number of reservoir per pixels helps but it kills performance. The scene has three lights so theorically no more than 3 reservoirs should be necessary.
