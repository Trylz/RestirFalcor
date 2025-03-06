# Restir Falcor

This repository host my implementation of the Restir algorithm inside Nvidia Falcor framework.
![restir_teaser](https://github.com/user-attachments/assets/7bc00960-7177-47bd-b570-993f8274dc3b)

# How to test

## Grab necessary assets
First grab the dragon buddha moded has explained here:
https://github.com/Trylz/FalcorRestir/tree/master/TestScenes/DragonBuddha
Necessary since git doesnt allow me to push file that are greater than 100 MB.

## Building prerequisites
- Windows 10 version 20H2 (October 2020 Update) or newer, OS build revision .789 or newer
- Visual Studio 2022 or newer
- Cmake
- [Windows 10 SDK (10.0.19041.0) for Windows 10, version 2004](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
- An Nvidia GPU since i use Optix
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX
- NVIDIA driver 466.11 or newer

## Building Restir Falcor
Run the setup_vs2022.bat script after cloning this repository. The solution files are written to build/windows-vs2022 and the binary output is located in build/windows-vs2022/bin.


## Test scenes
Two test scenes are provided. The Arcade scene that ships with Falcor and the DragonBuddha scene.  
To switch scene simply change the SCENE_NAME define  
0: Arcade scene  
1: Dragon baddha scene.


# Code overview


## Application entry point
The appplication entry point is located in the RestirApp.cpp file

## Reservoirs
Reservoirs are managed by the reservoir manager. See ReservoirManager.cpp.

## Lights
Lights are managed by the light manager. See LightManager.cpp.  
This is where lights are created for each scene.  
Right now only spherical so area lights are supported.

![AeraLightsPNG](https://github.com/user-attachments/assets/46d7662b-b1f5-4cbb-85e4-1747567e166b)

![AeraLights_CloseUp](https://github.com/user-attachments/assets/4d3ced30-e7da-4a77-aa9c-6cd644fe4f55)

## Settings
Scenes specific tweaks are available stored in the SceneSettings struct.  

## Render passes
### RIS
RIS is computed using as described in paper

### Visibility test
For each reservoir its weight is set to xzero if occluded

### Temporal filtering 
We combine the current frame reservoirs with thoses of th eprevious frame

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

Note that NRD is currently performaned on the final image(same for Optix).
It is not recommended when i read the doc. So we ewant to denoise the reservoirs insteead. This is work in progress see: NRDDenoiserPass_MultipleNrd_WIP.cpp

# ISSUES
This is work in progress and some artefacts are still visible.

### Noise on the Arcade scene floor
![Arcade_Floor](https://github.com/user-attachments/assets/113ace92-15a0-4d0a-af7c-81bb1ae930df)


### Flickering on the buddha of the dragon buddha scene when camera motion.
