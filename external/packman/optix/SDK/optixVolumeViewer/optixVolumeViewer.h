//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "volume.h"

#include <optix.h>

#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/GridHandle.h"

#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Record.h>
#include <sutil/Aabb.h>
#include <sutil/Matrix.h>

#include <cuda/Light.h>

#include <string>


// The Volume struct ties together the NanoVDB host representation
// (NanoVDB Grid) and the device-buffer containing the sparse volume
// representation (NanoVDB Tree). In addition to the Tree, the Grid
// also contains an affine transform relating index space (i.e. voxel
// indices) to world-space. 
// 
struct Volume
{
    nanovdb::GridHandle<> handle;
	CUdeviceptr d_volume = 0;
};

void loadVolume( Volume& volume, const std::string& filename );
void cleanupVolume( Volume& volume );
void createGrid( Volume& volume, std::string filename, std::string gridname );
void getOptixTransform( const Volume& volume, float transform[] );
sutil::Aabb worldAabb( const Volume& volume );

// The VolumeAccel struct contains a volume's geometric representation for
// Optix: a traversalbe handle, and the (compacted) GAS device-buffer.
struct VolumeAccel
{
	OptixTraversableHandle handle = 0;
	CUdeviceptr            d_buffer = 0;
};

void buildVolumeAccel( VolumeAccel& accel, const Volume& volume, const OptixDeviceContext& context );
void cleanupVolumeAccel( VolumeAccel& accel );


struct Plane
{
    sutil::Matrix4x4 transform;
    
    static const unsigned int num_indices          = 6;
    const unsigned int        indices[num_indices] = {0, 1, 3, 1, 2, 3};
    CUdeviceptr               d_indices            = 0;

    static const unsigned int num_positions            = 4;
    float3                    positions[num_positions] = {};
    CUdeviceptr               d_positions              = 0;

    MaterialData::Lambert material;

    sutil::Aabb aabb;
};

void createPlane( Plane& plane, const sutil::Aabb& aabb );
void cleanupPlane( Plane& plane );


struct PlaneAccel
{
    OptixTraversableHandle handle   = 0;
    CUdeviceptr            d_buffer = 0;
};

void buildPlaneAccel( PlaneAccel& plane_accel, const Plane& plane, const OptixDeviceContext& context );
void cleanupPlaneAccel( PlaneAccel& plane_accel );


// Cube is for more testing of volume-solid interactions, ignore for review
// as this will be taken out for release.
struct Cube
{
    sutil::Matrix4x4 transform;

    static const unsigned int num_indices          = 6 * 2 * 3;
    const unsigned int        indices[num_indices] = {0, 1, 2, 0, 2, 3,
        0, 3, 5, 0, 5, 4,
        3, 2, 6, 3, 6, 5,
        0, 4, 7, 0, 7, 1,
        1, 7, 6, 1, 6, 2,
        4, 5, 6, 4, 6, 7};
    CUdeviceptr               d_indices            = 0;

	static const unsigned int num_positions            = 8;
    float3                    positions[num_positions] = {};
    CUdeviceptr               d_positions              = 0;

    MaterialData::Lambert material;
    
    sutil::Aabb aabb;
};

void createCube( Cube& cube, const sutil::Aabb& aabb );
void cleanupCube( Cube& cube);


struct CubeAccel
{
    OptixTraversableHandle handle   = 0;
    CUdeviceptr            d_buffer = 0;
};

void buildCubeAccel( CubeAccel& cube_accel, const Cube& cube, const OptixDeviceContext& context );
void cleanupCubeAccel( CubeAccel& cube_accel );


struct ProgramGroups
{
    OptixProgramGroup raygen           = 0;

    OptixProgramGroup miss_radiance    = 0;
    OptixProgramGroup miss_occlusion   = 0;
     
    OptixProgramGroup mesh_radiance    = 0;
    OptixProgramGroup mesh_occlusion   = 0;
    
    OptixProgramGroup volume_radiance  = 0;
    OptixProgramGroup volume_occlusion = 0;
};

void createProgramGroups( ProgramGroups& program_groups, 
    const OptixModule& module, 
    const OptixDeviceContext& context );
void cleanupProgramGroups( ProgramGroups& program_groups );


struct Params
{
    LaunchParams  params   = {};
    LaunchParams* d_params = 0;
};

void initLaunchParams( Params& launch_params, const OptixTraversableHandle& handle, const sutil::Aabb& aabb );
void cleanupLaunchParams( Params& launch_params );


struct IAS
{
    OptixTraversableHandle handle          = 0;
    CUdeviceptr            d_buffer        = 0;
    OptixAccelBufferSizes  buffer_sizes    = {};
    OptixBuildInput        build_input     = {};
    CUdeviceptr            d_instances     = 0;
    CUdeviceptr            d_update_buffer = 0;
};

void buildIAS( IAS& ias, int rayTypeCount,     
    const Volume& volume, const VolumeAccel& volume_accel, 
	const Plane& plane, const PlaneAccel& plane_accel,
    const Cube& cube, const CubeAccel& cube_accel,
    const OptixDeviceContext& context );
void updateIAS( IAS& ias, const OptixDeviceContext& context );
void cleanupIAS( IAS& ias );


typedef sutil::Record<HitGroupData> HitGroupRecord;
