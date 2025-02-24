/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Scene/VertexAttrib.slangh"

import Scene.Raster;
import Utils.Math.MathHelpers;
import GBufferHelpers;
import Rendering.Materials.TexLODHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

// GBuffer channels
struct GBufferPSOut
{
    float4 posW : SV_TARGET0;
    float4 normW : SV_TARGET1;
    float4 tangentW : SV_TARGET2;
    float4 faceNormalW : SV_TARGET3;
    float2 texC : SV_TARGET4;
    float4 texGrads : SV_TARGET5;
    float2 mvec : SV_TARGET6;
    uint4 mtlData : SV_TARGET7;
};

VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}

float2 computeMotionVector(const VSOut vsOut, const int2 ipos)
{
    // Current sample in pixel coords.
    float2 pixelPos = ipos + float2(0.5, 0.5);
    // Sample in previous frame in clip space coords, no jittering applied.
    float4 prevPosH = vsOut.prevPosH;
    // Remove camera jitter from motion vector
    return calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
}

[earlydepthstencil]
GBufferPSOut psMain(VSOut vsOut, uint triangleIndex: SV_PrimitiveID, float3 barycentrics: SV_Barycentrics)
{
    // Using vOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(vsOut.posH.xy);

    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    let lod = ImplicitLodTextureSampler();

    const float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir);

    uint hints = 0;
    // Create material instance and query its properties.
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = mi.getProperties(sd);

    const GBufferData gbuf = prepareGBufferData(sd, v, mi, bsdfProperties);

    const float4 texGrads = float4(ddx(sd.uv), ddy(sd.uv));
    const float2 mvec = computeMotionVector(vsOut, ipos);

    GBufferPSOut psOut = {};

    psOut.posW = gbuf.posW;
    psOut.normW = gbuf.normW;
    psOut.tangentW = gbuf.tangentW;
    psOut.faceNormalW = gbuf.faceNormalW;
    psOut.texC = gbuf.texC;
    psOut.texGrads = texGrads;
    psOut.mvec = mvec;
    psOut.mtlData = gbuf.mtlData;


    return psOut;
}
