// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "aliceVision/depthMap/cuda/deviceCommon/device_matrix.cuh"
#include "aliceVision/depthMap/cuda/deviceCommon/device_color.cuh"

#define GRIFF_TEST

namespace aliceVision {
namespace depthMap {

__global__ void refine_selectPartOfDepthMapNearFPPlaneDepth_kernel(float* o0depthMap, int o0depthMap_p,
                                                                   float* o1depthMap, int o1depthMap_p,
                                                                   float* idepthMap, int idepthMap_p, int width,
                                                                   int height, float fpPlaneDepth,
                                                                   float fpPlaneDepthNext);

__global__ void refine_dilateDepthMap_kernel(float* depthMap, int depthMap_p, int width, int height, const float gammaC);

__global__ void refine_dilateFPPlaneDepthMapXpYp_kernel(float* fpPlaneDepthMap, int fpPlaneDepthMap_p, float* maskMap,
                                                        int maskMap_p, int width, int height, int xp, int yp,
                                                        float fpPlaneDepth);

__global__ void refine_convertFPPlaneDepthMapToDepthMap_kernel(float* depthMap, int depthMap_p, float* fpPlaneDepthMap,
                                                               int fpPlaneDepthMap_p, int width, int height);

__global__ void refine_computeDepthsMapFromDepthMap_kernel(float3* depthsMap, int depthsMap_p, float* depthMap,
                                                           int depthMap_p, int width, int height, bool moveByTcOrRc,
                                                           float step);

__global__ void refine_reprojTarTexLABByDepthsMap_kernel(float3* depthsMap, int depthsMap_p, uchar4* tex, int tex_p,
                                                         int width, int height, int id);

__global__ void refine_reprojTarTexLABByDepthMap_kernel(float* depthMap, int depthMap_p, uchar4* tex, int tex_p,
                                                        int width, int height);

__global__ void refine_reprojTarTexLABByDepthMapMovedByStep_kernel(float* depthMap, int depthMap_p, uchar4* tex,
                                                                   int tex_p, int width, int height, bool moveByTcOrRc,
                                                                   float step);

__global__ void refine_compYKNCCSimMap_kernel(float* osimMap, int osimMap_p, float* depthMap, int depthMap_p, int width,
                                              int height, int wsh, const float gammaC, const float gammaP);

__global__ void refine_compYKNCCSim_kernel(float3* osims, int osims_p, int id, float* depthMap, int depthMap_p,
                                           int width, int height, int wsh, const float gammaC, const float gammaP);

__global__ void refine_compYKNCCSimOptGammaC_kernel(float3* osims, int osims_p, int id, float* depthMap, int depthMap_p,
                                                    int width, int height, int wsh, const float gammaP);

__global__ void refine_computeBestDepthSimMaps_kernel(float* osim, int osim_p, float* odpt, int odpt_p, float3* isims,
                                                      int isims_p, float3* idpts, int idpts_p, int width, int height,
                                                      float simThr);

__global__ void refine_fuseThreeDepthSimMaps_kernel(float* osim, int osim_p, float* odpt, int odpt_p, float* isimLst,
                                                    int isimLst_p, float* idptLst, int idptLst_p, float* isimAct,
                                                    int isimAct_p, float* idptAct, int idptAct_p, int width, int height,
                                                    float simThr);

#ifdef GRIFF_TEST
__global__ void refine_compYKNCCSimMapPatch_kernel_A(
    const float* depthMap, int depthMap_p,
    int width, int height, int wsh, const float gammaC,
    const float gammaP, const float epipShift, const float tcStep,
    bool moveByTcOrRc, int xFrom, int imWidth, int imHeight,
    float3* lastThreeSimsMap, int lastThreeSimsMap_p, const int dimension );
#endif

#ifdef GRIFF_TEST
__global__ void refine_compUpdateYKNCCSimMapPatch_kernel(
    float* osimMap, int osimMap_p,
    float* odptMap, int odptMap_p,
    const float* depthMap, int depthMap_p, int width, int height,
    int wsh, const float gammaC, const float gammaP,
    const float epipShift,
    const int ntcsteps,
    bool moveByTcOrRc, int xFrom, int imWidth, int imHeight,
    float3* lastThreeSimsMap, int lastThreeSimsMap_p );
#else
__global__ void refine_compUpdateYKNCCSimMapPatch_kernel(
    float* osimMap, int osimMap_p,
    float* odptMap, int odptMap_p,
    const float* depthMap, int depthMap_p, int width, int height,
    int wsh, const float gammaC, const float gammaP,
    const float epipShift,
    const float tcStep,    // changing in loop
    int id,                // changing in loop
    bool moveByTcOrRc, int xFrom, int imWidth, int imHeight);
#endif

__global__ void refine_coputeDepthStepMap_kernel(float* depthStepMap, int depthStepMap_p, float* depthMap,
                                                 int depthMap_p, int width, int height, bool moveByTcOrRc);

__global__ void refine_compYKNCCDepthSimMapPatch_kernel(float2* oDepthSimMap, int oDepthSimMap_p, float* depthMap,
                                                        int depthMap_p, int width, int height, int wsh,
                                                        const float gammaC, const float gammaP, const float epipShift,
                                                        const float tcStep, bool moveByTcOrRc);

__global__ void refine_compYKNCCSimMapPatch_kernel(
    float* osimMap, int osimMap_p,
    const float* depthMap, int depthMap_p,
    int width, int height, int wsh, const float gammaC,
    const float gammaP, const float epipShift, const float tcStep,
    bool moveByTcOrRc, int xFrom, int imWidth, int imHeight);

__global__ void refine_compYKNCCSimMapPatchDMS_kernel(float* osimMap, int osimMap_p, float* depthMap, int depthMap_p,
                                                      int width, int height, int wsh, const float gammaC,
                                                      const float gammaP, const float epipShift,
                                                      const float depthMapShift);

__global__ void refine_setLastThreeSimsMap_kernel(float3* lastThreeSimsMap, int lastThreeSimsMap_p, float* simMap,
                                                  int simMap_p, int width, int height, int id);

__global__ void refine_computeDepthSimMapFromLastThreeSimsMap_kernel(float* osimMap, int osimMap_p, float* iodepthMap,
                                                                     int iodepthMap_p, float3* lastThreeSimsMap,
                                                                     int lastThreeSimsMap_p, int width, int height,
                                                                     bool moveByTcOrRc, int xFrom);

__global__ void refine_updateLastThreeSimsMap_kernel(float3* lastThreeSimsMap, int lastThreeSimsMap_p, float* simMap,
                                                     int simMap_p, int width, int height, int id);

__global__ void refine_updateBestStatMap_kernel(float4* bestStatMap, int bestStatMap_p, float3* lastThreeSimsMap,
                                                int lastThreeSimsMap_p, int width, int height, int id, int nids,
                                                float tcStepBefore, float tcStepAct);

__global__ void refine_computeDepthSimMapFromBestStatMap_kernel(float* simMap, int simMap_p, float* depthMap,
                                                                int depthMap_p, float4* bestStatMap, int bestStatMap_p,
                                                                int width, int height, bool moveByTcOrRc);

__global__ void refine_reprojTarTexLABByRcTcDepthsMap_kernel(uchar4* tex, int tex_p, float* rcDepthMap,
                                                             int rcDepthMap_p, int width, int height,
                                                             float depthMapShift);

#if 0
inline static __device__ double refine_convolveGaussSigma2(float* Im)
{
    double sum = 0.0;
    for(int yp = -2; yp <= +2; yp++)
    {
        for(int xp = -2; xp <= +2; xp++)
        {
            sum = sum + (double)Im[(xp + 2) * 5 + yp + 2] * (double)gauss5[yp + 2] * (double)gauss5[xp + 2];
        };
    };
    return sum;
}
#endif

__global__ void refine_compPhotoErr_kernel(float* osimMap, int osimMap_p, float* depthMap, int depthMap_p, int width,
                                           int height, double beta);

__global__ void refine_compPhotoErrStat_kernel(float* occMap, int occMap_p, float4* ostat1Map, int ostat1Map_p,
                                               float* depthMap, int depthMap_p, int width, int height, double beta);

__global__ void refine_compPhotoErrABG_kernel(float* osimMap, int osimMap_p, int width, int height);

#if 0
__device__ float2 ComputeSobelTarIm(int x, int y)
{
    unsigned char ul = 255.0f * tex2D(tTexU4, x - 1, y - 1).x; // upper left
    unsigned char um = 255.0f * tex2D(tTexU4, x + 0, y - 1).x; // upper middle
    unsigned char ur = 255.0f * tex2D(tTexU4, x + 1, y - 1).x; // upper right
    unsigned char ml = 255.0f * tex2D(tTexU4, x - 1, y + 0).x; // middle left
    unsigned char mm = 255.0f * tex2D(tTexU4, x + 0, y + 0).x; // middle (unused)
    unsigned char mr = 255.0f * tex2D(tTexU4, x + 1, y + 0).x; // middle right
    unsigned char ll = 255.0f * tex2D(tTexU4, x - 1, y + 1).x; // lower left
    unsigned char lm = 255.0f * tex2D(tTexU4, x + 0, y + 1).x; // lower middle
    unsigned char lr = 255.0f * tex2D(tTexU4, x + 1, y + 1).x; // lower right

    int Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
    int Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
    return make_float2((float)Vert, (float)Horz);
}

__device__ float2 DPIXTCDRC(const float3& P)
{
    float M3P = sg_s_tP[2] * P.x + sg_s_tP[5] * P.y + sg_s_tP[8] * P.z + sg_s_tP[11];
    float M3P2 = M3P * M3P;

    float m11 = ((sg_s_tP[0] * sg_s_tP[5] - sg_s_tP[2] * sg_s_tP[3]) * P.y +
                 (sg_s_tP[0] * sg_s_tP[8] - sg_s_tP[2] * sg_s_tP[6]) * P.z +
                 (sg_s_tP[0] * sg_s_tP[11] - sg_s_tP[2] * sg_s_tP[9])) /
                M3P2;
    float m12 = ((sg_s_tP[3] * sg_s_tP[2] - sg_s_tP[5] * sg_s_tP[0]) * P.x +
                 (sg_s_tP[3] * sg_s_tP[8] - sg_s_tP[5] * sg_s_tP[6]) * P.z +
                 (sg_s_tP[3] * sg_s_tP[11] - sg_s_tP[5] * sg_s_tP[9])) /
                M3P2;
    float m13 = ((sg_s_tP[6] * sg_s_tP[2] - sg_s_tP[8] * sg_s_tP[0]) * P.x +
                 (sg_s_tP[6] * sg_s_tP[5] - sg_s_tP[8] * sg_s_tP[3]) * P.y +
                 (sg_s_tP[6] * sg_s_tP[11] - sg_s_tP[8] * sg_s_tP[9])) /
                M3P2;

    float m21 = ((sg_s_tP[1] * sg_s_tP[5] - sg_s_tP[2] * sg_s_tP[4]) * P.y +
                 (sg_s_tP[1] * sg_s_tP[8] - sg_s_tP[2] * sg_s_tP[7]) * P.z +
                 (sg_s_tP[1] * sg_s_tP[11] - sg_s_tP[2] * sg_s_tP[10])) /
                M3P2;
    float m22 = ((sg_s_tP[4] * sg_s_tP[2] - sg_s_tP[5] * sg_s_tP[1]) * P.x +
                 (sg_s_tP[4] * sg_s_tP[8] - sg_s_tP[5] * sg_s_tP[7]) * P.z +
                 (sg_s_tP[4] * sg_s_tP[11] - sg_s_tP[5] * sg_s_tP[10])) /
                M3P2;
    float m23 = ((sg_s_tP[7] * sg_s_tP[2] - sg_s_tP[8] * sg_s_tP[1]) * P.x +
                 (sg_s_tP[7] * sg_s_tP[5] - sg_s_tP[8] * sg_s_tP[4]) * P.y +
                 (sg_s_tP[7] * sg_s_tP[11] - sg_s_tP[8] * sg_s_tP[10])) /
                M3P2;

    float3 _drc = P - sg_s_rC;

    float2 op;
    op.x = m11 * _drc.x + m12 * _drc.y + m13;
    op.y = m21 * _drc.x + m22 * _drc.y + m23;

    return op;
};
#endif

__global__ void refine_reprojTarSobelAndDPIXTCDRCRcTcDepthsMap_kernel(float4* tex, int tex_p, float* rcDepthMap,
                                                                      int rcDepthMap_p, int width, int height,
                                                                      float depthMapShift);

__global__ void refine_computeRcTcDepthMap_kernel(float* rcDepthMap, int rcDepthMap_p, int width, int height,
                                                  float pixSizeRatioThr);

} // namespace depthMap
} // namespace aliceVision
