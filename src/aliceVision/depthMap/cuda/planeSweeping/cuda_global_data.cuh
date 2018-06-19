// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// #include "aliceVision/depthMap/cuda/commonStructures.hpp"

// #include "aliceVision/depthMap/cuda/planeSweeping/device_code.cuh"
// #include "aliceVision/depthMap/cuda/planeSweeping/device_code_refine.cuh"
// #include "aliceVision/depthMap/cuda/planeSweeping/device_code_volume.cuh"
// #include "aliceVision/depthMap/cuda/planeSweeping/device_code_fuse.cuh"

// #include "aliceVision/depthMap/cuda/deviceCommon/device_color.cuh"
// #include "aliceVision/depthMap/cuda/deviceCommon/device_patch_es.cuh"
// #include "aliceVision/depthMap/cuda/deviceCommon/device_eig33.cuh"
// #include "aliceVision/depthMap/cuda/deviceCommon/device_global.cuh"

// #include <math_constants.h>
// #include <iostream>

#include <cuda_runtime.h>

#include <map>
#include <algorithm>

namespace aliceVision {
namespace depthMap {

struct GaussianArray
{
    cudaArray*          arr;
    cudaTextureObject_t tex;

    void create( float delta, int radius );
};

class GlobalData
{
    typedef std::pair<int,double> GaussianArrayIndex;
public:

    ~GlobalData( );

    GaussianArray* getGaussianArray( float delta, int radius );

private:
    std::map<GaussianArrayIndex,GaussianArray*> _gaussian_arr_table;
};

/*
 * We keep data in this array that is frequently allocated and freed, as well
 * as recomputed in the original code without a decent need.
 */
extern GlobalData global_data;

}; // namespace depthMap
}; // namespace aliceVision
