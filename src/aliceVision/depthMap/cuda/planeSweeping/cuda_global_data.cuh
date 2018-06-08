// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "aliceVision/depthMap/cuda/commonStructures.hpp"

#include <cuda_runtime.h>

#include <map>
#include <vector>
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

    void                 allocScaledPictureArrays( int scales, int ncams, int width, int height );
    void                 freeScaledPictureArrays( );
    CudaArray<uchar4,2>* getScaledPictureArrayPtr( int scale, int cam );
    CudaArray<uchar4,2>& getScaledPictureArray( int scale, int cam );
    cudaTextureObject_t  getScaledPictureTex( int scale, int cam );

private:
    std::map<GaussianArrayIndex,GaussianArray*> _gaussian_arr_table;

    std::vector<CudaArray<uchar4, 2>*>          _scaled_picture_array;
    std::vector<cudaTextureObject_t>            _scaled_picture_tex;
    int                                         _scaled_picture_scales;
};

/*
 * We keep data in this array that is frequently allocated and freed, as well
 * as recomputed in the original code without a decent need.
 */
extern GlobalData global_data;

}; // namespace depthMap
}; // namespace aliceVision
