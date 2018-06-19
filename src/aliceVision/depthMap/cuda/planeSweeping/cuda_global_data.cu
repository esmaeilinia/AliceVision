// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "aliceVision/depthMap/cuda/planeSweeping/cuda_global_data.cuh"

// #include "aliceVision/depthMap/cuda/commonStructures.hpp"

// #include "aliceVision/depthMap/cuda/planeSweeping/device_code.cuh"
// #include "aliceVision/depthMap/cuda/planeSweeping/device_code_refine.cuh"
// #include "aliceVision/depthMap/cuda/planeSweeping/device_code_volume.cuh"
// #include "aliceVision/depthMap/cuda/planeSweeping/device_code_fuse.cuh"

#include "aliceVision/depthMap/cuda/deviceCommon/device_color.cuh"
// #include "aliceVision/depthMap/cuda/deviceCommon/device_patch_es.cuh"
// #include "aliceVision/depthMap/cuda/deviceCommon/device_eig33.cuh"
// #include "aliceVision/depthMap/cuda/deviceCommon/device_global.cuh"

// #include <math_constants.h>
#include <iostream>
// #include <map>

// #include <algorithm>

// Macro for checking cuda errors
#define CHECK_CUDA_ERROR()                                                    \
    if(cudaError_t err = cudaGetLastError())                                  \
                                                                              \
{                                                                             \
        fprintf(stderr, "\n\nCUDAError: %s\n", cudaGetErrorString(err));      \
        fprintf(stderr, "  file:       %s\n", __FILE__);                      \
        fprintf(stderr, "  function:   %s\n", __FUNCTION__);                  \
        fprintf(stderr, "  line:       %d\n\n", __LINE__);                    \
                                                                              \
}


namespace aliceVision {
namespace depthMap {

/*
 * We keep data in this array that is frequently allocated and freed, as well
 * as recomputed in the original code without a decent need.
 *
 * The code is not capable of dealing with multiple GPUs yet (on multiple GPUs,
 * multiple allocations are probably required).
 */
GlobalData global_data;

// texture<float, cudaTextureType1D, cudaReadModeElementType> gaussianTex;

void GaussianArray::create( float delta, int radius )
{
    std::cerr << "Computing Gaussian table for radius " << radius << " and delta " << delta << std::endl;

    int size = 2 * radius + 1;

    float* d_gaussian;
    cudaMalloc((void**)&d_gaussian, (2 * radius + 1) * sizeof(float));
    CHECK_CUDA_ERROR();

    // generate gaussian array
    generateGaussian_kernel<<<1, size>>>(d_gaussian, delta, radius);
    cudaThreadSynchronize();

    // create cuda array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&arr, &channelDesc, size, 1);
    CHECK_CUDA_ERROR();
    cudaMemcpyToArray(arr, 0, 0, d_gaussian, size * sizeof(float), cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR();
    cudaFree(d_gaussian);
    CHECK_CUDA_ERROR();

    cudaResourceDesc res_desc;
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = arr;

    cudaTextureDesc      tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.readMode         = cudaReadModeElementType; // read as float
    tex_desc.filterMode       = cudaFilterModePoint; // apparently default for references
    // tex_desc.filterMode       = cudaFilterModeLinear; // no interpolation

    cudaCreateTextureObject( &tex, &res_desc, &tex_desc, 0 );
    CHECK_CUDA_ERROR();
}

GlobalData::~GlobalData( )
{
    auto end = _gaussian_arr_table.end();
    for( auto it=_gaussian_arr_table.begin(); it!=end;it++ )
    {
        // cudaDestroyTexture( it->second->tex );
        cudaFreeArray( it->second->arr );
    }
}

GaussianArray* GlobalData::getGaussianArray( float delta, int radius )
{
    auto it = _gaussian_arr_table.find( GaussianArrayIndex(radius,delta) );
    if( it != _gaussian_arr_table.end() )
    {
        return it->second;
    }

    GaussianArray* a = new GaussianArray;
    a->create( delta, radius );

    _gaussian_arr_table.insert( std::pair<GaussianArrayIndex,GaussianArray*>( GaussianArrayIndex(radius,delta), a ) );

    return a;
}

}; // namespace depthMap
}; // namespace aliceVision

