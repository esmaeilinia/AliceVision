// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "aliceVision/depthMap/cuda/planeSweeping/cuda_global_data.cuh"

#include "aliceVision/depthMap/cuda/deviceCommon/device_color.cuh"

#include <iostream>

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

void GlobalData::allocScaledPictureArrays( int scales, int ncams, int width, int height )
{
    _scaled_picture_scales = scales;

    _scaled_picture_array.resize( scales * ncams );
    _scaled_picture_tex  .resize( scales * ncams );

    cudaResourceDesc res_desc;
    res_desc.resType = cudaResourceTypeArray;

    cudaTextureDesc      tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.readMode         = cudaReadModeNormalizedFloat;
    tex_desc.filterMode       = cudaFilterModeLinear;

    for( int c=0; c<ncams; c++ )
    {
        for( int s=0; s<scales; s++ )
        {
            int w = width / (s + 1);
            int h = height / (s + 1);
            _scaled_picture_array[ c * scales + s ] = new CudaArray<uchar4, 2>( CudaSize<2>( w, h ) );

            res_desc.res.array.array = _scaled_picture_array[ c * scales + s ]->getArray();

            cudaCreateTextureObject( &_scaled_picture_tex[ c * scales + s ],
                                     &res_desc,
                                     &tex_desc,
                                     0 );
        }
    }

}

void GlobalData::freeScaledPictureArrays( )
{
    _scaled_picture_scales = 0;

    for( CudaArray<uchar4,2>* ptr : _scaled_picture_array )
    {
        delete ptr;
    }

    _scaled_picture_array.clear();

    for( cudaTextureObject_t& obj : _scaled_picture_tex )
    {
        cudaDestroyTextureObject( obj );
    }

    _scaled_picture_tex.clear();
}

CudaArray<uchar4,2>* GlobalData::getScaledPictureArrayPtr( int scale, int cam )
{
    return _scaled_picture_array[ cam * _scaled_picture_scales + scale ];
}

CudaArray<uchar4,2>& GlobalData::getScaledPictureArray( int scale, int cam )
{
    return *_scaled_picture_array[ cam * _scaled_picture_scales + scale ];
}

cudaTextureObject_t GlobalData::getScaledPictureTex( int scale, int cam )
{
    return _scaled_picture_tex[ cam * _scaled_picture_scales + scale ];
}

void GlobalData::allocPyramidArrays( int levels, int w, int h )
{
    _pyramid_levels = levels;

    _pyramid_array.resize( levels );
    _pyramid_tex  .resize( levels );

    cudaTextureDesc      tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.readMode         = cudaReadModeNormalizedFloat;
    tex_desc.filterMode       = cudaFilterModeLinear;

    for( int lvl=0; lvl<levels; lvl++ )
    {
        _pyramid_array[ lvl ] = new CudaDeviceMemoryPitched<uchar4, 2>( CudaSize<2>( w, h ) );

        cudaResourceDesc res_desc;
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.desc         = cudaCreateChannelDesc<uchar4>();
        res_desc.res.pitch2D.devPtr       = _pyramid_array[ lvl ]->getBuffer();
        res_desc.res.pitch2D.width        = _pyramid_array[ lvl ]->getSize()[0];
        res_desc.res.pitch2D.height       = _pyramid_array[ lvl ]->getSize()[1];
        res_desc.res.pitch2D.pitchInBytes = _pyramid_array[ lvl ]->getPitch();

        cudaCreateTextureObject( &_pyramid_tex[ lvl ],
                                 &res_desc,
                                 &tex_desc,
                                 0 );
        w /= 2;
        h /= 2;
    }
}

void GlobalData::freePyramidArrays( )
{
    _pyramid_levels = 0;

    for( CudaDeviceMemoryPitched<uchar4,2>* ptr : _pyramid_array )
    {
        delete ptr;
    }

    _pyramid_array.clear();

    for( cudaTextureObject_t& obj : _pyramid_tex )
    {
        cudaDestroyTextureObject( obj );
    }

    _pyramid_tex.clear();
}

CudaDeviceMemoryPitched<uchar4,2>& GlobalData::getPyramidArray( int level )
{
    return *_pyramid_array[ level ];
}

cudaTextureObject_t GlobalData::getPyramidTex( int level )
{
    return _pyramid_tex[ level ];
}

PitchedMem_LinearTexture<uchar4>* GlobalData::getPitchedMemUchar4_LinearTexture( int width, int height )
{
    auto it = _pitched_mem_uchar4_linear_tex_cache.find( PitchedMem_Tex_Index( width, height ) );
    if( it == _pitched_mem_uchar4_linear_tex_cache.end() )
    {
        std::cerr << "Allocate pitched mem uchar4 with linear tex " << width << "X" << height << std::endl;
        PitchedMem_LinearTexture<uchar4>* ptr = new PitchedMem_LinearTexture<uchar4>( width, height );
        return ptr;
    }
    else
    {
        std::cerr << "Getting pitched mem uchar4 with linear tex " << width << "X" << height << std::endl;
        PitchedMem_LinearTexture<uchar4>* ptr = it->second;
        _pitched_mem_uchar4_linear_tex_cache.erase( it );
        return ptr;
    }
}

void GlobalData::putPitchedMemUchar4_LinearTexture( PitchedMem_LinearTexture<uchar4>* ptr )
{
    int width  = ptr->mem->getSize()[0];
    int height = ptr->mem->getSize()[1];
    std::cerr << "Putting pitched mem uchar4 with linear tex " << width << "X" << height << std::endl;
    PitchedMem_Tex_Index idx( width, height );
    _pitched_mem_uchar4_linear_tex_cache.insert(
        std::pair<PitchedMem_Tex_Index,PitchedMem_LinearTexture<uchar4>*>(
            idx, ptr ) );
}

}; // namespace depthMap
}; // namespace aliceVision

