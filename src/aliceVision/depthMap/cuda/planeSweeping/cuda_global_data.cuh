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

typedef std::pair<int,double> GaussianArrayIndex;

struct GaussianArray
{
    cudaArray*          arr;
    cudaTextureObject_t tex;

    void create( float delta, int radius );
};

typedef std::pair<int,int> PitchedMem_Tex_Index;

template<typename T,cudaTextureFilterMode fMode>
struct PitchedMem_LinearTexture
{
    CudaDeviceMemoryPitched<T,2>* mem;
    cudaTextureObject_t           tex;

    PitchedMem_LinearTexture( int w, int h )
    {
        mem = new CudaDeviceMemoryPitched<T,2>( CudaSize<2>( w, h ) );

        cudaTextureDesc      tex_desc;
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.readMode         = cudaReadModeNormalizedFloat;
        tex_desc.filterMode       = fMode;

        cudaResourceDesc res_desc;
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.desc         = cudaCreateChannelDesc<T>();
        res_desc.res.pitch2D.devPtr       = mem->getBuffer();
        res_desc.res.pitch2D.width        = mem->getSize()[0];
        res_desc.res.pitch2D.height       = mem->getSize()[1];
        res_desc.res.pitch2D.pitchInBytes = mem->getPitch();

        cudaCreateTextureObject( &tex,
                                 &res_desc,
                                 &tex_desc,
                                 0 );
    }

    ~PitchedMem_LinearTexture( )
    {
        cudaDestroyTextureObject( tex );
        delete mem;
    }
};

class GlobalData
{
    typedef unsigned char uchar;
public:

    ~GlobalData( );

    GaussianArray* getGaussianArray( float delta, int radius );

    void                 allocScaledPictureArrays( int scales, int ncams, int width, int height );
    void                 freeScaledPictureArrays( );
    CudaArray<uchar4,2>* getScaledPictureArrayPtr( int scale, int cam );
    CudaArray<uchar4,2>& getScaledPictureArray( int scale, int cam );
    cudaTextureObject_t  getScaledPictureTex( int scale, int cam );

    void                               allocPyramidArrays( int levels, int width, int height );
    void                               freePyramidArrays( );
    CudaDeviceMemoryPitched<uchar4,2>& getPyramidArray( int level );
    cudaTextureObject_t                getPyramidTex( int level );

    PitchedMem_LinearTexture<uchar4,cudaFilterModeLinear>*  getPitchedMemUchar4_LinearTexture( int width, int height );
    void                                                    putPitchedMemUchar4_LinearTexture( PitchedMem_LinearTexture<uchar4,cudaFilterModeLinear>* ptr );

    PitchedMem_LinearTexture<uchar,cudaFilterModeLinear>*   getPitchedMemUchar_LinearTexture( int width, int height );
    void                                                    putPitchedMemUchar_LinearTexture( PitchedMem_LinearTexture<uchar,cudaFilterModeLinear>* ptr );

private:
    std::map<GaussianArrayIndex,GaussianArray*> _gaussian_arr_table;

    std::vector<CudaArray<uchar4, 2>*>          _scaled_picture_array;
    std::vector<cudaTextureObject_t>            _scaled_picture_tex;
    int                                         _scaled_picture_scales;

    std::vector<CudaDeviceMemoryPitched<uchar4, 2>*> _pyramid_array;
    std::vector<cudaTextureObject_t>                 _pyramid_tex;
    int                                              _pyramid_levels;

    std::multimap<PitchedMem_Tex_Index,PitchedMem_LinearTexture<uchar4,cudaFilterModeLinear>*> _pitched_mem_uchar4_linear_tex_cache;
    std::multimap<PitchedMem_Tex_Index,PitchedMem_LinearTexture<uchar,cudaFilterModeLinear>*>  _pitched_mem_uchar_linear_tex_cache;
};

/*
 * We keep data in this array that is frequently allocated and freed, as well
 * as recomputed in the original code without a decent need.
 */
extern GlobalData global_data;

}; // namespace depthMap
}; // namespace aliceVision
