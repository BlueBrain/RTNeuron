//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2016 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#include "fragments.h"

#define CUSTOM_SORT

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math_constants.h>

#include "cub/cub/device/device_scan.cuh"

#include <cassert>
#include <stdexcept>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace cuda
{

/*
  CUDA kernels
*/

texture<uint32_t, cudaTextureType2D, cudaReadModeElementType> headsRef;

#define COMPARE_AND_SWAP(i, j, k, v) \
    if (k[i] > k[j])             \
    {                            \
        const float tk = k[j];   \
        const uint32_t tc = v[j];\
        k[j] = k[i];             \
        k[i] = tk;               \
        v[j] = v[i];             \
        v[i] = tc;               \
    }

__device__ void blockCompare(const unsigned int index,
                             const unsigned int maxStride,
                             float* depths, uint32_t* colors)
{
    bool ascending = index & maxStride; // maxStride must be a power of 2.
    for (unsigned int stride = maxStride; stride > 0; stride >>= 1)
    {
        /* The assignment below yields the sequences:
           0, 2, 4, 6, 8, 10, 12, 14, ...
           0, 1, 4, 5, 8, 9, 12, 13, ...
           0, 1, 2, 3, 8, 9, 10, 11, ...
        */
        unsigned int i = 2 * index - (index & (stride - 1));
        if (!ascending)
            i += stride;
        const unsigned int j = i + (ascending ? stride : -stride);
        COMPARE_AND_SWAP(i, j, depths, colors)
        __syncthreads();
    }
}

__device__ void bitonicMerge(const unsigned int index,
                             const unsigned int maxStride,
                             float* depths, uint32_t* colors)
{
    for (unsigned int stride = maxStride; stride > 0; stride >>= 1)
    {
        const unsigned int i = 2 * index - (index & (stride - 1));
        const unsigned int j = i + stride;
        COMPARE_AND_SWAP(i, j, depths, colors)
        __syncthreads();
    }
}

template<typename Key, typename Value>
__device__ __forceinline__ void swapKeyValue(Key& k, Value& v,
                                             const uint32_t mask,
                                             const bool dir)
{
    Key peerK = __shfl_xor(k, mask);
    Value peerV = __shfl_xor(v, mask);
    if (dir == 1 ? k < peerK : k > peerK)
    {
        k = peerK;
        v = peerV;
    }
}

template<typename T>
__device__ __forceinline__ void swap(T& x, T& y)
{
    const T t = x;
    x = y;
    y = t;
}

template<uint32_t bit>
__device__ __forceinline__ bool testBit(const uint32_t x)
{
    const uint32_t mask = 1 << bit;
    return (x & mask) != 0;
}

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ bool bitonicSort32(
    const uint32_t* offsets, const size_t lists,
    float* depths, int* colors,
    float& depth, int& color, uint32_t& offset, int& count)
{
    /* colors needs to be int to get the code compiling in CUDA 6.0, because
       __shfl_xor lacks the proper overload in that version. */

    const int list = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (list >= lists)
        return false;
    offset = __ldg(offsets + list);
    count = __ldg(offsets + list + 1) - offset;
    if (count > 32)
        return false;

    if (count == 1)
        return true;

    const int laneid = threadIdx.x & 0x0000001f;
    depth = laneid < count ? depths[offset + laneid] : 1;
    color = laneid < count ? colors[offset + laneid] : 0xffffffff;

    swapKeyValue(depth, color, 0x01, testBit<1>(laneid) ^ testBit<0>(laneid));

    if (count <= 2)
        return true;

    swapKeyValue(depth, color, 0x02, testBit<2>(laneid) ^ testBit<1>(laneid));
    swapKeyValue(depth, color, 0x01, testBit<2>(laneid) ^ testBit<0>(laneid));

    if (count <= 4)
        return true;

    swapKeyValue(depth, color, 0x04, testBit<3>(laneid) ^ testBit<2>(laneid));
    swapKeyValue(depth, color, 0x02, testBit<3>(laneid) ^ testBit<1>(laneid));
    swapKeyValue(depth, color, 0x01, testBit<3>(laneid) ^ testBit<0>(laneid));

    if (count <= 8)
        return true;

    swapKeyValue(depth, color, 0x08, testBit<4>(laneid) ^ testBit<3>(laneid));
    swapKeyValue(depth, color, 0x04, testBit<4>(laneid) ^ testBit<2>(laneid));
    swapKeyValue(depth, color, 0x02, testBit<4>(laneid) ^ testBit<1>(laneid));
    swapKeyValue(depth, color, 0x01, testBit<4>(laneid) ^ testBit<0>(laneid));

    if (count <= 16)
        return true;

    swapKeyValue(depth, color, 0x10, testBit<4>(laneid));
    swapKeyValue(depth, color, 0x08, testBit<3>(laneid));
    swapKeyValue(depth, color, 0x04, testBit<2>(laneid));
    swapKeyValue(depth, color, 0x02, testBit<1>(laneid));
    swapKeyValue(depth, color, 0x01, testBit<0>(laneid));

    return true;
}
#endif

/* This kernel operates differently depending on the target architectures.
   For compute capability < 3.5 is sorts two lists per warp with blocks of 8
   warps. For compute capability >= 3.5 it sorts one lists per warp and there
   are no restrictions on the block size.  */
__global__ void bitonicSort32(const uint32_t* offsets, const size_t lists,
                              float* depths, uint32_t* colors)
{
#if __CUDA_ARCH__ < 350
    /* This kernel processes two lists per warp with blocks of 8 warps. */

    #define COPY_TO_SHARED_MEM(index, offset)  \
    if (index < counts)                        \
    {                                          \
        depthsSh[list][index] = depths[offset];\
        colorsSh[list][index] = colors[offset];\
    }                                          \
    else                                       \
        depthsSh[list][index] = CUDART_INF_F;

    #define COPY_TO_GLOBAL_MEM(index, offset)  \
    if (index < counts)                        \
    {                                          \
        depths[offset] = depthsSh[list][index];\
        colors[offset] = colorsSh[list][index];\
    }

    __shared__ float depthsSh[16][32];
    __shared__ uint32_t colorsSh[16][32];

    const unsigned int list = (threadIdx.x >> 4) + threadIdx.y * 2;
    const unsigned int index = threadIdx.x & ((32 >> 1) - 1);
    const unsigned int globalIdx =
        (blockIdx.x + gridDim.x * blockIdx.y) * 16 + list;

    if (globalIdx >= lists)
        return;

    uint32_t offset = offsets[globalIdx];
    const uint32_t counts = offsets[globalIdx + 1] - offset;
    if (counts < 2 || counts > 32)
        return;

    const unsigned int index2 = index * 2;
    offset += index2;
    COPY_TO_SHARED_MEM(index2, offset);
    COPY_TO_SHARED_MEM(index2 + 1, offset + 1);

    /* Code loosely based on the bitonicsSort.cu source code from the
       NVidia SDK example for sortingNetworks. */
    for (unsigned int stride = 1; stride < 32 / 2; stride <<= 1)
        blockCompare(index, stride, depthsSh[list], colorsSh[list]);
    bitonicMerge(index, 32 / 2, depthsSh[list], colorsSh[list]);

    COPY_TO_GLOBAL_MEM(index2, offset);
    COPY_TO_GLOBAL_MEM(index2 + 1, offset + 1);

    #undef COPY_TO_GLOBAL_MEM
    #undef COPY_TO_SHARED_MEM
#else
    int* icolors = (int*)colors;
    float depth;
    int color;
    int count;
    uint32_t offset;
    if (!bitonicSort32(offsets, lists, depths, icolors,
                       depth, color, offset, count) || count == 1)
        return;

    const int laneid = threadIdx.x & 0x0000001f;
    if (laneid < count)
    {
        depths[offset + laneid] = depth;
        icolors[offset + laneid] = color;
    }
#endif
}

__global__ void bitonicSort32WithThreshold(
    const uint32_t* offsets, const size_t lists,
    float* depths, uint32_t* colors,
    uint32_t* outCounts, const float alphaThreshold)
{
#if __CUDA_ARCH__ >= 350
    int* icolors = (int*)colors;

    float depth;
    int color;
    int count;
    uint32_t offset;

    if (!bitonicSort32(offsets, lists, depths, icolors,
                       depth, color, offset, count))
        return;

    const int laneid = threadIdx.x & 0x0000001f;
    const int list = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (count == 1)
    {
        if (laneid == 0)
            outCounts[list] = 1;
        return;
    }

    /* Finding out the last fragment that is visible due to alpha
       accumulation. */

    /* Warp wide reduction of the transparency.
       Each thread first computes t = (1 - a), a reduction follows and each
       thread i computes a value t_i = t_i * t_i-1 * ... * t_0. */
    float transparency = 1 - (uint32_t(color) >> 24) / 255.0;

    for (unsigned int i = 1; i < 32; i <<= 1)
    {
        const float incoming = __shfl_up(transparency, i);
        if (laneid >= i)
            transparency *= incoming;
    }

    /* Each thread compares its t_i to the minimum requested and a warp ballot
       proceeds. Each thread counts the number of 1's using the intrinsic
       popc. The new list length is this number + 1. */
    count = min(count, __popc(__ballot(1 - transparency < alphaThreshold)) + 1);

    if (laneid < count)
    {
        depths[offset + laneid] = depth;
        icolors[offset + laneid] = color;
    }
    if (laneid == 0)
        outCounts[list] = count;
#endif
}

/**
   Shuffles the values of the two input variables between the threads of a warp
   in the following way:

   Let x be an array of values of size #warp * 2 that holds the values of each
   thread t pair of variables in the following way:
   - v1 = x[t]
   - v2 = x[t + 32]
   At the output, each thead will contain:
   - v1 = x[t * 2]
   - v2 = x[t * 2 + 1]

   Using a warp of size as an example, if the input is the following:
           t0 t1 t2 t3      t0 t1 t2 t3
       v1: 0  1  2  3   v2: 4  5  6  7
   the output is:
           t0 t1 t2 t3      t0 t1 t2 t3
       v1: 0  2  4  6   v2: 1  2  5  7
*/
template<typename T>
__device__ __forceinline__ void shufflePairs(T& v1, T& v2, const int lane)
{
    /* Using a warp of size 4 as an example an a single variable,
       this sequence of operations does the following:
                        var 1     var 2
         initial state: 0 1 2 3   4 5 6 7
            shuffle v1: 0 2 1 3   4 5 6 7
            shuffle v2: 0 2 1 3   4 6 5 7
       exchange halves: 0 2 4 6   1 3 5 7 */
    v1 = __shfl(v1, lane * 2 + (lane > 15));
    v2 = __shfl(v2, lane * 2 + (lane > 15));
    T t = lane < 16 ? v2 : v1;
    t = __shfl(t, lane + 16);
    if (lane < 16)
        v2 = t;
    else
        v1 = t;
}

/* Only works for compute capability >= 3.5 */
#if __CUDA_ARCH__ >= 350
__device__ __forceinline__ bool bitonicSort64(
    const uint32_t* offsets, const size_t lists,
    float* depths, int* colors,
    float& depth1, float& depth2, int& color1, int& color2,
    uint32_t& offset, int& count)
{
    const int list = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (list >= lists)
        return false;
    offset = __ldg(offsets + list);
    count = __ldg(offsets + list + 1) - offset;
    if (count > 64 || count < 33)
        return false;

    /* Reading the 64 input depths and colors (or assigning an spurious value
       if out of the list range). Reads are coalesced to maximize bandwith. */
    const int lane = threadIdx.x & 0x0000001f;
    const int block2 = lane + 32;
    depth1 = lane < count ? __ldg(depths + offset + lane) : 1;
    color1 = lane < count ? __ldg(colors + offset + lane) : 0xffffffff;
    depth2 = block2 < count ? __ldg(depths + offset + block2) : 1;
    color2 = block2 < count? __ldg(colors + offset + block2) : 0xffffffff;

    /* Now we shuffle the data so each thread has two consecutive positions of
       the input data. */
    shufflePairs(depth1, depth2, lane);
    shufflePairs(color1, color2, lane);

    /* Sorting network. Note that for operations of stride 1 no interthread
       communication is needed as each thread holds two consecutive values. */
    const int lane2 = lane * 2;
    #define STRIDE1_EXCHANGE(step) \
        if (depth1 < depth2 == testBit<step>(lane2) ^ testBit<0>(lane2)) \
            { swap(depth1, depth2); swap(color1, color2); }
    #define SWAP_KEY_VALUES(mask, dir) \
        swapKeyValue(depth1, color1, mask, dir); \
        swapKeyValue(depth2, color2, mask, dir);

    STRIDE1_EXCHANGE(1)

    SWAP_KEY_VALUES(0x01, testBit<1>(lane) ^ testBit<0>(lane));
    STRIDE1_EXCHANGE(2)

    SWAP_KEY_VALUES(0x02, testBit<2>(lane) ^ testBit<1>(lane));
    SWAP_KEY_VALUES(0x01, testBit<2>(lane) ^ testBit<0>(lane));
    STRIDE1_EXCHANGE(3)

    SWAP_KEY_VALUES(0x04, testBit<3>(lane) ^ testBit<2>(lane));
    SWAP_KEY_VALUES(0x02, testBit<3>(lane) ^ testBit<1>(lane));
    SWAP_KEY_VALUES(0x01, testBit<3>(lane) ^ testBit<0>(lane));
    STRIDE1_EXCHANGE(4)

    SWAP_KEY_VALUES(0x08, testBit<4>(lane) ^ testBit<3>(lane));
    SWAP_KEY_VALUES(0x04, testBit<4>(lane) ^ testBit<2>(lane));
    SWAP_KEY_VALUES(0x02, testBit<4>(lane) ^ testBit<1>(lane));
    SWAP_KEY_VALUES(0x01, testBit<4>(lane) ^ testBit<0>(lane));
    STRIDE1_EXCHANGE(5)

    SWAP_KEY_VALUES(0x10, testBit<4>(lane));
    SWAP_KEY_VALUES(0x08, testBit<3>(lane));
    SWAP_KEY_VALUES(0x04, testBit<2>(lane));
    SWAP_KEY_VALUES(0x02, testBit<1>(lane));
    SWAP_KEY_VALUES(0x01, testBit<0>(lane));
    if (depth1 < depth2 == testBit<0>(lane2))
    {
        swap(depth1, depth2);
        swap(color1, color2);
    }
    #undef STRIDE1_EXCHANGE
    #undef SWAP_KEY_VALUES

    return true;
}
#endif

__global__ void bitonicSort64(const uint32_t* offsets, const size_t lists,
                              float* depths, uint32_t* colors)
{
#if __CUDA_ARCH__ >= 350
    int* icolors = (int*)colors;
    float depth1, depth2;
    int color1, color2;
    int count;
    uint32_t offset;
    if (!bitonicSort64(offsets, lists, depths, icolors,
                       depth1, depth2, color1, color2, offset, count))
        return;

    const int lane = threadIdx.x & 0x0000001f;
    const int lane2 = lane * 2;

    if (lane2 < count)
    {
        depths[offset + lane2] = depth1;
        icolors[offset + lane2] = color1;
    }
    if (lane2 + 1 < count)
    {
        depths[offset + lane2 + 1] = depth2;
        icolors[offset + lane2 + 1] = color2;
    }
#endif
}

__global__ void bitonicSort64WithThreshold(
    const uint32_t* offsets, const size_t lists,
    float* depths, uint32_t* colors,
    uint32_t* outCounts, const float alphaThreshold)
{
#if __CUDA_ARCH__ >= 350
    int* icolors = (int*)colors;
    float depth1, depth2;
    int color1, color2;
    int count;
    uint32_t offset;
    if (!bitonicSort64(offsets, lists, depths, icolors,
                       depth1, depth2, color1, color2, offset, count))
        return;

    const int lane = threadIdx.x & 0x0000001f;
    const int lane2 = lane * 2;
    const int list = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    /* Finding out the last fragment that is visible due to alpha
       accumulation. */

    /* Warp wide reduction of the transparency.
       Each thread first computes t = (1 - a), a reduction follows and each
       thread i computes a value t_i = t_i * t_i-1 * ... * t_0. */
    float transparency = (1 - (uint32_t(color1) >> 24) / 255.0) *
                         (1 - (uint32_t(color2) >> 24) / 255.0);

    for (unsigned int i = 1; i < 32; i <<= 1)
    {
        const float incoming = __shfl_up(transparency, i);
        if (lane >= i)
            transparency *= incoming;
    }
    /* Each thread compares its t_i to the minimum requested and a warp ballot
       proceeds. Each thread counts the number of 1's using the intrinsic
       popc. The new list length is (this number + 1) * 2. The multiplication
       is due to the fact each thread processes two fragments. */
    count = min(count,
                (__popc(__ballot(1 - transparency < alphaThreshold)) + 1) * 2);

    if (lane2 < count)
    {
        depths[offset + lane2] = depth1;
        icolors[offset + lane2] = color1;
    }
    if (lane2 + 1 < count)
    {
        depths[offset + lane2 + 1] = depth2;
        icolors[offset + lane2 + 1] = color2;
    }
    if (lane == 0)
        outCounts[list] = count;
#endif
}

#define PARENT(index) ((index - 1) / 2)
#define LEFT(index) (index * 2 + 1)
#define RIGHT(index) (index * 2 + 2)

template<typename T>
__forceinline__ __device__ void swap(T* v, unsigned int i, unsigned int j)
{
    const T t = v[i];
    v[i] = v[j];
    v[j] = t;
}

/** Rearranges a list of indices that dereference an array of values in a given
    range to ensure that the heap property holds in that range. */
template<typename Value>
__device__ void fixHeapDown(const uint32_t start, const uint32_t end,
                            uint32_t* indices, const Value* values)
{
#if __CUDA_ARCH__ < 350
#  define __ldg *
#endif
    uint32_t root = start;
    uint32_t left = LEFT(root);
    uint32_t right = RIGHT(root);
    while (left <= end)
    {
        /* Find which of the elements at the root and its children is the
           biggest */
        const Value leftV = __ldg(values + indices[left]);
        const Value rootV = __ldg(values + indices[root]);
        const Value rightV =
            right <= end ? __ldg(values + indices[right]) : leftV;

        if (rightV > leftV && rightV > rootV)
        {
            swap(indices, right, root);
            root = right;
        }
        else if (leftV > rootV)
        {
            swap(indices, left, root);
            root = left;
        }
        else
            return;
        left = LEFT(root);
        right = RIGHT(root);
    }
}

/** Creates a heap structure in a list of indices that is sorting in descending
    order (largest element first) based on the values pointed by the indices.. */
template<typename Value>
__device__ void heapify(uint32_t* indices, const Value* values,
                        const uint32_t size)
{
    for (uint32_t start = PARENT(size - 1); start > 0; --start)
        fixHeapDown(start, size - 1, indices, values);
    fixHeapDown(0, size - 1, indices, values);
}


template <int MIN_LENGTH, int MAX_LENGTH>
__device__ __forceinline__ uint32_t _heapSort(
    const uint32_t* offsets, const size_t lists,
    float*& depths, uint32_t*& colors, uint32_t indices[MAX_LENGTH])

{
#if __CUDA_ARCH__ < 350
#  define __ldg *
#endif
    const size_t index =
        (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x + threadIdx.x;

    if (index >= lists)
        return 0;

    const uint32_t offset = __ldg(offsets + index);
    const uint32_t count = __ldg(offsets + index + 1) - offset;
    if (count < MIN_LENGTH || count > MAX_LENGTH)
        return 0;

    depths += offset;
    colors += offset;

    for (uint32_t i = 0; i != count; ++i)
        indices[i] = i;

    heapify(indices, depths, count);
    for (uint32_t end = count - 1; end > 0; --end)
    {
        swap(indices, end, 0);
        fixHeapDown(0, end - 1, indices, depths);
    }
    return count;
}

template <int MIN_LENGTH, int MAX_LENGTH>
__global__ void heapSort(const uint32_t* offsets, const size_t lists,
                         float* depths, uint32_t* colors)
{
    uint32_t indices[MAX_LENGTH];
    uint32_t count = _heapSort<MIN_LENGTH, MAX_LENGTH>(
        offsets, lists, depths, colors, indices);
    if (!count)
        return;

    {
        float tmp[MAX_LENGTH];
        for (uint32_t i = 0; i != count; ++i)
            tmp[i] = __ldg(depths + i);
        for (uint32_t i = 0; i != count; ++i)
            depths[i] = tmp[indices[i]];
    }
    {
        uint32_t tmp[MAX_LENGTH];
        for (uint32_t i = 0; i != count; ++i)
            tmp[i] = __ldg(colors + i);
        for (uint32_t i = 0; i != count; ++i)
            colors[i] = tmp[indices[i]];
    }
}

template <int MIN_LENGTH, int MAX_LENGTH>
__global__ void heapSortWithThreshold(
    const uint32_t* offsets, const size_t lists,
    float* depths, uint32_t* colors,
    uint32_t* outCounts, const float alphaThreshold)
{
    uint32_t indices[MAX_LENGTH];
    uint32_t count = _heapSort<MIN_LENGTH, MAX_LENGTH>(
        offsets, lists, depths, colors, indices);
    if (!count)
        return;

    const size_t index =
        (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x + threadIdx.x;

    float transparency = 1;
    {
        uint32_t tmp[MAX_LENGTH];
        for (uint32_t i = 0; i != count; ++i)
        {
            const uint32_t color = __ldg(colors + indices[i]);
            tmp[i] = color;
            transparency *= 1 - (color >> 24) / 255.0;
            if (transparency <= 1 - alphaThreshold)
            {
                count = i + 1;
                break;
            }
        }
        for (uint32_t i = 0; i != count; ++i)
            colors[i] = tmp[i];
    }
    outCounts[index] = count;
    {
        float tmp[MAX_LENGTH];
        for (uint32_t i = 0; i != count; ++i)
            tmp[i] = __ldg(depths + indices[i]);
        for (uint32_t i = 0; i != count; ++i)
            depths[i] = tmp[i];
    }
}

/*
  Compacts the entangled fragment lists produced osgTransparency separating
  the dephs and rgba tuples in two different buffers.
*/
__global__ void compact(
    const size_t width, const size_t height,
    const size_t xOffset, const size_t yOffset,
    const uint32_t* offsets, const uint32_t* fragments,
    uint32_t* depths, uint32_t* colors)
{
#if __CUDA_ARCH__ < 350
#  define __ldg *
#endif
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const size_t index = y * width + x;
    const uint32_t offset = __ldg(offsets + index);
    const uint32_t count = __ldg(offsets + index + 1) - offset;
    if (count == 0)
        return;

    uint32_t next = tex2D(headsRef, xOffset + x, yOffset + y);

    depths += offset;
    colors += offset;
    for (uint32_t i = 0; i != count; ++i)
    {
        const uint32_t index = next * 3;
        next = __ldg(fragments + index);
        /* This following code peforms better than directly assining the values
           to the destination arrays. */
        const uint32_t depth = __ldg(fragments + index + 1);
        const uint32_t color = __ldg(fragments + index + 2);
        *(depths++) = depth;
        *(colors++) = color;
    }
}

/*
  Compacts the entangled fragment lists produced osgTransparency separating
  the dephs and rgba tuples in two different buffers.

  Taking advantage that alpha acummulation is commutative, the iteration of
  a fragment list may stop early if the alpha value is found to accumulate
  over a predefined threshold. If the end of the list was not reached, a depth
  value of 1 is written in the fragment after the last one.
*/
__global__ void recompact(
    const uint32_t* iOffsets, const uint32_t* iDepths, const uint32_t* iColors,
    uint32_t* oOffsets, uint32_t* oDepths, uint32_t* oColors,
    size_t pixels)
{
#if __CUDA_ARCH__ < 350
#  define __ldg *
#endif
    const int threadsPerPixel = 8;

    const size_t index =
                    ((blockIdx.x * blockDim.x) + threadIdx.x) / threadsPerPixel;
    if (index >= pixels)
        return;

    const int lane = threadIdx.x & (threadsPerPixel - 1);

    const uint32_t iOffset = __ldg(iOffsets + index);
    const uint32_t oOffset = __ldg(oOffsets + index);
    const uint32_t count = __ldg(oOffsets + index + 1) - oOffset;

    for (uint32_t i = 0; i < count; i += threadsPerPixel)
    {
        const size_t k = i + lane;
        if (k >= count)
            continue;
        const uint32_t depth = __ldg(iDepths + iOffset + k);
        const uint32_t color = __ldg(iColors + iOffset + k);
        oDepths[oOffset + k] = depth;
        oColors[oOffset + k] = color;
    }
}

template<size_t MAX_BUFFERS>
__global__ void mergeAndBlend(
    const float* depths[], const uint32_t* colors[], const uint32_t* offsets[],
    const unsigned int buffers, const uint32_t background,
    const size_t pixels, uint32_t* output, const float alphaCutoffThreshold)
{
#if __CUDA_ARCH__ < 350
#  define __ldg *
#endif
    const size_t index =
        (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x + threadIdx.x;
    if (index >= pixels)
        return;

    assert(buffers <= MAX_BUFFERS);

    float depthQueue[MAX_BUFFERS];
    uint32_t counts[MAX_BUFFERS];
    uint32_t positions[MAX_BUFFERS];

    int total = 0;
    for (unsigned int i = 0; i < buffers && i < MAX_BUFFERS; ++i)
    {
        const uint32_t offset = __ldg(offsets[i] + index);
        positions[i] = offset;
        const uint32_t count = __ldg(offsets[i] + index + 1) - offset;
        counts[i] = count;
        total += count;
    }
    const float inv255 = 1 / 255.0;

    for (unsigned int i = 0; i < MAX_BUFFERS; ++i)
        depthQueue[i] = 0;
    for (unsigned int i = 0; i < buffers; ++i)
        if (counts[i] > 0)
            depthQueue[i] = 1 - __ldg(depths[i] + positions[i]);

    float red = 0;
    float green = 0;
    float blue = 0;
    float alpha = 0;

    for (int i = 0; i < total; ++i)
    {
        int k = 0;
        for (int i = 0; i < MAX_BUFFERS; ++i)
            if (depthQueue[i] > depthQueue[k])
                k = i;

        const uint32_t raw = __ldg(colors[k] + positions[k]);
        if (--counts[k] == 0)
            depthQueue[k] = 0;
        else
            depthQueue[k] = 1 - __ldg(depths[k] + ++positions[k]);

        const float a = (raw >> 24) * (1 - alpha) * inv255;
        alpha += a;
        const float f = a * inv255;
        red += (raw & 0xff) * f;
        green += (raw >> 8 & 0xff) * f;
        blue += (raw >> 16 & 0xff) * f;

        if (alpha >= alphaCutoffThreshold)
            break;
    }

    /* Blending with the background color. */
    float br = (background & 0xff) * inv255;
    float bg = ((background >> 8) & 0xff) * inv255;
    float bb = ((background >> 16) & 0xff) * inv255;
    float ba = (background >> 24) * inv255;

    const float f = ba * (1 - alpha);
    const uint32_t rgba = min((int)round((red + br * f) * 255), 255) |
                          min((int)round((green + bg * f) * 255), 255) << 8 |
                          min((int)round((blue + bb * f) * 255), 255) << 16 |
                          min((int)round((alpha + f) * 255), 255) << 24;

    output[index] = rgba;
}

/*
  Invokation wrappers
*/

cudaError_t fragmentCountsToOffsets(
    const size_t width, const size_t height, const size_t x, const size_t y,
    cudaArray_t counts, uint32_t*& offsets, void*& tmp, size_t& tmpSize)
{
    /* Allocating the offsets pointer if not given externally, otherwise assume
       that it has the correct size already.
       The size of the buffer with offsets is one unit more because we want
       it to start at 0 and contain the total size at the end. */
    const size_t pixelCount = width * height;
    const size_t size = sizeof(uint32_t) * (pixelCount + 1);
    cudaError_t error;
    if (!offsets)
    {
        error = cudaMalloc(&offsets, size);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
    }

    /* We cannot use CUB's scan on a 2D array, so we need to copy the counts
       to a separate buffer, which will also be used as the output. */
    error = cudaMemcpy2DFromArrayAsync(
        offsets, width * 4, counts, x, y,
        width * 4, height, cudaMemcpyDeviceToDevice, 0);
    if (error != cudaSuccess)
    {
        cudaFree(offsets);
        offsets = 0;
        throw std::runtime_error(
            std::string("RTNeuron: error copying counts texture: ") +
            cudaGetErrorString(error));
    }

    size_t sizeNeeded = 0;
    cub::DeviceScan::ExclusiveSum(0, sizeNeeded, offsets, offsets,
                                  int(pixelCount) + 1);
    if (sizeNeeded > tmpSize || tmp == 0)
    {
        error = cudaMalloc(&tmp, sizeNeeded);
        tmpSize = sizeNeeded;
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
    }
    return cub::DeviceScan::ExclusiveSum(tmp, tmpSize, offsets, offsets,
                                         int(pixelCount) + 1);
}

void* allocateFragmentBuffer(const size_t pixelCount,
                             const uint32_t* offsets, size_t& size)
{
    /* Find out the size */
    uint32_t items;
    cudaMemcpy(&items, offsets + pixelCount, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
    size = items * (sizeof(uint32_t) + sizeof(float));

    if (!items)
        return 0;

    /* Allocating the space for the compacted fragment lists. The buffer for
       compacted lists contains the depths and colors in separate regions. */
    void* buffer = 0;
    const cudaError_t error = cudaMalloc(&buffer, size);
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("RTNeuron: error allocating device memory: ") +
            cudaGetErrorString(error));

    return buffer;
}

cudaError_t compactAndSortFragmentLists(
    const size_t width, const size_t height, const size_t x, const size_t y,
    cudaArray_t heads, const uint32_t* fragments, uint32_t* offsets,
    void*& output, size_t& size, float alphaThreshold, void*& tmp)
{
    cudaError_t error = compactFragmentLists(
        width, height, x, y, offsets, heads, fragments, output, size);
    const size_t items = size / (sizeof(uint32_t) + sizeof(float));

    if (error != cudaSuccess)
        goto error;

    if (size == 0)
        return cudaSuccess;

    if (alphaThreshold == 0)
        error = sortFragmentLists(output, items, width * height, offsets);
    else
        error = sortFragmentLists(output, items, width * height, offsets,
                                  alphaThreshold, tmp);

    if (error != cudaSuccess)
        goto error;

    return cudaSuccess;

error:
    if (output)
        cudaFree(output);
    output = 0;
    size = 0;
    return error;
}

cudaError_t compactAndSortFragmentLists(
    const size_t width, const size_t height, const size_t x, const size_t y,
    cudaArray_t heads, const uint32_t* fragments, uint32_t* offsets,
    void*& output, size_t& size)
{
    void* tmp = 0;
    return compactAndSortFragmentLists(width, height, x, y, heads, fragments,
                                       offsets, output, size, 0.f, tmp);
}

cudaError_t mergeAndBlendFragments(
    const std::vector<const float*>& depths,
    const std::vector<const uint32_t*>& colors,
    const std::vector<const uint32_t*>& offsets,
    const uint32_t background, const size_t pixels, void*& output,
    const float alphaThreshold)
{
    cudaFuncSetCacheConfig(mergeAndBlend<2>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mergeAndBlend<4>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mergeAndBlend<8>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mergeAndBlend<16>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(mergeAndBlend<32>, cudaFuncCachePreferL1);

    if (output == 0)
    {
        cudaError_t error = cudaMalloc(&output, pixels * 4);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
    }

    /* Copying the pointer to the fragment and offset buffers to the device. */
    const void** devPointers = 0;
    const size_t count = depths.size();
    assert(depths.size() == colors.size());
    assert(depths.size() == offsets.size());
    cudaError_t error = cudaMalloc(&devPointers, sizeof(uint32_t*) * count * 3);
    if (error != cudaSuccess)
    {
        cudaFree(output);
        output = 0;
        throw std::runtime_error(
            std::string("RTNeuron: error allocating device memory: ") +
            cudaGetErrorString(error));
    }

    const void** pointers = new const void*[count * 3];
    std::copy(depths.begin(), depths.end(), (const float**)pointers);
    std::copy(colors.begin(), colors.end(), (const uint32_t**)pointers + count);
    std::copy(
        offsets.begin(), offsets.end(), (const uint32_t**)pointers + count * 2);
    cudaMemcpy(devPointers, pointers, sizeof(uint32_t*) * count * 3,
               cudaMemcpyHostToDevice);
    delete[] pointers;

    /* Invoking the kernel.
       This has been fine tuned for the K20. The device occupancy with this
       number of threads per block is 25% maximum, but due to thread divergence,
       increasing the block size decreases the final performance. */
    dim3 block(32, 1, 1);
    dim3 grid(256, ((pixels + block.x - 1) / block.x + 255) / 256, 1);

#define MERGE_AND_BLEND(MAX_BUFFERS) \
    mergeAndBlend<MAX_BUFFERS><<<grid, block>>>( \
        (const float**)devPointers, (const uint32_t**)devPointers + count, \
        (const uint32_t**)devPointers + count * 2, \
        count, background, pixels, (uint32_t*)output, alphaThreshold)

    if (count <= 2)
        MERGE_AND_BLEND(2);
    else if (count <= 4)
        MERGE_AND_BLEND(4);
    else if (count <= 8)
        MERGE_AND_BLEND(8);
    else if (count <= 16)
        MERGE_AND_BLEND(16);
    else if (count <= 32)
        MERGE_AND_BLEND(32);
    else
    {
        cudaFree(devPointers);
        cudaFree(output);
        output = 0;
        throw std::runtime_error(
            "Unsupported number of image buffer for fragment merging");
    }

    error = cudaGetLastError();
    cudaFree(devPointers); /* This is synchronous :( */

    return error;
}

cudaError_t compactFragmentLists(
    const size_t width, const size_t height, const size_t x, const size_t y,
    const uint32_t* offsets, cudaArray_t heads, const uint32_t* fragments,
    void*& compacted, size_t& size)
{
    if (!compacted)
        compacted = allocateFragmentBuffer(width * height, offsets, size);

    if (size == 0)
        return cudaSuccess;

    const size_t items = size / (sizeof(uint32_t) + sizeof(float));

    uint32_t* depths = (uint32_t*)compacted;
    uint32_t* colors = (uint32_t*)compacted + items;
    cudaBindTextureToArray(headsRef, heads);
    /* Only 25% occupancy again, but increasing the block size decreases
       performance. */
    dim3 block(8, 4, 1);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y, 1 );
    compact<<<grid, block>>>(width, height, x, y, offsets, fragments,
                             depths, colors);
    cudaUnbindTexture(headsRef);

    return cudaGetLastError();
}

cudaError_t sortFragmentLists(void* fragments, const size_t items,
                              const size_t pixels, const uint32_t* offsets)
{
    float* depths = (float*)fragments;
    uint32_t* colors = (uint32_t*)fragments + items;

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    const bool sm_35 = props.major > 3 || (props.major == 3 && props.minor >= 5);

    /* The first step is to sort pixels with 2 to 32 fragments. We do this
       operation with a bitonic sort kernel where each warp sorts one list. */
    if (sm_35)
    {
        const dim3 block(128, 1, 1);
        const dim3 grid((pixels + block.x - 1) / block.x * 32, 1, 1);
        bitonicSort32<<<grid, block>>>(offsets, pixels, depths, colors);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;

        bitonicSort64<<<grid, block>>>(offsets, pixels, depths, colors);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
    }
    else
    {
        const dim3 block(32, 8, 1); /* Do not change without reviewing the
                                       kernel code. */
        const uint32_t lists = 8 * 2;
        const dim3 grid(
            4096, (((pixels + lists - 1) / lists) + 4095) / 4096, 1);
        bitonicSort32<<<grid, block>>>(offsets, pixels, depths, colors);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
    }

    /* Then the rest of the lists. */
    const dim3 block(32, 1, 1); /* This achieves only 25% occupancy, but bigger
                                   blocks make the performance worse for some
                                   reason I still don't understand. */
    const dim3 grid(256, ((pixels + block.x - 1) / block.x + 255) / 256, 1);
    if (sm_35)
    {
        cudaFuncSetCacheConfig(heapSort<65, 256>, cudaFuncCachePreferL1);
        heapSort<65, 256><<<grid, block>>>(offsets, pixels, depths, colors);
    }
    else
    {
        cudaFuncSetCacheConfig(heapSort<33, 256>, cudaFuncCachePreferL1);
        heapSort<33, 256><<<grid, block>>>(offsets, pixels, depths, colors);
    }

    return cudaGetLastError();
}

cudaError_t sortFragmentLists(void* fragments, const size_t items,
                              const size_t pixels, uint32_t* offsets,
                              const float alphaThreshold, void*& tmp)
{
    float* depths = (float*)fragments;
    uint32_t* colors = (uint32_t*)fragments + items;

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    if (props.major < 3 || (props.major == 3 && props.minor < 5))
        return cudaErrorNotYetImplemented;

    if (tmp == 0)
    {
        const size_t size = sizeof(uint32_t) * (pixels + 1 + items * 2);
        cudaError_t error = cudaMalloc(&tmp, size);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
    }
    uint32_t* newOffsets = (uint32_t*)tmp;

    /* The first step is to sort pixels with 2 to 32 fragments. We do this
       operation with a bitonic sort kernel where each warp sorts one list. */
    dim3 block(128, 1, 1);
    dim3 grid((pixels + block.x - 1) / block.x * 32, 1, 1);
    bitonicSort32WithThreshold<<<grid, block>>>(
        offsets, pixels, depths, colors, newOffsets, alphaThreshold);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;

    bitonicSort64WithThreshold<<<grid, block>>>(
        offsets, pixels, depths, colors, newOffsets, alphaThreshold);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;

    /* Then the rest of the lists. */
    block = dim3(32, 1, 1); /* This achieves only 25% occupancy, but bigger
                               blocks make the performance worse for some
                               reason I still don't understand. */
    grid = dim3(256, ((pixels + block.x - 1) / block.x + 255) / 256, 1);
    cudaFuncSetCacheConfig(heapSortWithThreshold<65, 256>,
                           cudaFuncCachePreferL1);
    heapSortWithThreshold<65, 256><<<grid, block>>>(
        offsets, pixels, depths, colors, newOffsets, alphaThreshold);

    /* Now we compute the prefix sum of the new counts. */
    size_t sizeNeeded = 0;
    cub::DeviceScan::ExclusiveSum(
        0, sizeNeeded, newOffsets, newOffsets, int(pixels) + 1);
    /* We will try to use the buffer used as temporary fragment storage later
       one, for this. If not, the price of a cudaMalloc and cudaFree will have
       to be paid. */
    void* scanTmp = newOffsets + pixels + 1;
    bool allocTmp = sizeNeeded > (items * 2) * sizeof(uint32_t);
    if (allocTmp)
    {
        error = cudaMalloc(&scanTmp, sizeNeeded);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
    }
    error = cub::DeviceScan::ExclusiveSum(
        scanTmp, sizeNeeded, newOffsets, newOffsets, int(pixels) + 1);
    if (error != cudaSuccess)
        return error;
    if (allocTmp)
        cudaFree(scanTmp);

    /* Copying the result from the temporary buffers to the output ones.
       This copy could be certainly avoided, but it would complicate the API
       and device to device copies are specially fast. */
    uint32_t newItems;
    error = cudaMemcpy(&newItems, newOffsets + pixels, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        return error;

    uint32_t* newFragments = newOffsets + pixels + 1;
    block = dim3(128, 1, 1);
    grid = dim3((pixels + block.x - 1) / block.x * 32, 1, 1);
    recompact<<<grid, block>>>(
        offsets, (uint32_t*)depths, colors, newOffsets,
        (uint32_t*)newFragments, (uint32_t*)newFragments + newItems, pixels);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;

    cudaMemcpy(offsets, newOffsets, (pixels + 1) * sizeof(uint32_t),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(fragments, newFragments, newItems * sizeof(uint32_t) * 2,
               cudaMemcpyDeviceToDevice);

    return cudaGetLastError();
}

}
}
}
}
