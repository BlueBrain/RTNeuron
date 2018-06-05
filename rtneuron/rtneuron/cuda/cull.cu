//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2016 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>
#include "geometry_functions.cu"

#include "cull.h"

#define HANDLE_SCALING_MODELVIEW

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

__global__ void cullSkeleton_(Matrix4x4 modelview, CullFrustum f,
                              SkeletonInfo *skeleton,
                              uint32_t *visibilities)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    /* Computing the address of the different data pointers. */
    size_t size = skeleton->length;
    const uint8_t  *array = ((const uint8_t*)skeleton) + sizeof(SkeletonInfo);
    const float3   *starts =   (const float3*) array;
    array +=                   sizeof(float3) * size;
    const float3   *ends =     (const float3*) array;
    array +=                   sizeof(float3) * size;
    const float    *widths =   (const float*) array;
    array +=                   sizeof(float) * size;
    const uint16_t *sections = (const uint16_t*) array;
    array +=                   sizeof(uint16_t) * size;
    const uint8_t  *portions = (const uint8_t*) array;

    if (index >= size)
        return;

    const uint8_t portion = portions[index];
    const uint16_t section = sections[index];
    const float3 start = mult3(starts[index], modelview);
    const float3 end = mult3(ends[index], modelview);
#ifdef HANDLE_SCALING_MODELVIEW
    const float width = widths[index] * get_scale(modelview).x;
#else
    const float width = widths[index];
#endif

    /** Near and far are not properly set from the calling context
        this is because clampled near/far values aren't known until the
        end of the cull visitor traversal. On the other hand the
        cull visitor ensures that the whole drawable is encompassed by the
        clampled values so testing for near and far is not really needed. */
    if (/* The free term of all the plane equations is 0 (they all
           contain the origin) so we only need the plane normal to
           check in which side a point is falling. */
        ((f.rn.x * start.x + f.rn.y * start.z <= width ||
          f.rn.x * end.x   + f.rn.y * end.z   <= width) &&
         (f.tn.x * start.y + f.tn.y * start.z <= width ||
          f.tn.x * end.y   + f.tn.y * end.z   <= width) &&
         (f.ln.x * start.x + f.ln.y * start.z >= -width ||
          f.ln.x * end.x   + f.ln.y * end.z   >= -width) &&
         (f.bn.x * start.y + f.bn.y * start.z >= -width ||
          f.bn.x * end.y   + f.bn.y * end.z   >= -width)))
    {
        /* Capsule-frustum intersection test passed */
        atomicOr((int*)&visibilities[section], 1 << portion);
    }
}

/**
   Version that implements shared memory reduce.
 */
__global__ void cullSkeletonShMem_(Matrix4x4 modelview, CullFrustum f,
                                   SkeletonInfo *skeleton,
                                   uint32_t *visibilities)
{
    __shared__ uint32_t bits[CULL_BLOCK_SIZE];
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    /* Computing the address of the different data pointers. */
    const size_t size = skeleton->length;
    const size_t blocks = skeleton->blockCount;
    const size_t sectionsStartsLength = skeleton->sectionsStartsLength;
    const uint8_t *array = ((const uint8_t*)skeleton) + sizeof(SkeletonInfo);
    const float3   *starts =                (const float3*) array;
    array += sizeof(float3) * size;
    const float3   *ends =                  (const float3*) array;
    array += sizeof(float3) * size;
    const float    *widths =                (const float*) array;
    array += sizeof(float) * size;
    const uint16_t *firstBlockSection =     (const uint16_t*) array;
    array += sizeof(uint16_t) * blocks;
    const uint16_t *accumSectionsPerBlock = (const uint16_t*) array;
    array += sizeof(uint16_t) * (blocks + 1);
    const uint16_t *perBlockSectionStarts = (const uint16_t*) array;
    array += sizeof(uint16_t) * sectionsStartsLength;
    const uint8_t *portions =               (const uint8_t*) array;

    bits[threadIdx.x] = 0;
    if (index < size)
    {
        const uint8_t portion = portions[index];
        const float3 start = mult3(starts[index], modelview);
        const float3 end = mult3(ends[index], modelview);
#ifdef HANDLE_SCALING_MODELVIEW
        const float width = widths[index] * get_scale(modelview).x;
#else
        const float width = widths[index];
#endif

        /** Near and far are not properly set from the calling context
            this is because clampled near/far values aren't known until the
            end of the cull visitor traversal. On the other hand the
            cull visitor ensures that the whole drawable is encompassed by the
            clampled values so testing for near and far is not really needed. */
        if (/* The free term of all the plane equations is 0 (they all
               contain the origin) so we only need the plane normal to
               check in which side a point is falling. */
            ((f.rn.x * start.x + f.rn.y * start.z <= width ||
              f.rn.x * end.x   + f.rn.y * end.z   <= width) &&
             (f.tn.x * start.y + f.tn.y * start.z <= width ||
              f.tn.x * end.y   + f.tn.y * end.z   <= width) &&
             (f.ln.x * start.x + f.ln.y * start.z >= -width ||
              f.ln.x * end.x   + f.ln.y * end.z   >= -width) &&
             (f.bn.x * start.y + f.bn.y * start.z >= -width ||
              f.bn.x * end.y   + f.bn.y * end.z   >= -width))) {
                /* Capsule-frustum intersection test passed */
            bits[threadIdx.x] = 1 << portion;
        }
    }

    __syncthreads();

    if (index >= size)
        return;

    uint16_t accumSections = accumSectionsPerBlock[blockIdx.x];
    const uint16_t atomicFlags = accumSections & 0xC000;
    /* This number indicates where in the perBlockSectionStarts we have
       to search for the start indices of each section within the current
       block. Note that sections between two blocks count twice so this
       number is presumably larger than the total section count. */
    accumSections &= ~0xC000;
    /* Now we need to now how many threads participate in the reduce
       operation for this block, for that
       accumSectionsPerBlock[blockIdx.x + 1] is guaranteed to be always
       defined. */
    const uint16_t sectionsInThisBlock =
        ((accumSectionsPerBlock[blockIdx.x + 1] & ~0xC000) - accumSections);
    if (threadIdx.x < sectionsInThisBlock)
    {
        /* This thread participates in the reduce operation. Sections are
           supposed to be enumerated exhaustively so we only have to add
           the thread id to the first section of the block to get the
           section this thread will be computing. */
        const uint16_t section = firstBlockSection[blockIdx.x] + threadIdx.x;
        /* Now we compute the interval in which the masks of this section
           have been stored. */
        const int index = accumSections + threadIdx.x;
        const int sectionStart = perBlockSectionStarts[index];
        int sectionEnd = perBlockSectionStarts[index + 1];
        bool useAtomic = false;
        if (sectionEnd == 0)
        {
            /* This happens in the last section within the block because
               index + 1 points to the offset for the first section of the
               next block (which should always 0). The allocation and
               preprocessing must ensure that no out of bounds access is
               performed. */
            sectionEnd = CULL_BLOCK_SIZE;
            useAtomic = atomicFlags & 0x4000;
        }
        if (threadIdx.x == 0)
            useAtomic = atomicFlags & 0x8000;

        uint32_t mask = 0;
        for (int i = sectionStart; i < sectionEnd; ++i)
            mask |= bits[i];
        if (mask != 0)
        {
            if (useAtomic)
                atomicOr((int*)&visibilities[section], mask);
            else
                visibilities[section] = mask;
        }
    }
}

/*
  Invokation wrappers
*/

/* Wrapper for cullSkeleton_ */

cudaError_t cullSkeleton(
    const float modelview[16], const CullFrustum &frustum,
    SkeletonInfo *skeleton, uint32_t *visibilities, size_t length,
    bool useSharedMemory, cudaStream_t stream)
{
    dim3 block(CULL_BLOCK_SIZE);
    dim3 grid((length + CULL_BLOCK_SIZE - 1) / CULL_BLOCK_SIZE);
    const Matrix4x4 *m =
        static_cast<const Matrix4x4 *>(static_cast<const void*>(modelview));
    if (useSharedMemory)
        cullSkeletonShMem_<<<grid, block, 0, stream>>>(*m, frustum, skeleton,
                                                       visibilities);
    else
        cullSkeleton_<<<grid, block, 0, stream>>>(*m, frustum, skeleton,
                                                  visibilities);

    return cudaGetLastError();
}

}
}
}
}
