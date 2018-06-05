/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Juan Hernando <juan.hernando@epfl.ch>
 *
 * This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef RTNEURON_CUDA_FRAGMENTS_H
#define RTNEURON_CUDA_FRAGMENTS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <vector>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace cuda
{
/**
   Convert the per pixel fragment counts of an image region to offsets in the
   fragment buffer.

   @param width Viewport width.
   @param height Viewport height.
   @param x Viewport left side coordinate.
   @param y VIewport bottom side coordinate.
   @param counts a 2D CUDA array mapped to the counts texture returned by
          osgTransparency::FragmentListOITBin.
   @param offsets width * height + 1 size array with the exclusive scan (prefix
          sum) of the counts and total number of fragments at the last position.
   @param tmp Optional GPU buffer for temporary storage. If not 0 and tmpSize
          is greater or equal than the temporary storage needed, this buffer
          will be used for that purpose. If 0 or the size of not big enough,
          temporary storage will be allocated and the pointer returned in this
          parameter.
   @param tmpSize The size of the temporary buffer provided at input if not 0.
          At output, it will contain the size of the temporary buffer if it was
          allocated internally, otherwise it remains the same.
*/
cudaError_t fragmentCountsToOffsets(size_t width, size_t height, size_t x,
                                    size_t y, cudaArray_t counts,
                                    uint32_t*& offsets, void*& tmp,
                                    size_t& tmpSize);

/**
   Allocates a buffer with the necessary space to hold the fragment data.
   This function can be used to preallocate the buffer needed by
   compactAndSortFragmentLists.
   Throws if the allocation fails.
   @param pixelCount Total number of pixels
   @param offsets The offsets computed by fragmentCountsToOffsets.
   @param size The size in bytes of the allocated buffer (output parameter)
   @return The device pointer to the buffer. The caller takes the ownership
           of the device pointer, which has to be deallocated with cudaFree.
*/
void* allocateFragmentBuffer(const size_t pixelCount, const uint32_t* offsets,
                             size_t& size);

/**
   Compacts the fragment lists of a viewport in a single buffer.

   The fragment buffer is supposed to provide per pixel linked lists that start
   at the positions indicated by the heads texture.

   @param width Viewport width.
   @param height Viewport height.
   @param x Viewport left side coordinate.
   @param y VIewport bottom side coordinate.
   @param heads A 2D CUDA array mapped to the heads texture returned by
          osgTransparency::FragmentListOITBin.
   @param fragments The pointer to the mapped fragment buffer passed by
          osgTransparency::FragmentListOITBin.
   @param offsets The offsets computed by fragmentCountsToOffsets. These
          offsets may be updated if alphaThreshold is different from 0.
   @param output Compacted and sorted buffer. Allocated by the function call
          if equal to 0. Otherwise the caller must ensure that there is enough
          space to hold all the fragments.
   @param size Size of the output buffer in bytes. If the output buffer is
          given by the caller, this size must also be given.
   @param alphaThreshold Fragments can be discarded when the accumulated alpha
          is equal or greater than this value. 0 means no fragment may be
          discarded.
   @param tmp If alphaThreshold is != 0, the algorithm needs additional
          temporary storage sized at most pixels * 4 + offsets[pixels] * 8
          bytes, where pixels = width * height. If this pointer is different
          from 0, this function will use this as the temporary buffer and the
          caller is responsible of ensuring it has enough space. This
          function does not free the buffer before returning to avoid the
          host-device synchronization caused by cudaFree.
   @return cudaSuccess or the error code in case of error. In case of error
           the output and tmp pointer may have been allocated, the client is
           responsible of checking this pointers to avoid memory leaks.
*/
cudaError_t compactAndSortFragmentLists(size_t width, size_t height, size_t x,
                                        size_t y, cudaArray_t heads,
                                        const uint32_t* fragments,
                                        uint32_t* offsets, void*& output,
                                        size_t& size, float alphaThreshold,
                                        void*& tmp);

/** The same as above with no alphaThreshold cut-off. */
cudaError_t compactAndSortFragmentLists(size_t width, size_t height, size_t x,
                                        size_t y, cudaArray_t heads,
                                        const uint32_t* fragments,
                                        uint32_t* offsets, void*& output,
                                        size_t& size);

/**
   N-way Merge and blend of N compacted and sorted fragment buffers.

   @param depths The device pointers to the fragment depth buffers.
   @param colors The device pointers to the fragment color buffers.
   @param offsets Device pointers to the offset buffers with the offset of
          each pixel within each fragment buffer.
   @param background The background color as a RGBA tuple (from least to most
          significant byte) with 8-bit channels.
   @param pixels The number of pixels in the offsets and ouput.
   @param output Final destination RGBA buffer. Allocated by the function call
          if equal to 0.
   @param alphaThreshold The alpha value over which more distant fragments can
          be considered fully occluded.
*/
cudaError_t mergeAndBlendFragments(const std::vector<const float*>& depths,
                                   const std::vector<const uint32_t*>& colors,
                                   const std::vector<const uint32_t*>& offsets,
                                   uint32_t background, size_t pixels,
                                   void*& output, float alphaThreshold = 0.999);

/** @internal */
cudaError_t compactFragmentLists(size_t width, size_t height, size_t x,
                                 size_t y, const uint32_t* offsets,
                                 cudaArray_t heads, const uint32_t* fragments,
                                 void*& compacted, size_t& size);

/** @internal */
cudaError_t sortFragmentLists(void* fragments, size_t items, size_t pixels,
                              const uint32_t* offsets);

/** @internal */
cudaError_t sortFragmentLists(void* fragments, size_t items, size_t pixels,
                              uint32_t* offsets, float alphaThreshold,
                              void*& tmp);
}
}
}
}
#endif
