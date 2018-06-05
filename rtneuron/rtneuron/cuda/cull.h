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

#ifndef RTNEURON_CUDA_CULL_H
#define RTNEURON_CUDA_CULL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace cuda
{
struct CullFrustum
{
    float near;
    float far;
    float2 rn; // Right plane normal (in global x,z coordinates)
    float2 ln; // Left plane normal (in global x,z coordinates)
    float2 bn; // Bottom plane normal (in global y,z coordinates)
    float2 tn; // Top plane formal (in global y,z coordinates)
};

struct SkeletonInfo
{
    size_t length;
    size_t blockCount;
    size_t sectionsStartsLength;
};

const unsigned int CULL_BLOCK_SIZE = 64;

/*
  Additional declarations
*/
cudaError_t cullSkeleton(const float modelview[16],
                         const struct CullFrustum& frustum,
                         SkeletonInfo* arrays, uint32_t* visibilities,
                         size_t length, bool useSharedMemory,
                         cudaStream_t stream);
}
}
}
}
#endif
