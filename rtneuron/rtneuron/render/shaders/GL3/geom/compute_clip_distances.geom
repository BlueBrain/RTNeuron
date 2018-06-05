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

#version 410

uniform uint osg_NumClipPlanes;
uniform vec4 osg_ClipPlanes[8];

/* Compute all the gl_ClipDistance elements up to maximum active clip plane.

   This function is actually a workaround for what looks like a GLSL compiler
   bug in the NVIDIA driver 304.108. The problem is that if done with a loop
   the compiler issues a strange complaint about illegal expression. This
   error occurs in geometry shaders, but not in vertex shaders. */


void computeClipDistances(const vec4 vertex)
{
#define TEST_AND_CLIP(i) \
    if (osg_NumClipPlanes > i) { \
        gl_ClipDistance[i] = dot(vertex, osg_ClipPlanes[i]);

    TEST_AND_CLIP(0)
    TEST_AND_CLIP(1)
    TEST_AND_CLIP(2)
    TEST_AND_CLIP(3)
    TEST_AND_CLIP(4)
    TEST_AND_CLIP(5)
    TEST_AND_CLIP(6)
    TEST_AND_CLIP(7)

    }}}}}}}}
}
