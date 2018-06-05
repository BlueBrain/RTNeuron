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

#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

$DEFINES

varying out vec3 normal;
varying out vec3 eye;

#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
varying vec3 light;
#endif

void main()
{
    normal = cross(gl_PositionIn[1].xyz - gl_PositionIn[0].xyz,
                   gl_PositionIn[2].xyz - gl_PositionIn[0].xyz);
    if (normal.z < 0)
        normal = -normal;

    for (int i = 0; i < 3; ++i)
    {
        gl_FrontColor = gl_FrontColorIn[i];
        gl_ClipVertex = gl_PositionIn[i];
        gl_Position = gl_ProjectionMatrix * gl_PositionIn[i];
        eye = -gl_PositionIn[i].xyz;
#ifdef ACCURATE_HEADLIGHT
        light = vec3(lightSource + eye);
#endif
        EmitVertex();
    }
    EndPrimitive();
}
