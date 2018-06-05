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

$DEFINES

#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

uniform mat4x4 osg_ProjectionMatrix;

uniform uint osg_NumClipPlanes;

in vec4 colorIn[];

out vec3 normal, eye;
out vec4 color;

#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
out vec3 light;
#endif

void computeClipDistances(const vec4 vertex);

void main()
{
    vec3 tangent = normalize(gl_PositionIn[1].xyz - gl_PositionIn[0].xyz);

    for (int i = 0; i < 2; ++i)
    {
        color = colorIn[i];
        computeClipDistances(gl_PositionIn[i]);
        gl_Position = osg_ProjectionMatrix * gl_PositionIn[i];
        eye = -gl_PositionIn[i].xyz;
#ifdef ACCURATE_HEADLIGHT
        light = vec3(lightSource + eye);
#endif
        vec3 v = normalize(cross(eye, tangent));
        normal = cross(tangent, v);
        if (normal.z < 0)
            normal = -normal;

        EmitVertex();
    }
    EndPrimitive();
}
