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

uniform mat3x3 osg_NormalMatrix;
uniform mat4x4 osg_ModelViewMatrix;
uniform mat4x4 osg_ProjectionMatrix;
#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
#endif
in vec4 osg_Vertex;
in vec4 osg_Color;
in vec3 osg_Normal;
in vec4 osg_MultiTexCoord0;

out vec3 normal;
out vec4 color;
out vec3 eye;
#ifdef ACCURATE_HEADLIGHT
out vec3 light;
#endif
out vec4 texCoord;

void main()
{
    texCoord = osg_MultiTexCoord0;

    color = osg_Color;
    vec4 world = osg_ModelViewMatrix * osg_Vertex;
    eye = -world.xyz;
    normal = osg_NormalMatrix * osg_Normal;
#ifdef ACCURATE_HEADLIGHT
    /* Light direction. */
    light = vec3(lightSource + eye);
#endif
    gl_Position = osg_ProjectionMatrix * world;
}
