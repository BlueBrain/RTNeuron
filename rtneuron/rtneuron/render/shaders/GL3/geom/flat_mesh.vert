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

uniform mat4x4 osg_ModelViewMatrix;
uniform mat4x4 osg_ProjectionMatrix;
uniform mat3x3 osg_NormalMatrix;
in vec4 osg_Vertex;
in vec3 osg_Normal;
#ifdef USE_COLOR_UNIFORM
uniform vec4 osg_Color;
#else
in vec4 osg_Color;
#endif

out vec4 colorIn;

void shadeVertex()
{
    vec4 vertex = osg_Vertex;
    colorIn = osg_Color;

    vec4 worldPos = osg_ModelViewMatrix * vertex;
    gl_Position = worldPos;
}

void trivialShadeVertex()
{
    gl_Position = osg_ModelViewMatrix * osg_Vertex;
    colorIn = osg_Color;
}
