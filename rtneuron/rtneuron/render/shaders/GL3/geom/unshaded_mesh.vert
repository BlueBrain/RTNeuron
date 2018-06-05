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

uniform mat4x4 osg_ModelViewMatrix;
uniform uint osg_NumClipPlanes;
uniform vec4 osg_ClipPlanes[8];
uniform mat4x4 osg_ProjectionMatrix;
in vec4 osg_Vertex;
in vec4 osg_Color;

out vec4 color;

void main()
{
    color = osg_Color;
    vec4 clipVertex = osg_ModelViewMatrix * osg_Vertex;
    for (uint j = 0; j < osg_NumClipPlanes; j++)
        gl_ClipDistance[j] = dot(clipVertex, osg_ClipPlanes[j]);
    gl_Position = osg_ProjectionMatrix * clipVertex;
}
