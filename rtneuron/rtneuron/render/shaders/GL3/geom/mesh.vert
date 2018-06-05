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

#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
out vec3 light;
#endif

out vec3 normal, eye;
out vec4 color;

vec4 getVertexColor();
void computeClipDistances(const vec4 vertex);

void shadeVertex()
{
    vec4 vertex = osg_Vertex;
    normal = osg_NormalMatrix * osg_Normal;

    color = getVertexColor();

    vec4 worldPos = osg_ModelViewMatrix * vertex;
    computeClipDistances(worldPos);

    /* Eye vector in world coordinates. */
    eye = -vec3(worldPos);

#ifdef ACCURATE_HEADLIGHT
    /* Light direction. */
    light = vec3(lightSource + eye);
#endif
    gl_Position = osg_ProjectionMatrix * worldPos;
}

void trivialShadeVertex()
{
    gl_Position = osg_ProjectionMatrix * (osg_ModelViewMatrix * osg_Vertex);
    color = getVertexColor();
}
