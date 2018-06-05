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

#ifdef INFLATABLE
uniform float inflation;
#endif

out vec3 normal, eye;
out float reportedVariable;
out vec4 color;

float getReportedVariable();
vec4 getVertexColor();
void computeClipDistances(const vec4 vertex);

void basicVertexShading()
{
    vec4 vertex = osg_Vertex;

#ifdef INFLATABLE
    vertex.xyz += osg_Normal * inflation;
#endif

    /* World space normal. */
    /** \bug Normals from mesh models seem to be inverted it could be because
             of the apparently wrong winding of triangles. */
    normal = osg_NormalMatrix * osg_Normal;
    reportedVariable = getReportedVariable();

    color = getVertexColor();

#ifdef SHOW_SPIKES
    if (reportedVariable <= 0.0)
        vertex.xyz += osg_Normal * -reportedVariable * 0.5;
#endif
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
