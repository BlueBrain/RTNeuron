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

#extension GL_EXT_gpu_shader4 : enable

$DEFINES

#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
varying vec3 light;
#endif

#ifdef INFLATABLE
uniform float inflation;
#endif

varying vec3 normal;
varying vec3 eye;
varying vec4 color;
varying float reportedVariable;

float getReportedVariable();
vec4 getVertexColor();

void basicVertexShading()
{
    vec4 vertex = gl_Vertex;

#ifdef INFLATABLE
    vertex.xyz += gl_Normal * inflation;
#endif

    /* World space normal. */
    /** \bug Normals from mesh models seem to be inverted it could be because
             of the apparently wrong winding of triangles. */
    normal = gl_NormalMatrix * gl_Normal;
    reportedVariable = getReportedVariable();

    color = getVertexColor();

#ifdef SHOW_SPIKES
    if (reportedVariable <= 0.0)
        vertex.xyz += gl_Normal * -reportedVariable * 0.5;
#endif
    vec4 worldPos = gl_ModelViewMatrix * vertex;

    /* Vertex coordinates for clipping planes */
    gl_ClipVertex = worldPos;

    /* Eye vector in world coordinates. */
    eye = -vec3(worldPos);

#ifdef ACCURATE_HEADLIGHT
    /* Light direction. */
    light = vec3(lightSource + eye);
#endif
    gl_Position = gl_ProjectionMatrix * worldPos;
}
