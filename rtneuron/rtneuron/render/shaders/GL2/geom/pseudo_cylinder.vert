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
#extension GL_NV_coverage_sample : enable

$DEFINES

#ifdef INFLATABLE
uniform float inflation;
#endif

attribute vec4 tangentAndThickness; /* [tangent.xyz, radius] */

varying float reportedVariableIn;
varying vec3 tangentIn;
varying float radiusIn;
varying vec4 colorIn;

float getReportedVariable();
vec4 getVertexColor();

void shadeVertex()
{
    /* World space tangent. */
    tangentIn = vec3(gl_ModelViewMatrix * vec4(tangentAndThickness.xyz, 0.0));
    radiusIn = tangentAndThickness.w;
#ifdef INFLATABLE
    radiusIn += inflation;
#endif
    reportedVariableIn = getReportedVariable();
    gl_Position = gl_ModelViewMatrix * gl_Vertex;

    colorIn = getVertexColor();

#ifdef SHOW_SPIKES
    if (reportedVariableIn <= 0.0)
        radiusIn += -reportedVariableIn * 0.5;
#endif
}

void trivialShadeVertex()
{
    shadeVertex();
}
