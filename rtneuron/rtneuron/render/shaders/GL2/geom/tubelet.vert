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

#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_geometry_shader4 : enable

$DEFINES

#ifdef INFLATABLE
uniform float inflation;
#endif

attribute float radiusAttrib;
attribute vec4 cutPlaneAttrib;

varying float radius;
varying vec4 cutPlane;
varying float reportedVariableIn;
varying vec4 colorIn;

float getReportedVariable();
vec4 getVertexColor();

void shadeVertex()
{
    gl_Position = gl_ModelViewMatrix * gl_Vertex;
    radius = radiusAttrib;
#ifdef INFLATABLE
    radius += inflation;
#endif
    cutPlane = gl_ModelViewMatrixInverseTranspose * cutPlaneAttrib;
    reportedVariableIn = getReportedVariable();
    colorIn = getVertexColor();

#ifdef SHOW_SPIKES
    if (reportedVariableIn <= 0.0)
        radius += -reportedVariableIn * 0.5;
#endif
}

void trivialShadeVertex()
{
    shadeVertex();
}
