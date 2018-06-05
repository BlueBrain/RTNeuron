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

$DEFINES

#ifdef INFLATABLE
uniform float inflation;
#endif

attribute float colorMapValueAttrib;
varying float colorMapValue;

void basicVertexShading();
#ifdef SHOW_SPIKES
float getReportedVariable();
#endif

void shadeVertex()
{
    basicVertexShading();
}

/* This function is used in geometry passes of the depth peeling algorithm
   that don't require vertex attribute interpolation */
void trivialShadeVertex()
{
    vec4 vertex = gl_Vertex;
#ifdef INFLATABLE
    vertex.xyz += gl_Normal * inflation;
#endif

#ifdef SHOW_SPIKES
    float reportedVariable = getReportedVariable();
    if (reportedVariable <= 0.0)
    {
        vertex.xyz += gl_Normal * -reportedVariable * 0.5;
        colorMapValue =
            1.0 - pow(1 - colorMapValueAttrib, -reportedVariable * 2.0);
    }
    else
        colorMapValue = colorMapValueAttrib;
#else
    colorMapValue = colorMapValueAttrib;
#endif

    gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * vertex);
}
