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
#ifdef INFLATABLE
uniform float inflation;
#endif

in vec4 osg_Vertex;
#if defined SHOW_SPIKES || defined INFLATABLE
in vec3 osg_Normal;
#endif

in float colorMapValueAttrib;
out float colorMapValue;

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
    vec4 vertex = osg_Vertex;
#ifdef INFLATABLE
    vertex.xyz += osg_Normal * inflation;
#endif

#ifdef SHOW_SPIKES
    float reportedVariable = getReportedVariable();
    if (reportedVariable <= 0.0)
    {
        vertex.xyz += osg_Normal * -reportedVariable * 0.5;
        colorMapValue =
            1.0 - pow(1 - colorMapValueAttrib, -reportedVariable * 2.0);
    }
    else
        colorMapValue = colorMapValueAttrib;
#else
    colorMapValue = colorMapValueAttrib;
#endif
    /** \bug There's something wrong with osg_ModelViewProjectionMatrix,
        so we use the decomposition in projection and model-view. */
    gl_Position = osg_ProjectionMatrix * (osg_ModelViewMatrix * vertex);
}
