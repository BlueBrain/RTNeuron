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

$DEFINES

#ifdef USE_COLOR_UNIFORM
uniform vec4 globalColor;
#endif

void shadeVertex()
{
    vec4 vertex = gl_Vertex;
#ifdef USE_COLOR_UNIFORM
    gl_FrontColor = globalColor;
#else
    gl_FrontColor = gl_Color;
#endif

    vec4 worldPos = gl_ModelViewMatrix * vertex;

    /* Vertex coordinates for clipping planes */
    gl_ClipVertex = worldPos;
    gl_Position = worldPos;
}

void trivialShadeVertex()
{
    gl_Position = gl_ModelViewMatrix * gl_Vertex;
}
