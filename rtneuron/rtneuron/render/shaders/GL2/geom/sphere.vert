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

#extension GL_EXT_geometry_shader4 : enable

#ifndef USE_RADIUS_UNIFORM
varying float radius;
#endif

#ifdef INFLATABLE
uniform float inflation;
#endif

vec4 getVertexColor();

void shadeVertex()
{
    gl_Position = gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1);
    gl_FrontColor = getVertexColor();

#ifndef USE_RADIUS_UNIFORM
    radius = gl_Vertex.w;
#ifdef INFLATABLE
    if (radius >= 0) /* Negative radius are used for LOD culling, so we need
                        to preserve them. */
        radius += inflation;
#endif
#endif
}

void trivialShadeVertex()
{
    shadeVertex();
}
