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

uniform vec4 highlightColor;

/* Soma colors are assigned per object, not per fragment. That means that
   the base color for shading is fully resolved here. */

uniform bool useStaticColor;
/* This variable is converted from byte (the type used in the C++ side)
   to float by the API. */
in float highlighted;

vec4 simulationValueToColor(float value);

float getReportedVariable();

/**
   Returns the per vertex color of a model considering whether it's
   highlighted or not.

   The alpha channel encodes how to mix dynamic and static colors when
   simulation values are interpolated at fragment level.

   - [0-1]: multiply alpha
   - [2-3]: add colors
*/
vec4 getVertexColor()
{
    if (useStaticColor)
    {
        if (highlighted == 1.0)
            return highlightColor;
        return gl_Color;
    }

    float reportedVariable = getReportedVariable();
    vec4 color = simulationValueToColor(reportedVariable);

    if (highlighted == 1.0)
    {
        color += highlightColor;
        /* Clamping is needed here (in GL2 it's not needed because clamping
           is automatic in gl_FrontColor. */
        color = clamp(color, vec4(0), vec4(1));
    }
    else
        color.a *= gl_Color.a;

    return color;
}
