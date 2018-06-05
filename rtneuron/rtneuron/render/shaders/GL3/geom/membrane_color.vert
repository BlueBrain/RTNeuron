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

uniform vec4 highlightColor;
uniform bool highlighted;

/* Uniforms and attributes for static (i.e. no simulation) color map based
   coloring. */
uniform bool useColorMap;
uniform sampler1D staticColorMap;
uniform vec2 staticColorMapRange;
in float colorMapValueAttrib;

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
    if (highlighted)
        return vec4(highlightColor.rgb, highlightColor.a + 2);

    if (useColorMap)
    {
#ifdef SHOW_SPIKES
        float reportedVariable = getReportedVariable();
        if (reportedVariable <= 0.0)
        {
            /* This code assumes that the coloring mode is by-width, it needs
               fixing for by-distance. */
            float colorMapValue = colorMapValueAttrib * -reportedVariable * 2;
            return texture(staticColorMap, colorMapValue);
        }
#endif
        float v = (colorMapValueAttrib - staticColorMapRange[0]) /
                  (staticColorMapRange[1] - staticColorMapRange[0]);
        return texture(staticColorMap, v);
    }

    return gl_Color;
}
