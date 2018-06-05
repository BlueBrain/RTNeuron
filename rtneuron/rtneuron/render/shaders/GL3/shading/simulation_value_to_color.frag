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

uniform sampler1D simulationColorMap;
uniform vec2 simulationColorMapRange;
uniform sampler1D spikeColorMap;
uniform float threshold;
uniform vec4 aboveThresholdColor;

vec4 simulationValueToColor(float value)
{
#ifdef SHOW_SPIKES
    if (value <= 0.0)
    {
        value = -value;
        if (value < 1e-7)
            value = 0.0;
        vec4 color = texture(spikeColorMap, value);
#ifndef USE_ALPHA_BLENDING
        /* Needed in DB decomposition modes */
        color[3] = 1.0;
#endif
        return color;
    }

    value -= 128.0;
#endif

    vec4 color;
    if (value < threshold)
    {
        vec2 range = simulationColorMapRange;
        color = texture(simulationColorMap,
                        (value - range[0]) / (range[1] - range[0]));
    }
    else
        color = aboveThresholdColor;

#ifndef USE_ALPHA_BLENDING
    /* Needed in DB decomposition modes */
    color[3] = 1.0;
#endif

    return color;
}
