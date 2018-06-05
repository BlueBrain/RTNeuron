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

/*
   This file provides the definition of:
   - getBaseColor()
   - getBaseAlpha()
   for neuron models that perform interpolation of simulation values at
   fragment level.

   These methods consider the difference between the display modes with
   and without simulation and highlighting.
*/

/* General uniforms and attributes */
uniform bool useStaticColor;

varying float reportedVariable;
varying vec4 color;

vec4 simulationValueToColor(float value);

const int MIX_MULTIPLY_ALPHA = 1;
const int MIX_ADD_COLORS = 0;

/**
   Recover the mixing function encoded in the alpha channel.
*/
int getMixingFunction(const float alpha)
{
    if (alpha >= 1.5) /* 2.0 is not used because interpolation errors make
                         the selection unstable when the alpha channel of
                         the highlight color equals 0. */
        return MIX_ADD_COLORS;
    return MIX_MULTIPLY_ALPHA;
}

float decodeAlpha(const float alpha, const int colorMix)
{
    if (colorMix == MIX_ADD_COLORS)
        return alpha - 2;
    return alpha;
}

float getBaseAlpha(const float base)
{
    int colorMix = getMixingFunction(base);
    float alpha = decodeAlpha(base, colorMix);

    if (useStaticColor)
        return alpha;

    /* Modulating the simulation color according to base color */
    if (colorMix == MIX_MULTIPLY_ALPHA)
        return alpha * simulationValueToColor(reportedVariable).a;
    /* else colorMix == MIX_ADD_COLORS */
    return alpha + simulationValueToColor(reportedVariable).a;
}

vec4 getBaseColor(const vec4 base)
{
    int colorMix = getMixingFunction(base.a);
    float alpha = decodeAlpha(base.a, colorMix);

    if (useStaticColor)
        return vec4(base.rgb, alpha);

    vec4 color = simulationValueToColor(reportedVariable);

    if (colorMix == MIX_MULTIPLY_ALPHA)
        color.a *= alpha;
    else
    {
        color += vec4(base.rgb, alpha);
        color = clamp(color, vec4(0.0), vec4(1.0));
    }

    return color;
}

float getBaseAlpha()
{
    return getBaseAlpha(color.a);
}

vec4 getBaseColor()
{
    return getBaseColor(color);
}
