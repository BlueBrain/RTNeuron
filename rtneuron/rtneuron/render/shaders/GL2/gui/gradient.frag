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

#version 130

$DEFINES

in vec3 normal;
in float variable;

vec4 computeDataColor(float value);

uniform float colorMapMax;

void main()
{
    vec3 n = normalize(normal);
    vec4 color = computeDataColor(variable);
#ifndef ALPHA_BLENDING
    color.a = 1.0;
#endif
    color.rgb *= dot(n, vec3(0.0, 0.0, 1.0));
    color.rgb += (vec3(0.4, 0.4, 0.4) *
                  pow(max(dot(n, normalize(vec3(-0.5, 0.0, 1.0))), 0), 16.0));
    gl_FragColor = color;
}
