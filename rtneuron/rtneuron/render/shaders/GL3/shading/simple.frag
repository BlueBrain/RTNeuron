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
const vec3 specular = vec3(0.2, 0.2, 0.2);

$DEFINES

in vec4 color;
in vec3 normal;
in vec3 eye;
#ifdef ACCURATE_HEADLIGHT
in vec3 light;
#endif

out vec4 fragColor;

void main()
{
#ifndef ACCURATE_HEADLIGHT
    const vec3 light = vec3(0.0, 0.0, 1.0);
#endif

    float lambertTerm = dot(normal, light);
    fragColor.rgb = color.rgb * lambertTerm;
    vec3 r = normalize(reflect(-light, normal));
    fragColor.rgb += specular * pow(max(dot(r, normalize(eye)), 0.0), 8.0);
    fragColor.a = color.a;
}
