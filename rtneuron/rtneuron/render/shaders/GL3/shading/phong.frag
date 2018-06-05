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

const vec3 specular = vec3(0.2, 0.2, 0.1);
const vec3 ambient = vec3(0.0, 0.0, 0.0);

/**
   normal, eye and light should be normalized
*/
vec4 phong(const vec4 objectColor, const vec3 normal, const vec3 norm_eye,
           const vec3 light)
{
    vec4 color = objectColor;

    if (color.a == 0.0)
        discard;

    float lambertTerm = dot(normal, light);
#ifdef DOUBLE_FACED
    lambertTerm = abs(lambertTerm);
#else
    if (color.a < 0.9999)
        lambertTerm = abs(lambertTerm);
    else
        lambertTerm = max(lambertTerm, 0.0);
#endif

    /* Adding diffuse color */
    color.rgb = color.rgb * lambertTerm + ambient;

    /* Adding specular reflection */
    vec3 r = reflect(-light, normal);
    color.rgb += specular * pow(max(dot(r, norm_eye), 0.0), 8.0);

    return color;
}
