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
in vec3 normal, eye;
#ifdef ACCURATE_HEADLIGHT
in vec3 light;
#else
const vec3 light = vec3(0, 0, 1);
#endif
in vec4 color;

uniform float alpha;
const vec3 specularColor = vec3(0.2, 0.2, 0.1);

float fragmentDepth()
{
    return gl_FragCoord.z;
}

float fragmentAlpha()
{
    return alpha;
}

vec4 shadeFragment()
{
    /* Getting the base color */
    vec4 baseColor = color;

    vec3 n = normalize(normal);
    vec3 l = normalize(light);
    vec3 e = normalize(eye);

    float lambertTerm = dot(n, l);

    if (lambertTerm < 0.0)
        lambertTerm = -lambertTerm;

    /* Adding diffuse color */
    vec3 shadedColor = baseColor.rgb * lambertTerm;

    /* Adding specular reflection */
    vec3 r = reflect(-l, n);
    float specular = pow(max(dot(r, e), 0.0), 8.0);
    shadedColor += specularColor * specular;

    float a = alpha;

    return vec4(shadedColor, a);
}
