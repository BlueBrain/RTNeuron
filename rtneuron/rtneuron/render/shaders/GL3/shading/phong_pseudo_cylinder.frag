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

$DEFINES

#ifdef FASTER_PSEUDOCYLINDER_NORMALS
in vec4 tangentAndOffset;
#else
in vec3 normal;
#endif

#ifdef ACCURATE_HEADLIGHT
in vec3 light, eye;
#else
in vec3 eye;
#endif

vec4 phong(const vec4 color, const vec3 normal, const vec3 eye,
           const vec3 light);
vec4 getBaseColor();
float getBaseAlpha();

vec4 shadeFragment()
{
#ifdef FASTER_PSEUDOCYLINDER_NORMALS
    /* Paradoxically, this version runs faster. A plausible explanation is
       that it requires half the tessellation level of the other */
    vec3 y = normalize(cross(eye, tangentAndOffset.xyz));
    vec3 x = normalize(cross(tangentAndOffset.xyz, y));
    float t = tangentAndOffset.w;
    vec3 n = normalize(sqrt(1 - pow(t, 2)) * x + abs(t) * y);
#else
    vec3 n = normalize(normal);
#endif

#ifdef ACCURATE_HEADLIGHT
    vec3 l = normalize(light);
#else
    vec3 l = vec3(0, 0, 1);
#endif
    vec3 e = normalize(eye);

    vec4 outColor = phong(getBaseColor(), n, e, l);
#ifdef USE_ALPHA_BLENDING
    outColor.a = outColor.a * (2 - outColor.a);
#endif
    return outColor;
}

float fragmentDepth()
{
    return gl_FragCoord.z;
}

float fragmentAlpha()
{
    float a = getBaseAlpha();
#ifdef USE_ALPHA_BLENDING
    a = a * (2 - a);
#endif
    return a;
}
