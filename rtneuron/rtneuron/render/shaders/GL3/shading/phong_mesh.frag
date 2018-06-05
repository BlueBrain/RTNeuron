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
#extension GL_EXT_geometry_shader4 : enable

$DEFINES

#ifdef ACCURATE_HEADLIGHT
in vec3 light;
#endif

in vec3 normal, eye;

vec4 phong(const vec4 color, const vec3 normal, const vec3 eye,
           const vec3 light);

vec4 getBaseColor();
float getBaseAlpha();

vec4 shadeFragment()
{
    vec3 n = normalize(normal);
    vec3 e = normalize(eye);
#ifdef ACCURATE_HEADLIGHT
    vec3 l = normalize(light);
#else
    const vec3 l = vec3(0.0, 0.0, 1.0);
#endif
    return phong(getBaseColor(), n, e, l);
}

float fragmentDepth()
{
    return gl_FragCoord.z;
}

float fragmentAlpha()
{
    return getBaseAlpha();
}
