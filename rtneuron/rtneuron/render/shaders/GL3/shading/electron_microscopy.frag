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

uniform sampler3D noise;

in vec3 normal, light, eye;
in vec3 position;

const vec3 specularColor = vec3(0.5, 0.5, 0.5);

const vec4 sampleCoef1 = vec4(4.0, 1.2, 2.4, 10.0);
const float sampleOffset1 = sampleCoef1[0] * 0.25 + sampleCoef1[1] * 0.125 +
                            sampleCoef1[2] * 0.0625 + sampleCoef1[3] * 0.03125;

const vec4 sampleCoef2 = vec4(6.0, 0.2, 0.2, 20.0);
const float sampleOffset2 = sampleCoef2[0] * 0.25 + sampleCoef2[1] * 0.125 +
                            sampleCoef2[2] * 0.0625 + sampleCoef2[3] * 0.03125;

float getBaseAlpha();
vec4 getBaseColor();

vec3 noiseDisplacement(vec3 position, vec4 sampleCoef, float offset)
{
    vec3 disp;
    vec4 noisevec;
    noisevec = texture(noise, position);
    disp.x = dot(noisevec, sampleCoef) - offset;
    noisevec = texture(noise, position + vec3(noisevec));
    disp.y = dot(noisevec, sampleCoef) - offset;
    noisevec = texture(noise, position - vec3(noisevec));
    disp.z = dot(noisevec, sampleCoef) - offset;
    return disp;
}

vec4 shadeFragment()
{
    vec3 n = normalize(normal);
#ifdef ACCURATE_HEADLIGHT
    vec3 l = normalize(light);
#else
    vec3 l = vec3(0, 0, 1);
#endif

    /* Ambient color */
    vec3 _color = vec3(0.05, 0.05, 0.05);

    /* Getting the base color */
    vec4 baseColor = getBaseColor();

    if (baseColor.a == 0.0)
        discard;

    float lambertTerm = abs(dot(n, l));
    lambertTerm = pow(lambertTerm, 2.0);

    vec3 disp = normalize(
        noiseDisplacement(position / 10.0, sampleCoef1, sampleOffset1));
    lambertTerm += dot(disp, l) * 0.12;

    disp = normalize(
        noiseDisplacement(position / 30.0, sampleCoef2, sampleOffset2));
    lambertTerm -= pow(dot(disp, l), 2.0) * 0.1;

    lambertTerm = max(0.1, 1.0 - lambertTerm);
    _color += baseColor.rgb * lambertTerm;
    return vec4(_color, baseColor.a);
}

float fragmentDepth()
{
    return gl_FragCoord.z;
}

float fragmentAlpha()
{
    return getBaseAlpha();
}
