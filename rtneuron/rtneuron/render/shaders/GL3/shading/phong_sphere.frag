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

in vec4 color;

#define MULTISAMPLE

$DEFINES

uniform mat4x4 osg_ProjectionMatrix;

flat in vec4 sphere;
#ifdef MULTISAMPLE
sample in vec3 eye;
#else
in vec3 eye;
#endif
#ifdef ACCURATE_HEADLIGHT
in vec3 light;
#endif

float depthCache;
bool solved = false;
vec3 normalized_eye;
vec3 normal;

vec4 phong(const vec4 color, const vec3 normal, const vec3 eye,
           const vec3 light);

float sphereLineTest(vec3 c, float r, vec3 d)
{
    /* Real Time Rendering pg. 571 */
    float s = dot(c, d);
    float c2 = dot(c, c);
    float r2 = r * r;
    if (s < 0 && c2 > r2)
        return -1;
    float m2 = c2 - s * s;
    if (m2 > r2)
        return -1;
    float q = sqrt(r2 - m2);

    return (c2 > r2 ? s - q : s + q);
}

void solveOrtho()
{
    vec2 xy = eye.xy - sphere.xy;
    float R2 = sphere.w * sphere.w;
    float r2 = dot(xy, xy);
    if (r2 > R2)
        discard;
    float a2 = r2 / R2;
    float z = sqrt(1 - a2);

    normal.xy = xy / sphere.w;
    normal.z = z;
    normalized_eye = vec3(0, 0, -1);

    float depth = eye.z - (1 - z) * sphere.w;
    depth = (osg_ProjectionMatrix[2][2] * depth + osg_ProjectionMatrix[3][2]);
    depthCache = (depth + 1) * 0.5;
}

void solvePerspective()
{
    vec3 e = normalize(eye);
    float t = sphereLineTest(sphere.xyz, sphere.w, e);
    if (t == -1)
        discard;
    normal = normalize(e * t - sphere.xyz);
    normalized_eye = e;
    /* Computing a new depth value in [0, 1] based on the position of the ray
       sphere intersection. */
    depthCache = 0.5 * (1.0 - osg_ProjectionMatrix[2][2] -
                        osg_ProjectionMatrix[3][2] / (e.z * t));
}

void solveIntersection()
{
    if (solved)
        return;

    if (gl_ProjectionMatrix[2][3] == -1)
        solvePerspective();
    else
        solveOrtho();
    solved = true;
}

vec4 shadeFragment()
{
#ifdef ACCURATE_HEADLIGHT
    vec3 l = normalize(light);
#else
    vec3 l = vec3(0, 0, 1);
#endif
    solveIntersection();

    vec4 outColor = phong(color, normal, -normalized_eye, l);
#ifdef USE_ALPHA_BLENDING
    outColor.a = outColor.a * (2 - outColor.a);
#endif
    return outColor;
}

float fragmentDepth()
{
    solveIntersection();

    return depthCache;
}

float fragmentAlpha()
{
    solveIntersection();

    float a = color.a;
#ifdef USE_ALPHA_BLENDING
    a = a * (2 - a);
#endif
    return a;
}
