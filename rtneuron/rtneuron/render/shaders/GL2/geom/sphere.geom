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

#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

$DEFINES

#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
#endif

#ifdef USE_RADIUS_UNIFORM
uniform float radius;
#else
varying in float radius[];
#endif

flat varying out vec4 sphere;
varying out vec3 eye;
#ifdef ACCURATE_HEADLIGHT
varying out vec3 light;
#endif

float cathetus(float hypot, float cat)
{
    return sqrt(hypot * hypot - cat * cat);
}

/**
   Returns the coordinates of the tangency point of a line passing by
   a point located at (d, 0) with d > 0 and tangent to a circle
   centered at origin and radius r.
 */
vec2 tangencyPoint(float radius, float distance)
{
    float a = (radius * radius) / distance;
    return vec2(a, cathetus(radius, a));
}

#ifndef USE_RADIUS_UNIFORM
#define radius radius[0]
#endif

void perspectiveMain()
{
    sphere = vec4(gl_PositionIn[0].xyz, radius);
    float quadDepth = sphere.z + radius;

    /* Left and right tangent lines */
    float d_floor = length(sphere.xz);
    vec2 t_floor = sphere.xz / d_floor;
    vec2 ortho_t_floor = vec2(t_floor.y, -t_floor.x);
    vec2 point = tangencyPoint(radius, d_floor);
    vec2 leftTangent = sphere.xz - t_floor * point.x + ortho_t_floor * point.y;
    vec2 rightTangent = sphere.xz - t_floor * point.x - ortho_t_floor * point.y;

    /* Top and bottom tangent lines */
    float d_side = length(sphere.yz);
    vec2 t_side = sphere.yz / d_side;
    vec2 ortho_t_side = vec2(t_side.y, -t_side.x);
    point = tangencyPoint(radius, d_side);
    vec2 topTangent = sphere.yz - t_side * point.x + ortho_t_side * point.y;
    vec2 bottomTangent = sphere.yz - t_side * point.x - ortho_t_side * point.y;

    vec4 corners[4];
    corners[0].x = corners[1].x = leftTangent.x * quadDepth / leftTangent.y;
    corners[2].x = corners[3].x = rightTangent.x * quadDepth / rightTangent.y;

    corners[0].y = corners[2].y = bottomTangent.x * quadDepth / bottomTangent.y;
    corners[1].y = corners[3].y = topTangent.x * quadDepth / topTangent.y;

    corners[0].zw = corners[1].zw = vec2(quadDepth, 1);
    corners[2].zw = corners[3].zw = vec2(quadDepth, 1);

    gl_FrontColor = gl_FrontColorIn[0];

#ifdef ACCURATE_HEADLIGHT
#define VERTEX(i) \
    eye = corners[i].xyz; \
    light = lightSource - eye; \
    gl_ClipVertex = vec4(eye, 1); \
    gl_Position = gl_ProjectionMatrix * corners[i]; \
    EmitVertex();
#else
#define VERTEX(i) \
    eye = corners[i].xyz; \
    gl_ClipVertex = vec4(eye, 1); \
    gl_Position = gl_ProjectionMatrix * corners[i]; \
    EmitVertex();
#endif
    VERTEX(0);
    VERTEX(1);
    VERTEX(2);
    VERTEX(3);
    EndPrimitive();
#undef VERTEX
}

void orthoMain()
{
    sphere = vec4(gl_PositionIn[0].xyz, radius);
    float quadDepth = sphere.z + radius;

    vec4 corners[4];
    corners[0].x = corners[1].x = sphere.x - radius;
    corners[2].x = corners[3].x = sphere.x + radius;

    corners[0].y = corners[2].y = sphere.y - radius;
    corners[1].y = corners[3].y = sphere.y + radius;

    corners[0].zw = corners[1].zw = vec2(quadDepth, 1);
    corners[2].zw = corners[3].zw = vec2(quadDepth, 1);

    gl_FrontColor = gl_FrontColorIn[0];

    /* ACCURATE_HEADLIGHT only makes sense in head-tracked environments.
       Since orthographics projections don't make sense there, we ignore it. */
#define VERTEX(i) \
    eye = corners[i].xyz; \
    gl_ClipVertex = vec4(eye, 1); \
    gl_Position = gl_ProjectionMatrix * corners[i]; \
    EmitVertex();
    VERTEX(0);
    VERTEX(1);
    VERTEX(2);
    VERTEX(3);
    EndPrimitive();
#undef VERTEX
}

void main()
{
    if (radius < 0)
        return;

#ifdef USE_ALPHA_BLENDING
    if (gl_FrontColorIn[0].a == 0)
       return;
#endif

    if (gl_ProjectionMatrix[2][3] != -1)
        orthoMain();
    else
        perspectiveMain();
}
