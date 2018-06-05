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

#ifdef USE_CONTINUOUS_LOD
uniform float clodThreshold;
#endif

#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
#endif

varying in vec4 colorIn[];
varying in float reportedVariableIn[];
varying in float radius[];
varying in vec4 cutPlane[];

flat varying vec4 c1;
flat varying vec4 c2;
flat varying out vec4 color1;
flat varying out vec4 color2;
flat varying out vec4 planes[2];
#ifdef ACCURATE_HEADLIGHT
varying out vec3 light;
#endif

varying out float reportedVariable;
varying out vec3 eye;

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

void main()
{
#ifdef USE_CONTINUOUS_LOD
    if (length(gl_PositionIn[0].xyz) / radius[0] > clodThreshold &&
        length(gl_PositionIn[1].xyz) / radius[1] > clodThreshold)
    {
        return;
    }
#endif

    c1 = vec4(gl_PositionIn[0].xyz, radius[0]);
    c2 = vec4(gl_PositionIn[1].xyz, radius[1]);
    color1 = colorIn[0];
    color2 = colorIn[1];
    planes[0] = cutPlane[0];
    planes[1] = -cutPlane[1];

    float offset = 1.4242135623730951;

    vec4 cornerRT1;
    vec4 cornerLB1;
    float left1, right1, bottom1, top1;
    float depth1;
    {
        depth1 = c1.z + radius[0];
        float d_floor = length(c1.xz);
        vec2 t_floor = c1.xz / d_floor;
        vec2 ortho_t_floor = vec2(t_floor.y, -t_floor.x);
        vec2 point = tangencyPoint(radius[0], d_floor);
        vec2 leftTangent = c1.xz - t_floor * point.x + ortho_t_floor * point.y;
        vec2 rightTangent = c1.xz - t_floor * point.x - ortho_t_floor * point.y;

        /* Top and bottom tangent lines */
        float d_side = length(c1.yz);
        vec2 t_side = c1.yz / d_side;
        vec2 ortho_t_side = vec2(t_side.y, -t_side.x);
        point = tangencyPoint(radius[0], d_side);
        vec2 topTangent = c1.yz - t_side * point.x - ortho_t_side * point.y;
        vec2 bottomTangent = c1.yz - t_side * point.x + ortho_t_side * point.y;

        left1 = leftTangent.x * depth1 / leftTangent.y;
        right1 = rightTangent.x * depth1 / rightTangent.y;
        bottom1 = bottomTangent.x * depth1 / bottomTangent.y;
        top1 = topTangent.x * depth1 / topTangent.y;

        cornerRT1 = gl_ProjectionMatrix * vec4(right1, top1, depth1, 1);
        cornerRT1.xyz /= cornerRT1.w;
        cornerLB1 = gl_ProjectionMatrix * vec4(left1, bottom1, depth1, 1);
        cornerLB1.xyz /= cornerLB1.w;
    }

    vec4 cornerRT2;
    vec4 cornerLB2;
    float left2, right2, bottom2, top2;
    float depth2;
    {
        depth2 = c2.z + radius[1];
        float d_floor = length(c2.xz);
        vec2 t_floor = c2.xz / d_floor;
        vec2 ortho_t_floor = vec2(t_floor.y, -t_floor.x);
        vec2 point = tangencyPoint(radius[1], d_floor);
        vec2 leftTangent = c2.xz - t_floor * point.x + ortho_t_floor * point.y;
        vec2 rightTangent = c2.xz - t_floor * point.x - ortho_t_floor * point.y;

        /* Top and bottom tangent lines */
        float d_side = length(c2.yz);
        vec2 t_side = c2.yz / d_side;
        vec2 ortho_t_side = vec2(t_side.y, -t_side.x);
        point = tangencyPoint(radius[1], d_side);
        vec2 topTangent = c2.yz - t_side * point.x - ortho_t_side * point.y;
        vec2 bottomTangent = c2.yz - t_side * point.x + ortho_t_side * point.y;

        left2 = leftTangent.x * depth2 / leftTangent.y;
        right2 = rightTangent.x * depth2 / rightTangent.y;
        bottom2 = bottomTangent.x * depth2 / bottomTangent.y;
        top2 = topTangent.x * depth2 / topTangent.y;

        cornerRT2 = gl_ProjectionMatrix * vec4(right2, top2, depth2, 1);
        cornerRT2.xyz /= cornerRT2.w;
        cornerLB2 = gl_ProjectionMatrix * vec4(left2, bottom2, depth2, 1);
        cornerLB2.xyz /= cornerLB2.w;
    }

    vec4 corners[6];

    if (cornerLB1.y  > cornerLB2.y)
    {
        {
            vec2 tmp;
            tmp = cornerRT2.xy;
            cornerRT2.xy = cornerRT1.xy;
            cornerRT1.xy = tmp;
            tmp = cornerLB2.xy;
            cornerLB2.xy = cornerLB1.xy;
            cornerLB1.xy = tmp;
        }
        {
            vec4 tmp;
            tmp = vec4(left1, right1, top1, bottom1);
            left1 = left2;
            right1 = right2;
            top1 = top2;
            bottom1 = bottom2;
            left2 = tmp.x;
            right2 = tmp.y;
            top2 = tmp.z;
            bottom2 = tmp.w;
        }
        {
            float tmp = depth1;
            depth1 = depth2;
            depth2 = tmp;
        }
    }

    /* Bottom right corner */
    corners[0] = vec4(left1, bottom1, depth1, 1);
    corners[1] = vec4(right1, bottom1, depth1, 1);

    /* Checking special cases in which one square is at both the top and the
       bottom. */
    if (cornerRT1.y > cornerRT2.y)
    {
        if (cornerRT1.x > cornerRT2.x)
        {
            corners[2] = vec4(right1, top1, depth1, 1);
            /* Largest square is the first and is at the right */
            corners[3] = vec4(left1, top1, depth1, 1);
            if (cornerLB1.x < cornerLB2.x)
            {
                /* The second square contains the first one. */
                corners[4] = corners[5] = corners[0];
            }
            else
            {
                corners[4] = vec4(left2, top2, depth2, 1);
                corners[5] = vec4(left2, bottom2, depth2, 1);
            }
        }
        else
        {
            /* Largest square is the first and is at the left */
            corners[2] = vec4(right2, bottom2, depth2, 1);
            corners[3] = vec4(right2, top2, depth2, 1);
            corners[4] = vec4(right1, top1, depth1, 1);
            corners[5] = vec4(left1, top1, depth1, 1);
        }
    }
    else
    {
        corners[3] = vec4(right2, top2, depth2, 1);
        corners[4] = vec4(left2, top2, depth2, 1);
        if (cornerRT1.x > cornerRT2.x)
            corners[2] = vec4(right1, top1, depth1, 1);
        else
            corners[2] = vec4(right2, bottom2, depth2, 1);

        if (cornerLB2.x < cornerLB1.x)
            corners[5] = vec4(left2, bottom2, depth2, 1);
        else
            corners[5] = vec4(left1, top1, depth1, 1);
    }

#ifdef ACCURATE_HEADLIGHT
#define VERTEX(i) \
    eye = corners[i].xyz; \
    gl_ClipVertex = vec4(eye, 1); \
    light = lightSource - eye; \
    gl_Position = gl_ProjectionMatrix * corners[i]; \
    EmitVertex();
#else
#define VERTEX(i) \
    eye = corners[i].xyz; \
    gl_ClipVertex = vec4(eye, 1); \
    gl_Position = gl_ProjectionMatrix * corners[i]; \
    EmitVertex();
#endif

    /** \bug Assign a different value to each vertex */
    reportedVariable = reportedVariableIn[0];

    VERTEX(0);
    VERTEX(1);
    VERTEX(5);
    VERTEX(2);
    VERTEX(4);
    VERTEX(3);
    EndPrimitive();
}
