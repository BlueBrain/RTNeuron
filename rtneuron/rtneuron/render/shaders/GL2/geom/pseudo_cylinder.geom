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

$DEFINES

#ifdef USE_CONTINUOUS_LOD
uniform float clodThreshold;
#endif
#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
#endif

varying in vec4 colorIn[];
varying in float reportedVariableIn[];
varying in vec3 tangentIn[];
varying in float radiusIn[];

#ifdef FASTER_PSEUDOCYLINDER_NORMALS
varying out vec4 tangentAndOffset;
#else
varying out vec3 normal;
#endif
varying out vec3 eye;
#ifdef ACCURATE_HEADLIGHT
varying out vec3 light;
#endif
varying out float reportedVariable;
varying out vec4 color;


void main()
{
#ifdef USE_CONTINUOUS_LOD
    if (length(gl_PositionIn[0].xyz) / radiusIn[0] <= clodThreshold ||
        length(gl_PositionIn[1].xyz) / radiusIn[1] <= clodThreshold)
    {
        return;
    }
#endif

#ifdef FASTER_PSEUDOCYLINDER_NORMALS
    const float flattening = 1.0;
    /* find the displacement vector */
    for (int i = 0; i < 2; ++i)
    {
        vec4 center = gl_PositionIn[i];
        vec3 y = normalize(cross(-center.xyz, tangentIn[i]));
        vec3 x = normalize(cross(tangentIn[i], y));
        vec4 translation = vec4(y * radiusIn[i], 0.0);
        color = colorIn[i];
        reportedVariable = reportedVariableIn[i];
        tangentAndOffset.xyz = tangentIn[i];

        gl_ClipVertex = center + translation;
        gl_Position = gl_ProjectionMatrix * gl_ClipVertex;
        eye = -gl_ClipVertex.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = gl_LightSource[0].position.xyz + eye;
#endif
        tangentAndOffset.w = 1.0;
        EmitVertex();

        /* The same for the next corner */
        gl_ClipVertex = center - translation;
        gl_Position = gl_ProjectionMatrix * gl_ClipVertex;
        eye = -gl_ClipVertex.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        tangentAndOffset.w = -1.0;
        EmitVertex();
    }
    EndPrimitive();

#else // FASTER_PSEUDOCYLINDER_NORMALS

    vec3 y[2];
    vec3 x[2];
    vec2 n2d[2];

    y[0] = normalize(cross(gl_PositionIn[0].xyz, tangentIn[0]));
    x[0] = normalize(cross(y[0], tangentIn[0]));
    n2d[0] = vec2(0.5, 1.0);

    y[1] = normalize(cross(gl_PositionIn[1].xyz, tangentIn[1]));
    x[1] = normalize(cross(y[1], tangentIn[1]));
    n2d[1] = vec2(0.5, 1.0);

    {
        color = colorIn[0];
        reportedVariable = reportedVariableIn[0];
        vec4 clip = gl_PositionIn[0] + vec4(y[0] * radiusIn[0], 0.0);
        gl_ClipVertex = clip;
        gl_Position = gl_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        normal = normalize(n2d[0].x * x[0] + n2d[0].y * y[0]);
        EmitVertex();
    }

    {
        color = colorIn[1];
        reportedVariable = reportedVariableIn[1];
        vec4 clip = gl_PositionIn[1]  + vec4(y[1] * radiusIn[1], 0.0);
        gl_ClipVertex = clip;
        gl_Position = gl_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        normal = normalize(n2d[1].x * x[1] + n2d[1].y * y[1]);
        EmitVertex();
    }

    {
        color = colorIn[0];
        reportedVariable = reportedVariableIn[0];
        vec4 clip = gl_PositionIn[0] + vec4(x[0] * radiusIn[0], 0.0);
        gl_ClipVertex = clip;
        gl_Position = gl_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        normal = x[0];
        EmitVertex();
    }

    {
        color = colorIn[1];
        reportedVariable = reportedVariableIn[1];
        vec4 clip = gl_PositionIn[1]  + vec4(x[1] * radiusIn[1], 0.0);
        gl_ClipVertex = clip;
        gl_Position = gl_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        normal = x[1];
        EmitVertex();
    }

    {
        color = colorIn[0];
        reportedVariable = reportedVariableIn[0];
        vec4 clip = gl_PositionIn[0] - vec4(y[0] * radiusIn[0], 0.0);;
        gl_ClipVertex = clip;
        gl_Position = gl_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        normal = normalize(n2d[0].x * x[0] - n2d[0].y * y[0]);
        EmitVertex();
    }

    {
        color = colorIn[1];
        reportedVariable = reportedVariableIn[1];
        vec4 clip = gl_PositionIn[1]  - vec4(y[1] * radiusIn[1], 0.0);
        gl_ClipVertex = clip;
        gl_Position = gl_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        normal = normalize(n2d[1].x * x[1] - n2d[1].y * y[1]);
        EmitVertex();
    }

    EndPrimitive();
#endif
}
