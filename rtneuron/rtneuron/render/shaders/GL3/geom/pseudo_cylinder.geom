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

uniform mat4x4 osg_ProjectionMatrix;
uniform mat4x4 osg_ModelViewMatrix;
#ifdef USE_CONTINUOUS_LOD
uniform float clodThreshold;
#endif
#ifdef ACCURATE_HEADLIGHT
uniform vec3 lightSource;
#endif

in float reportedVariableIn[];
in vec3 tangentIn[];
in float radiusIn[];
in vec4 colorIn[];

out vec4 color;
#ifdef FASTER_PSEUDOCYLINDER_NORMALS
out vec4 tangentAndOffset;
#else
out vec3 normal;
#endif
out vec3 eye;
#ifdef ACCURATE_HEADLIGHT
out vec3 light;
#endif
out float reportedVariable;

void computeClipDistances(const vec4 vertex);

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
        vec4 translation = vec4(y * radiusIn[i], 0);
        color = colorIn[i];
        reportedVariable = reportedVariableIn[i];
        tangentAndOffset.xyz = tangentIn[i];

        vec4 clip = center + translation;
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        tangentAndOffset.w = 1;
        EmitVertex();

        /* The same for the next corner */
        clip = center - translation;
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
        eye = -clip.xyz;
#ifdef ACCURATE_HEADLIGHT
        /* calculate the light direction. */
        light = lightSource + eye;
#endif
        tangentAndOffset.w = -1;
        EmitVertex();
    }
    EndPrimitive();

#else // FASTER_PSEUDOCYLINDER_NORMALS

    vec3 y[2];
    vec3 x[2];
    vec2 n2d[2];

    for (int i = 0; i < 2; ++i)
    {
        y[i] = normalize(cross(gl_PositionIn[i].xyz, tangentIn[i]));
        x[i] = normalize(cross(y[i], tangentIn[i]));
        float d = abs(dot(x[i], gl_PositionIn[i].xyz));
        n2d[i] = vec2(0.5, 1);
    }

    {
        color = colorIn[0];
        reportedVariable = reportedVariableIn[0];
        vec4 clip = gl_PositionIn[0] + vec4(y[0] * radiusIn[0], 0);
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
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
        vec4 clip = gl_PositionIn[1]  + vec4(y[1] * radiusIn[1], 0);
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
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
        vec4 clip = gl_PositionIn[0] + vec4(x[0] * radiusIn[0], 0);
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
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
        vec4 clip = gl_PositionIn[1]  + vec4(x[1] * radiusIn[1], 0);
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
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
        vec4 clip = gl_PositionIn[0] - vec4(y[0] * radiusIn[0], 0);;
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
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
        vec4 clip = gl_PositionIn[1]  - vec4(y[1] * radiusIn[1], 0);
        computeClipDistances(clip);
        gl_Position = osg_ProjectionMatrix * clip;
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
