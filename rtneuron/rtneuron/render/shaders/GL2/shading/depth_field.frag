/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
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
#extension GL_ARB_gpu_shader5 : enable

$DEFINES

uniform sampler2D depth;
uniform sampler2D color;
uniform sampler2D blur;

uniform float focalDistance;
uniform float focalRange;
uniform mat4 projectionMatrix;

varying vec2 uv;

float unproject(float x)
{
    return -projectionMatrix[3][2] / (projectionMatrix[2][2] + 2.0 * x - 1.0);
}

float getBlurFromDepth(vec2 uv)
{
    vec3 ndcPos = vec3(uv, texture2D(depth, uv).x);
    ndcPos -= 0.5;
    ndcPos *= 2.0;
    vec4 eyePos = inverse(projectionMatrix) * vec4(ndcPos, 1);

    // get fragment world Z-coordinate
    float z = -unproject(ndcPos.z);

    // distance from fragment to eye
    float distance = sqrt(pow(-eyePos.x, 2) + pow(-eyePos.y, 2) + pow(z, 2));

    // focal distances (beginning and end of area on focus)
    float p1 = focalDistance - focalRange * 0.5;
    float p2 = focalDistance + focalRange * 0.5;

    // blur factor
    float output;
    if (distance < focalDistance)
        output = (distance - p1) * (-1 / (focalDistance - p1)) + 1;
    else
        output = (distance - focalDistance) * (1 / (p2 - focalDistance));

    return clamp(output, 0.0, 1.0);
}

void main()
{
    vec4 fullColor = texture2D(color, uv);
    vec4 blurColor = texture2D(blur, uv);
    float blurValue = getBlurFromDepth(uv);
    gl_FragColor = fullColor + blurValue * (blurColor - fullColor);
}
