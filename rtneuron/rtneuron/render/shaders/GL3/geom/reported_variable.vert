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

#extension GL_EXT_gpu_shader4 : enable

uniform samplerBuffer compartmentsBuffer;
uniform int compartmentBufferObjectOffset;
/* This variable is converted from short (the type used in the C++ side)
   to float by the API. */
in float bufferOffsetsAndDelays;

#ifdef SHOW_SPIKES
uniform int cellIndex;
uniform samplerBuffer spikesBuffer;
uniform float timestamp;
uniform float spikeTail;

const float UNDEFINED = 1.0 / 0.0; /* This generates +infinity */
float cachedValue = UNDEFINED;
#endif

float getReportedVariable()
{
#ifdef SHOW_SPIKES
    if (cachedValue != UNDEFINED)
        return cachedValue;

    if (bufferOffsetsAndDelays >= 0)
    {
        int index = compartmentBufferObjectOffset + int(bufferOffsetsAndDelays);
        cachedValue = 128.0 + texelFetchBuffer(compartmentsBuffer, index)[0];
        return cachedValue;
    }

    float delay = -(bufferOffsetsAndDelays + 1) / 1024.0;
    int index = cellIndex * 5;
    /* Action potential */
    float value = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        float spikeTime = texelFetchBuffer(spikesBuffer, index + i)[0];
        spikeTime += delay;
        float d = timestamp - spikeTime;
        if (d >= 0.0 && d < spikeTail)
        {
            value = -(1.0 - d * 1 / spikeTail);
            break;
        }
    }
    cachedValue = value;
    return value;
#else
    /* Compartimental data. */
    int index = compartmentBufferObjectOffset + int(bufferOffsetsAndDelays);
    return texelFetchBuffer(compartmentsBuffer, index)[0];
#endif
}
