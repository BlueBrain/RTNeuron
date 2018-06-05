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

#extension GL_EXT_gpu_shader4 : enable

$DEFINES

uniform samplerBuffer compartmentsBuffer;

/** This may be changed to an integer array if support for OSG < 3.2 is
    dropped.
    @sa BBPRTN-379 */
in float cellIndex;

#ifdef SHOW_SPIKES
uniform samplerBuffer spikesBuffer;
uniform float timestamp;
uniform float spikeTail;
#endif

float getReportedVariable()
{
#ifdef SHOW_SPIKES
    /* Action potential */
    int index = int(cellIndex) * 5;

    /* There's no need to check any other spike in the buffer. The host
       code must ensure that the most recent spike which is smaller than
       the current timestamp is at the front of the spike buffer
       (regardless of playback going backward or forward). */
    float spikeTime = texelFetchBuffer(spikesBuffer, index)[0];
    float d = timestamp - spikeTime;
    if (d >= 0.0 && d < spikeTail)
    {
        return -(1.0 - d * 1 / spikeTail);
    }
    return 0;
#else
    /* Compartmental data. */
    return texelFetchBuffer(compartmentsBuffer, int(cellIndex))[0];
#endif
}
