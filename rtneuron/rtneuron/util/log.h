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

#ifndef RTNEURON_UTIL_LOG_H
#define RTNEURON_UTIL_LOG_H

#include <eq/log.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
enum LogTopics
{
    LOG_SIMULATION_PLAYBACK = eq::LOG_CUSTOM,  // 65536
    LOG_SCENE_UPDATES = eq::LOG_CUSTOM << 1,   // 131072
    LOG_FRAME_RECORDING = eq::LOG_CUSTOM << 2, // 262144
    LOG_TRACKER_DEVICES = eq::LOG_CUSTOM << 3  // 524288
};
}
}
}
#endif
