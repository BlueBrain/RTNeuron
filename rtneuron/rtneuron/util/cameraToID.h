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

#ifndef RTNEURON_CAMERATOID_H
#define RTNEURON_CAMERATOID_H

#include <stdint.h>

namespace osg
{
class Camera;
class RenderInfo;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   Returns a unique identifier for a camera suitable as an array index.

   ID are guanranteed to be low numbers so they can be used to index arrays.
   The set of ID for all cameras for which an ID has been requested is always
   starts at 0. If no camera has been destroyed the ID set is also compact.
   When a camera is destroyed its ID is freed to be reused for the next
   one for which and ID is requested and doesn't already have one.

   This function is thread-safe.
*/
uint32_t getUniqueCameraID(const osg::Camera* camera);

uint32_t getCameraAndEyeID(const osg::Camera* camera);
uint32_t getCameraAndEyeID(osg::RenderInfo& info);
}
}
}
#endif
