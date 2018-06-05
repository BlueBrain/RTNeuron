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

#ifndef RTNEURON_OSGEQ_COMPOSITOR_H
#define RTNEURON_OSGEQ_COMPOSITOR_H

#include <eq/types.h>
#include <eq/util/types.h>

#include <boost/function.hpp>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Channel;

typedef boost::function<int(const eq::Frame&)> FramePositionFunctor;

namespace Compositor
{
/**
   \brief An assembly function similar to eq::compositor::assembleFrames
   @param frames The frames to assemble
   @param accum An accumulation buffer for idle AA and subpixel compounds.
          May be nil. A buffer will be allocated for the given channel
          if needed.
   @param channel The destination channel. Its viewport, buffer and assembly
          state must be already applied.
   @param framePosition A function object that returns the position in which
          a frame must be composited (starting from 0) or -1 if the position
          doesn't matter.
 */
unsigned int assembleFrames(
    const eq::Frames& frames, eq::util::Accum* accum, Channel* channel,
    const FramePositionFunctor& framePosition = FramePositionFunctor());
}
}
}
}
#endif
