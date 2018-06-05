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

#ifndef RTNEURON_OSGEQ_CONFIGEVENT_H
#define RTNEURON_OSGEQ_CONFIGEVENT_H

#include <eq/types.h>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
namespace ConfigEvent
{
enum Type
{
    REQUEST_REDRAW = eq::EVENT_USER,

    /** Almost the same as a request redraw, but it cannot
        be masked out by the redraw mask. */
    FORCE_REDRAW,

    /** Update the request redraw mask.
        Maskable redraw requests are all those signalled using REQUEST_REDRAW.
        FORCE_REDRAW is used for unmaskable requests.
        A change in the redraw mask implies a call to Config::finishAllFrames()

        Command data: bool */
    REDRAW_MASK,

    FINISH_ALL_FRAMES,

    /** Command data: int32_t */
    IDLE_AA_LEFT,

    /** Requests a call to Config::update

        Command data:
        - bool, If true the request will be served
        - [uint32_t], The request ID to serve the request. Provided only if
          the first parameter is true.

        The request type is lunchbox::Request<bool>. The return type indicates
        if the reconfiguration was successful. */
    RECONFIGURE_REQUEST,

    /** Announce update of texture containing rendering of current frame; to
        be used by GUI

        Command data: unsigned */
    ACTIVE_VIEW_TEXTURE,

    /** Announce creation of a new offscreen surface and the QObject to which
        the GUI has to send interaction events.

        Command data: QObject* */
    ACTIVE_VIEW_EVENT_PROCESSOR,

    /** Announce a grabbed frame encoded in JPEG when View::setGrabFrameToSignal
       is set

        Command data:
        - uint128_t, ViewID
        - uint64_t: JPEG image size
        - uint8_t*: JPEG image data */
    GRAB_FRAME,

    APPLICATION_EVENT
};
}
}
}
}

#endif
