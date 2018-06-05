/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Daniel Nachbaur <danielnachbaur@googlemail.com>
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

#ifndef RTNEURON_API_NET_RESTINTERFACE_H
#define RTNEURON_API_NET_RESTINTERFACE_H

#include "../types.h"

#include <lunchbox/types.h>

#include <boost/noncopyable.hpp>

namespace bbp
{
namespace rtneuron
{
namespace net
{
/**
  Setup a REST interface which uses ZeroEQ for communication.

  Supported incoming events:
    - EVENT_CAMERA
    - EVENT_REQUEST

  Supported outgoing events:
    - EVENT_HEARTBEAT
    - EVENT_IMAGEJPEG
    - EVENT_VOCABULARY
 */
class RestInterface : boost::noncopyable
{
public:
/**
  Setup HTTP server on the given hostname and port and receive
  events associated from the given view.

  The argument list is passed directly to the zeroeq::http::Server object
  created internally.
 */
#ifndef DOXYGEN_TO_BREATHE
    RestInterface(int argc, char** argv, const ViewPtr& view);
#else
    RestInterface(list args, const ViewPtr& view);
#endif

    ~RestInterface();

private:
    class Impl;
    Impl* const _impl;
};
}
}
}

#endif
