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

#ifndef RTNEURON_API_NET_CAMERABROKER_H
#define RTNEURON_API_NET_CAMERABROKER_H

#include "../types.h"

#include <zeroeq/uri.h>

#include <boost/noncopyable.hpp>

namespace bbp
{
namespace rtneuron
{
namespace net
{
namespace detail
{
class CameraBroker;
}

/**
   Subscriber/publisher for camera matrix events.

   Camera broker that publishes camera events to other applications
   subscribed to it, and can also susbcribe to others to receive events from
   them. The camera event will contain only the camera matrix.

   By synchronizing with others, every update in the publisher application's
   camera will be sent to the subscribers, allowing them to set their camera
   parameters so all of them are in the same state.
*/
class CameraBroker : boost::noncopyable
{
public:
    friend class Testing;

    /**
       Creates a broker to send and receive camera events between this
       and other applications.

       The default ZeroEQ session will be used

       @param camera The application camera that will be synchronized with
              others.
       @version 2.10
     */
    CameraBroker(const CameraPtr& camera);

    /**
       Creates a broker to send and receive camera events between this
       and other applications.

       The broker will use the default ZeroEQ session.
       @param session The ZeroEQ session to use for the publisher and
                      subscriber
       @param camera The application camera that will be synchronized with
              others.
       @version 2.10
     */
    CameraBroker(const std::string& session, const CameraPtr& camera);

    /**
       Creates a broker to send and receive camera events between this
       and other applications.
       @param publisher The URI that the object will use to publish the
              camera events.
       @param subscriber The URI that the object will subscribe to in order
              to receive camera events from other publishers.
       @param camera The application camera that will be synchronized with
              others.
       @version 2.10
     */
    CameraBroker(const zeroeq::URI& publisher, const zeroeq::URI& subscriber,
                 const CameraPtr& camera);

    ~CameraBroker();

    /* @internal For testing purposes. */
    zeroeq::URI getURI() const;

private:
    detail::CameraBroker* const _impl;
};
}
}
}
#endif
