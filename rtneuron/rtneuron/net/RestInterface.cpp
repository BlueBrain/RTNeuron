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

#include "RestInterface.h"

#include "../Camera.h"
#include "../View.h"
#include "../util/math.h"
#include "../util/vec_to_vec.h"

#include <lunchbox/log.h>
#include <lunchbox/monitor.h>

#include <osg/Matrix>

#include <zeroeq/http/server.h>
#include <zeroeq/zeroeq.h>

#include <lexis/render/imageJPEG.h>
#include <lexis/render/lookOut.h>

#include <thread>

namespace bbp
{
namespace rtneuron
{
namespace net
{
namespace
{
const float DEFAULT_HEARTBEAT_TIME = 1000.0f;

typedef std::unique_ptr<zeroeq::http::Server> ServerPtr;

ServerPtr createServer(const int argc, char** argv)
{
    ServerPtr server = zeroeq::http::Server::parse(argc, argv);
    if (!server)
        LBTHROW(std::runtime_error("REST interface could not initialized"));
    return server;
}
}

/**
   This class is non thread-safe, that means that all functions are considered
   to be called from the same processing loop, which in particular is the
   Equalizer rendering loop.
*/
class RestInterface::Impl
{
public:
    Impl(const int argc, char** argv, const ViewPtr& view)
        : _server(createServer(argc, argv))
        , _view(view)
    {
        _lookOut.registerDeserializedCallback(
            boost::bind(&Impl::_onLookOutChanged, this));
        _server->handlePUT(_lookOut);
        _view.lock()->frameGrabbed.connect(
            boost::bind(&Impl::_onGrabFrame, this, _1));
        _frame.registerSerializeCallback(
            boost::bind(&Impl::_onFrameRequested, this));
        _server->handleGET(_frame);
        /* The server is going to be used from this thread. The memeory barrier
           implied by thread creation guarantees that the internal ZeroMQ
           sockets are used safely. */
        _receiveThread = std::thread(&Impl::_run, this);
    }

    ~Impl()
    {
        if (!_view.expired())
            _view.lock()->frameGrabbed.disconnect(
                boost::bind(&Impl::_onGrabFrame, this, _1));
        _view.reset();
        _receiveThread.join();
    }

private:
    void _run()
    {
        while (!_view.expired())
            _server->receive(100);
    }

    void _onGrabFrame(const std::vector<uint8_t>& data)
    {
        _frame.setData(data);
        ++_monitor;
    }

    void _onFrameRequested()
    {
        if (_view.expired())
            return;

        const unsigned int current = _monitor.get();
        _view.lock()->grabFrame();

        _monitor.waitGE(current + 1);
    }

    void _onLookOutChanged()
    {
        brion::Vector3d position;
        brion::Vector4d orientation;
        core::decomposeMatrix(osg::Matrixd(_lookOut.getMatrix()), position,
                              orientation);
        const CameraPtr& camera = _view.lock()->getCamera();
        camera->setViewNoDirty(brion::Vector3f(position) * 1000000.,
                               brion::Vector4f(orientation));
    }

    ServerPtr _server;
    std::weak_ptr<View> _view;

    lexis::render::LookOut _lookOut;
    lexis::render::ImageJPEG _frame;
    lunchbox::Monitor<unsigned int> _monitor;

    std::thread _receiveThread;
};

RestInterface::RestInterface(const int argc, char** argv, const ViewPtr& view)
    : _impl(new Impl(argc, argv, view))
{
}

RestInterface::~RestInterface()
{
    delete _impl;
}
}
}
}
