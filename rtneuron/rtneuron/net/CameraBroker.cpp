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

#include "CameraBroker.h"
#include "../Camera.h"

#include <util/math.h>
#include <util/vec_to_vec.h>

#include <lexis/lexis.h>
#include <zeroeq/zeroeq.h>

#include <osg/Matrix>

#include <thread>

namespace bbp
{
namespace rtneuron
{
namespace net
{
namespace detail
{
class CameraBroker
{
public:
    CameraBroker(const CameraPtr& camera)
        : _camera(camera)
    {
        _init();
    }

    CameraBroker(const std::string& session, const CameraPtr& camera)
        : _camera(camera)
        , _publisher(session)
        , _subscriber(session)
    {
        _init();
    }

    CameraBroker(const zeroeq::URI& publisherURI,
                 const zeroeq::URI& subscriberURI, const CameraPtr& camera)
        : _camera(camera)
        , _publisher(publisherURI)
        , _subscriber(subscriberURI)
    {
        _init();
    }

    ~CameraBroker()
    {
        if (!_camera.expired())
            _camera.lock()->viewDirty.disconnect(
                boost::bind(&CameraBroker::_publish, this));
        _camera.reset();

        _receiveThread.join();
    }

    zeroeq::URI getURI() const { return _publisher.getURI(); }
private:
    std::weak_ptr<rtneuron::Camera> _camera;

    std::thread _receiveThread;
    zeroeq::Publisher _publisher;
    zeroeq::Subscriber _subscriber;

    void _init()
    {
        _camera.lock()->viewDirty.connect(
            boost::bind(&CameraBroker::_publish, this));
        _subscriber.subscribe(lexis::render::LookOut::ZEROBUF_TYPE_IDENTIFIER(),
                              [&](const void* data, const size_t size) {
                                  _onUpdate(
                                      lexis::render::LookOut::create(data,
                                                                     size));
                              });
        _receiveThread = std::thread(&CameraBroker::_poll, this);
    }

    void _publish()
    {
        brion::Vector3f position;
        Orientation orientation;

        const CameraPtr& camera = _camera.lock();
        camera->getView(position, orientation);

        const std::vector<double>& cameraMatrix =
            _composeMatrix(position, orientation);

        _publisher.publish(lexis::render::LookOut(cameraMatrix));
    }

    void _onUpdate(const lexis::render::ConstLookOutPtr& event)
    {
        const CameraPtr& camera = _camera.lock();
        if (!camera)
            return;

        brion::Vector3f position;
        Orientation orientation;
        core::decomposeMatrix(osg::Matrixf(event->getMatrix()), position,
                              orientation);
        position *= 1000000.;
        camera->setViewNoDirty(position, orientation);
    }

    void _poll()
    {
        while (!_camera.expired())
            _subscriber.receive(100);
    }

    std::vector<double> _composeMatrix(const brion::Vector3f& position,
                                       const Orientation& orientation)
    {
        osg::Matrixd translation, rotation, cameraMatrix;
        translation.makeTranslate(core::vec_to_vec(position / 1000000.));
        rotation.makeRotate(orientation.w() * M_PI / 180, orientation.x(),
                            orientation.y(), orientation.z());

        cameraMatrix.mult(rotation, translation);
        cameraMatrix.invert(cameraMatrix);

        /* OpenSceneGraph stores the matrix in row-major order, but uses
           inverse multiply notation. If we want to return a column-major
           matrix to be used using the standard notation, we just need to
           copy the raw data as is */
        const double* values = cameraMatrix.ptr();
        return std::vector<double>(values, values + 16);
    }
};
}

CameraBroker::CameraBroker(const CameraPtr& camera)
    : _impl(new detail::CameraBroker(camera))
{
}

CameraBroker::CameraBroker(const std::string& session, const CameraPtr& camera)
    : _impl(new detail::CameraBroker(session, camera))
{
}

CameraBroker::CameraBroker(const zeroeq::URI& publisherURI,
                           const zeroeq::URI& subscriberURI,
                           const CameraPtr& camera)
    : _impl(new detail::CameraBroker(publisherURI, subscriberURI, camera))
{
}

CameraBroker::~CameraBroker()
{
    delete _impl;
}

zeroeq::URI CameraBroker::getURI() const
{
    return _impl->getURI();
}
}
}
}
