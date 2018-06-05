/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Glendon Holst <glendon.holst@kaust.edu.sa>
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

#include "GyrationMouse.h"

#include "util/log.h"
#include "util/vec_to_vec.h"

#include <osg/Vec4>

#include <iostream>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
GyrationMouse::SampleData::SampleData()
    : isTrackingAttitude(false)
    , attitude()
    , translation()
    , scroll(0.0)
    , reset(false)
{
}

GyrationMouse::GyrationMouse(const AttributeMap& configuration)
    : VRPNMultiDeviceBase(configuration)
    , _isTrackingInitialized(false)
    , _isTrackingPosition(false)
    , _isTrackingAttitude(false)
{
    if (_button->register_change_handler(this, _buttonCallback) == -1)
        throw(
            std::runtime_error("RTNeuron::GyrationMouse couldn't "
                               "register button handler to Button device"));

    if (_analog->register_change_handler(this, _analogCallback) == -1)
        throw(
            std::runtime_error("RTNeuron::GyrationMouse couldn't "
                               " register analog handler to Analog device"));

    if (_tracker->register_change_handler(this, _trackerCallback) == -1)
        throw(
            std::runtime_error("RTNeuron::GyrationMouse couldn't "
                               "register tracker handler to Tracker device"));
}

GyrationMouse::~GyrationMouse()
{
}

void GyrationMouse::reset()
{
    _transf.reset = true;
}

void GyrationMouse::sample(osg::Vec3d& translationDelta, osg::Vec3d&,
                           osg::Quat& attitude, bool& reset_)
{
    /* Initialize one-shot values. */
    _transf.scroll = 0.0;
    _transf.reset = false;

    if (!_isTrackingPosition)
        _transf.translation.set(0, 0, 0);

    /* Process VRPN events; there is no guarantee of new events. */
    _button->mainloop();
    _analog->mainloop();
    _tracker->mainloop();

    _transf.isTrackingAttitude = _isTrackingAttitude;

    /* Computing the return values. */
    reset_ = _transf.reset;

    translationDelta = _transf.translation * 0.05;
    translationDelta.z() += 0.1 * _transf.scroll;

    if (_transf.isTrackingAttitude)
        attitude = _transf.attitude;
}

void VRPN_CALLBACK GyrationMouse::_buttonCallback(void* userData,
                                                  const vrpn_BUTTONCB b)
{
    LBLOG(LOG_TRACKER_DEVICES) << "Button:" << b.button << " state:" << b.state
                               << std::endl;

    GyrationMouse* tracker = static_cast<GyrationMouse*>(userData);

    if (tracker->_isTrackingInitialized)
    {
        if (b.button == 0) // left-button - positional
        {
            if (b.state && !tracker->_isTrackingPosition)
                tracker->_startPosition = tracker->_position;

            tracker->_isTrackingPosition = b.state;
        }
        else if (b.button == 1) // right-button - attitude
        {
            tracker->_isTrackingAttitude = b.state;
        }
    }

    if (b.button == 2 && b.state) // middle-button - reset
        tracker->reset();
}

void VRPN_CALLBACK GyrationMouse::_analogCallback(void* userData,
                                                  const vrpn_ANALOGCB a)
{
    if (lunchbox::Log::topics & (LOG_TRACKER_DEVICES))
    {
        for (vrpn_int32 i = 0; i < a.num_channel; i++)
            LBLOG(LOG_TRACKER_DEVICES) << "Channel[" << i
                                       << "]:" << a.channel[i] << std::endl;
    }

    GyrationMouse* tracker = static_cast<GyrationMouse*>(userData);

    /* Scroll wheel is analog device, channel 1 (value is either -1 or 1 or
       0 for */
    tracker->_transf.scroll += a.channel[0];
}

void VRPN_CALLBACK GyrationMouse::_trackerCallback(void* userData,
                                                   const vrpn_TRACKERCB data)
{
    GyrationMouse* tracker = static_cast<GyrationMouse*>(userData);

    if (!tracker->canUseSensor(data))
        return;

    /* Update the matrix value */
    tracker->_updateTransformFromVRPN(data);
}

void GyrationMouse::_updateTransformFromVRPN(const vrpn_TRACKERCB& data)
{
    using namespace eq;

    LBLOG(LOG_TRACKER_DEVICES) << "Position " << data.pos[0] << ' '
                               << data.pos[1] << ' ' << data.pos[2] << std::endl
                               << "Quaternion " << data.quat[0] << ' '
                               << data.quat[1] << ' ' << data.quat[2] << ' '
                               << data.quat[3] << std::endl;

    _prevPosition = _position;

    const brion::Vector3f position =
        _positionTransform * brion::Vector3d(data.pos);
    const brion::Vector3f attitude =
        _attitudeTransform * brion::Vector3d(data.quat);

    _position = vec_to_vec(position);
    _attitude = osg::Quat(osg::Vec4(vec_to_vec(attitude), data.quat[3]));

    if (!_isTrackingInitialized)
    {
        _prevPosition = _position;
        _startPosition = _position;
        _isTrackingInitialized = true;
    }

    if (_isTrackingPosition)
    {
        _transf.translation = _position - _startPosition;
    }

    if (_isTrackingAttitude)
    {
        _transf.attitude = _attitude;
    }
}
}
}
}
