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

#include "Intersense.h"

#include "util/log.h"
#include "util/vec_to_vec.h"

#include <osg/Vec4>

#include <lunchbox/debug.h>

#include <iostream>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Constructors
 */
Intersense::Intersense(const AttributeMap& configuration)
    : VRPNMultiDeviceBase(configuration)
    , _isTrackingInitialized(false)
    , _isTrackingPosition(false)
    , _isTrackingAttitude(false)
    , _analogMode(TRANSFORM_MODE_TRANSLATE)
{
    if (_button->register_change_handler(this, _buttonCallback) == -1)
        throw std::runtime_error(
            "RTNeuron::Intersense couldn't "
            "register button handler to Button device");

    if (_analog->register_change_handler(this, _analogCallback) == -1)
        throw std::runtime_error(
            "RTNeuron::Intersense couldn't register"
            " analog handler to Analog device");

    if (_tracker->register_change_handler(this, _trackerCallback) == -1)
        throw std::runtime_error(
            "RTNeuron::Intersense couldn't register"
            " tracker handler to Tracker device");
}

Intersense::~Intersense()
{
}

void Intersense::reset()
{
    _transf.reset = true;
}

void Intersense::sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                        osg::Quat& attitude, bool& reset_)

{
    /* Initialize one-shot values. */
    _transf.reset = false;

    if (!_isTrackingPosition)
    {
        _trackerTranslation.set(0, 0, 0);
    }

    /* Process VRPN events; there is no guarantee of new events. */
    _button->mainloop();
    _analog->mainloop();
    _tracker->mainloop();

    _transf.translation = _analogTranslation + _trackerTranslation;
    _transf.isTrackingAttitude = _isTrackingAttitude;

    /* Compute return values. */
    reset_ = _transf.reset;
    translationDelta = osg::componentMultiply(_transf.translation,
                                              osg::Vec3f(0.05, 0.05, 0.01));
    rotationDelta = _transf.rotation * 0.02;
    if (_transf.isTrackingAttitude)
        attitude = _transf.attitude;
}

void VRPN_CALLBACK Intersense::_buttonCallback(void* userData,
                                               const vrpn_BUTTONCB b)
{
    LBLOG(LOG_TRACKER_DEVICES) << "Button:" << b.button << " state:" << b.state
                               << std::endl;

    Intersense* isenseWand = static_cast<Intersense*>(userData);

    if (isenseWand->_isTrackingInitialized)
    {
        if (b.button == 1) // 1st-button - positional
        {
            if (b.state && !isenseWand->_isTrackingPosition)
                isenseWand->_startPosition = isenseWand->_position;

            isenseWand->_isTrackingPosition = b.state;
        }
        else if (b.button == 0) // 2nd-button - attitude
        {
            isenseWand->_isTrackingAttitude = b.state;
        }
    }

    /* Button 4 = stick-button, switch between XY-axes and Z-axis
       Button 5 = trigger-button, switch between rotation and translation */
    if (b.button == 4 || b.button == 5)
    {
        bool stateChanged = false;
        switch (isenseWand->_analogMode)
        {
        case XY_TranslateX_TranslateY:
            if (b.state)
            {
                isenseWand->_analogMode =
                    b.button == 4 ? XY_None_TranslateZ : XY_RotateX_RotateY;
                stateChanged = true;
            }
            break;
        case XY_RotateX_RotateY:
            if (b.state)
            {
                isenseWand->_analogMode =
                    b.button == 4 ? XY_None_RotateZ : XY_TranslateX_TranslateY;
                stateChanged = true;
            }
            break;
        case XY_None_TranslateZ:
            if (!b.state)
            {
                isenseWand->_analogMode =
                    b.button == 4 ? XY_TranslateX_TranslateY : XY_None_RotateZ;
                stateChanged = true;
            }
            break;
        case XY_None_RotateZ:
            if (!b.state)
            {
                isenseWand->_analogMode =
                    b.button == 4 ? XY_RotateX_RotateY : XY_None_TranslateZ;
                stateChanged = true;
            }
            break;
        default:
            std::cerr << "Unhandled AnalogTransformMode: " << __LINE__ << ':'
                      << __FILE__ << std::endl;
            abort();
            break;
        }

        if (stateChanged)
        {
            isenseWand->_transf.rotation.set(0, 0, 0);
            isenseWand->_analogTranslation.set(0, 0, 0);
        }
    }

    if (b.button == 2 && b.state) // 3rd-button - reset
        isenseWand->reset();
}

void VRPN_CALLBACK Intersense::_analogCallback(void* userData,
                                               const vrpn_ANALOGCB a)
{
    if (lunchbox::Log::topics & (LOG_TRACKER_DEVICES))
    {
        for (vrpn_int32 i = 0; i < a.num_channel; i++)
            LBLOG(LOG_TRACKER_DEVICES) << "Channel[" << i
                                       << "]:" << a.channel[i] << std::endl;
    }

    /* Update the view data */
    static_cast<Intersense*>(userData)->_updateAnalogFromVRPN(a);
}

void VRPN_CALLBACK Intersense::_trackerCallback(void* userData,
                                                const vrpn_TRACKERCB data)
{
    Intersense* isenseTracker = static_cast<Intersense*>(userData);

    if (!isenseTracker->canUseSensor(data))
        return;

    /* Update the matrix value */
    isenseTracker->_updateTrackerFromVRPN(data);
}

void Intersense::_updateAnalogFromVRPN(const vrpn_ANALOGCB& a)
{
    if (a.num_channel < 2)
    {
        return;
    }

    LBLOG(LOG_TRACKER_DEVICES) << "Channel x:" << a.channel[0]
                               << " y:" << a.channel[1] << std::endl;

    _analogTranslation.set(0, 0, 0);
    _transf.rotation.set(0, 0, 0);

    switch (_analogMode)
    {
    case XY_TranslateX_TranslateY:
        _analogTranslation.x() = a.channel[0];
        _analogTranslation.y() = -a.channel[1];
        break;
    case XY_None_TranslateZ:
        _analogTranslation.z() = -a.channel[1];
        break;
    case XY_RotateX_RotateY:
        _transf.rotation.x() = -a.channel[1];
        _transf.rotation.y() = -a.channel[0];
        break;
    case XY_None_RotateZ:
        _transf.rotation.z() = -a.channel[1];
        break;
    default:
        LBUNREACHABLE;
        break;
    }
}

void Intersense::_updateTrackerFromVRPN(const vrpn_TRACKERCB& data)
{
    using namespace eq::fabric;

    LBLOG(LOG_TRACKER_DEVICES) << "Position " << data.pos[0] << ' '
                               << data.pos[1] << ' ' << data.pos[2] << std::endl
                               << "Quaternion " << data.quat[0] << ' '
                               << data.quat[1] << ' ' << data.quat[2] << ' '
                               << data.quat[3] << std::endl;

    _prevPosition = _position;

    const Vector3f position = _positionTransform * Vector3d(data.pos);
    const Vector3f attitude = _attitudeTransform * Vector3d(data.quat);

    _position = vec_to_vec(position);
    _attitude = osg::Quat(osg::Vec4(vec_to_vec(attitude), data.quat[3]));

    if (!_isTrackingInitialized)
    {
        _prevPosition = _position;
        _startPosition = _position;
        _trackerTranslation.set(0, 0, 0);
        _isTrackingInitialized = true;
    }

    if (_isTrackingPosition)
    {
        _trackerTranslation = _position - _startPosition;
    }

    if (_isTrackingAttitude)
    {
        _transf.attitude = _attitude;
    }
}
}
}
}
