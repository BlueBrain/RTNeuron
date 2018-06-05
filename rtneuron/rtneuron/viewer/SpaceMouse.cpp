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

#include "SpaceMouse.h"

#include "util/log.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const char* const DEFAULT_URL = "device@localhost";
}

/*
  Constructors/destructor
 */
SpaceMouse::SpaceMouse(const std::string& deviceName)
    : _button(deviceName.empty() ? DEFAULT_URL : deviceName.c_str())
    , _analog(deviceName.empty() ? DEFAULT_URL : deviceName.c_str())
{
    if (deviceName.empty())
    {
        LBINFO << "Using " << DEFAULT_URL << " for SpaceMouse VRPN device."
               << std::endl;
    }

    if (!_analog.connectionPtr() || !_button.connectionPtr())
        throw(std::runtime_error("Error opening " + deviceName));

    if (_button.register_change_handler(this, _buttonCallback) == -1)
        throw(
            std::runtime_error("osgEq::SpaceMouse couldn't register button "
                               "handler to SpaceMouse device" +
                               deviceName));

    if (_analog.register_change_handler(this, _analogCallback) == -1)
        throw(
            std::runtime_error("osgEq::SpaceMouse couldn't register analog "
                               "handler to SpaceMouse device" +
                               deviceName));
}

SpaceMouse::~SpaceMouse()
{
}

/*
  Member functions
*/
void SpaceMouse::reset()
{
    _transf.reset = true;
}

void SpaceMouse::sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                        osg::Quat&, bool& reset_)

{
    _button.mainloop();
    _analog.mainloop();

    reset_ = _transf.reset;
    _transf.reset = false;

    translationDelta = osg::componentMultiply(_transf.translation,
                                              osg::Vec3f(0.05, 0.05, 0.01));
    rotationDelta.x() = 0.02 * _transf.rotation.x();
    rotationDelta.y() = 0.02 * _transf.rotation.y();
    /* Rotating the model around Z axis could make the scene confusing
       so this rotation axis ignored. */
}

void VRPN_CALLBACK SpaceMouse::_buttonCallback(void* userData,
                                               const vrpn_BUTTONCB b)
{
    SpaceMouse* spMouse = static_cast<SpaceMouse*>(userData);

    if (b.state)
        spMouse->reset();
}

void VRPN_CALLBACK SpaceMouse::_analogCallback(void* userData,
                                               const vrpn_ANALOGCB a)
{
    SpaceMouse* spMouse = static_cast<SpaceMouse*>(userData);

    /* Update the view data */
    spMouse->_updateAnalogFromVRPN(a);
}

void SpaceMouse::_updateAnalogFromVRPN(const vrpn_ANALOGCB& a)
{
    _transf.translation.x() = a.channel[0];
    _transf.translation.y() = a.channel[2];
    _transf.translation.z() = a.channel[1];
    _transf.rotation.x() = a.channel[3];
    _transf.rotation.y() = a.channel[5];
    _transf.rotation.z() = a.channel[4];
}
}
}
}
