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

#include "Wiimote.h"

#include "util/log.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const char* const DEFAULT_URL = "WiiMote@localhost";
}

Wiimote::Wiimote(const std::string& deviceName)
    : _button(deviceName.empty() ? DEFAULT_URL : deviceName.c_str())
    , _analog(deviceName.empty() ? DEFAULT_URL : deviceName.c_str())
    , _mode(TRANSLATION)
    , _pointing(ORIENT)
{
    if (deviceName.empty())
    {
        LBINFO << "Using " << DEFAULT_URL << " for WiiMote VRPN device."
               << std::endl;
    }

    if (!_analog.connectionPtr() || !_button.connectionPtr())
        throw(std::runtime_error(
            "Error opening device for Wiimote manipulator" + deviceName));

    if (_button.register_change_handler(this, _buttonCallback) == -1)
        throw(
            std::runtime_error("osgEq::Wiimote couldn't register button "
                               "handler to Wiimote device" +
                               deviceName));

    if (_analog.register_change_handler(this, _analogCallback) == -1)
        throw(
            std::runtime_error("osgEq::Wiimote couldn't register analog "
                               "handler to Wiimote device" +
                               deviceName));
}

Wiimote::~Wiimote()
{
}

Wiimote::Mode Wiimote::getMode() const
{
    return _mode;
}

void Wiimote::setMode(const Mode mode)
{
    _mode = mode;
}

void Wiimote::setPointingMode(const PointingMode mode)
{
    _pointing = mode;
}

Wiimote::PointingMode Wiimote::getPointingMode() const
{
    return _pointing;
}

void Wiimote::reset()
{
    _transf.reset = true;
}

Wiimote::SampleData Wiimote::sample()
{
    _button.mainloop();
    _analog.mainloop();

    const Wiimote::SampleData data = _transf;
    _transf.reset = false;

    return data;
}

void Wiimote::sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                     osg::Quat&, bool& reset_)
{
    _button.mainloop();
    _analog.mainloop();

    reset_ = _transf.reset;
    _transf.reset = false;

    if (reset_)
        return;

    switch (_mode)
    {
    case core::Wiimote::TRANSLATION:
        translationDelta.x() += 0.05 * _transf.joyVec.x();
        translationDelta.y() += 0.05 * -_transf.joyVec.y();
        break;

    case core::Wiimote::ROTATION:
        rotationDelta.x() += 0.02 * -_transf.joyVec.y();
        rotationDelta.y() += 0.02 * -_transf.joyVec.x();
        break;

    case core::Wiimote::ZOOM:
        translationDelta.z() = 0.02 * -_transf.joyVec.y();
        break;

    default:
        break;
    }
}

void VRPN_CALLBACK Wiimote::_buttonCallback(void* userData,
                                            const vrpn_BUTTONCB b)
{
    Wiimote* wiimote = static_cast<Wiimote*>(userData);

    switch (b.button)
    {
    /* Home button - reset pointer */
    case 0:
        if (b.state)
            wiimote->buttonPressed(Wiimote::HOME_BUTTON);
        break;

    /* A button - element selection */
    case 3:
        if (b.state)
            wiimote->buttonPressed(Wiimote::A_BUTTON);
        break;

    /* B button - wiimote back trigger */
    case 4:
        wiimote->setPointingMode(b.state ? SHIFT : ORIENT);
        if (b.state)
            wiimote->buttonPressed(Wiimote::B_BUTTON);
        break;
    /* Down button */
    case 9:
        if (b.state)
            wiimote->buttonPressed(Wiimote::DOWN_BUTTON);
        break;
    /* Up button */
    case 10:
        if (b.state)
            wiimote->buttonPressed(Wiimote::UP_BUTTON);
        break;

    /* Z button */
    case 16:
        wiimote->setMode(b.state ? ROTATION : TRANSLATION);
        if (b.state)
            wiimote->buttonPressed(Wiimote::Z_BUTTON);
        break;
    /* C button */
    case 17:
        wiimote->setMode(b.state ? ZOOM : TRANSLATION);
        if (b.state)
            wiimote->buttonPressed(Wiimote::C_BUTTON);
        break;
    default:
        /** \todo Right, left, other buttons ? */
        break;
    }
}

void VRPN_CALLBACK Wiimote::_analogCallback(void* userData,
                                            const vrpn_ANALOGCB a)
{
    Wiimote* wiimote = static_cast<Wiimote*>(userData);

    /* Update the view data */
    wiimote->_updateCameraFromVRPN(a);
}

void Wiimote::_updateCameraFromVRPN(const vrpn_ANALOGCB& a)
{
    /* Magnitude multiplied by angle */
    const float xAnag = a.channel[20] * -cos(a.channel[19] * M_PI / 180.0f);
    const float yAnag = a.channel[20] * -sin(a.channel[19] * M_PI / 180.0f);

    _transf.ir1 = osg::Vec3(a.channel[4], a.channel[5], 0.0f);
    _transf.ir2 = osg::Vec3(a.channel[7], a.channel[8], 0.0f);
    // Absolute IR-leds position coming from Wiiuse (precomputed)
    _transf.ir = osg::Vec3(a.channel[99], a.channel[100], 0.0f);

    _transf.wmAccel = osg::Vec3(a.channel[1], a.channel[2], a.channel[3]);
    _transf.gyro = osg::Vec3(a.channel[21], a.channel[22], -a.channel[23]);
    _transf.wmOrient = osg::Vec3(a.channel[96], a.channel[97], a.channel[98]);

    /* Nunchuck is gently shaken */
    if (a.channel[16] > .25 && a.channel[17] > .25 && a.channel[18] > .25)
    {
        reset();
    }

    _transf.joyVec.x() = fabs(xAnag) > 0.01 ? xAnag : 0;
    _transf.joyVec.y() = fabs(yAnag) > 0.01 ? yAnag : 0;
}
}
}
}
