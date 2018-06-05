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

#ifndef RTNEURON_WIIMOTE_H
#define RTNEURON_WIIMOTE_H

#include "VRPNManipulatorDevice.h"

#include <vrpn_Analog.h>
#include <vrpn_Button.h>
#include <vrpn_WiiMote.h>

#include <osg/Matrix>
#include <osg/Vec2>
#include <osg/Vec3>

#include <boost/signals2/signal.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class Wiimote : public VRPNManipulatorDevice
{
public:
    /*--- Public declarations ---*/

    class SampleData
    {
    public:
        osg::Vec2f joyVec; //!< Nunchuk joystick vector
        osg::Vec3f ir;
        osg::Vec3f ir1;
        osg::Vec3f ir2;
        osg::Vec3f wmOrient;
        osg::Vec3f wmAccel;
        osg::Vec3f gyro;
        bool reset;
    };

    enum Mode
    {
        TRANSLATION,
        ROTATION,
        ZOOM
    };

    enum PointingMode
    {
        ORIENT,
        SHIFT
    };

    /** \todo This list is not complete */
    enum Button
    {
        HOME_BUTTON,
        A_BUTTON,
        B_BUTTON,
        DOWN_BUTTON,
        UP_BUTTON,
        Z_BUTTON,
        C_BUTTON
    };

    typedef void ButtonPressedSignalSignature(Button);
    typedef boost::signals2::signal<ButtonPressedSignalSignature>
        ButtonPressedSignal;

    /*--- Public constructors/destructor ---*/

    explicit Wiimote(const std::string& deviceName);

    ~Wiimote();

    /*--- Public member functions ---*/

    Mode getMode() const; //!< @return the interaction mode
    void setMode(const Mode mode);

    PointingMode getPointingMode() const;
    void setPointingMode(const PointingMode mode);

    void reset(); //!< @internal Trigger a reset

    void sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                osg::Quat& attitude, bool& reset);

    SampleData sample();

    /* Interaction */
    ButtonPressedSignal buttonPressed;

private:
    /*--- Private member variables ---*/

    SampleData _transf; //!< OSG data

    vrpn_Button_Remote _button; //!< VRPN button device
    vrpn_Analog_Remote _analog; //!< VRPN axis device

    Mode _mode;             //!< Interaction mode
    PointingMode _pointing; //!< Pointer mode

    /*--- Private member functions ---*/

    /** Callback function to call by VRPN when the Wiimote's button parameters
        are updated. */
    static void VRPN_CALLBACK _buttonCallback(void* userdata,
                                              const vrpn_BUTTONCB b);

    /** Callback function to call by VRPN when the Wiimote's analog parameters
        are updated. */
    static void VRPN_CALLBACK _analogCallback(void* userData,
                                              const vrpn_ANALOGCB a);

    void _updateCameraFromVRPN(const vrpn_ANALOGCB& a);
};
}
}
}
#endif
