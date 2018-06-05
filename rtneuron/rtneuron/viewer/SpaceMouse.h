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

#ifndef RTNEURON_SPACEMOUSE_H
#define RTNEURON_SPACEMOUSE_H

#include "VRPNManipulatorDevice.h"

#include <vrpn_3DConnexion.h>

#include <osg/Vec3>

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
class SpaceMouse : public VRPNManipulatorDevice
{
public:
    /*--- Public constructors/destructor ---*/
    explicit SpaceMouse(const std::string& deviceName);
    ~SpaceMouse();

    /*--- Public member functions ---*/
    void reset(); //!< @internal Trigger a reset

    void sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                osg::Quat& attitude, bool& reset);

private:
    /*--- Private declarations ---*/
    class SampleData
    {
    public:
        osg::Vec3f rotation;
        osg::Vec3f translation;
        bool reset;
    };

    /*--- Private member variables ---*/

    SampleData _transf; //!< OSG data

    vrpn_Button_Remote _button; //!< VRPN button device
    vrpn_Analog_Remote _analog; //!< VRPN axis device

    /*--- Private member functions ---*/

    /** \brief Callback function to call by VRPN when the SpaceMouse's
        button parameters are updated. */
    static void VRPN_CALLBACK _buttonCallback(void* userdata,
                                              const vrpn_BUTTONCB b);

    /** Callback function to call by VRPN when the SpaceMouse's analog
        parameters are updated. */
    static void VRPN_CALLBACK _analogCallback(void* userData,
                                              const vrpn_ANALOGCB a);

    void _updateAnalogFromVRPN(const vrpn_ANALOGCB& a);
};
}
}
}
#endif
