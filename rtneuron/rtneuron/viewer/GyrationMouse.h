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

#ifndef RTNEURON_GYRATIONMOUSE_H
#define RTNEURON_GYRATIONMOUSE_H

#include "VRPNManipulatorDevice.h"
#include "VRPNMultiDeviceBase.h"

#include <osg/Quat>
#include <osg/Vec3>

#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class GyrationMouse : public VRPNManipulatorDevice,
                      protected VRPNMultiDeviceBase
{
public:
    /*--- Public constructors/destructor */

    GyrationMouse(const AttributeMap& configuration);

    ~GyrationMouse();

    /*--- Public member functions */

    void reset(); //!< @internal Trigger a reset

    void sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                osg::Quat& attitude, bool& reset);

private:
    /*--- Private declarations ---*/

    struct SampleData
    {
        SampleData();

        bool isTrackingAttitude; /*!< If true, apply attitude manipulator data;
                                   otherwise, ignore data */
        osg::Quat attitude;      //!< Attitude of the manipulator
        osg::Vec3f translation;  //!< Positional delta of the manipulator
        vrpn_float64 scroll;     //!< Scroll-wheel delta of the manipulator
        bool reset;
    };

    /*--- Private member variables ---*/

    SampleData _transf; //!< OSG data

    bool _isTrackingInitialized;
    bool _isTrackingPosition;
    bool _isTrackingAttitude;

    osg::Quat _attitude;  //!< Attitude of the manipulator
    osg::Vec3f _position; //!< Position of the manipulator
    osg::Vec3f _prevPosition;
    osg::Vec3f _startPosition;

    /*--- Private member functions ---*/

    void _updateTransformFromVRPN(const vrpn_TRACKERCB& data);

    /** Callback function to call by VRPN when the tracker's position
        is updated. */
    static void VRPN_CALLBACK _trackerCallback(void* userData,
                                               const vrpn_TRACKERCB data);

    /** Callback function to call by VRPN when the button parameters
        are updated. */
    static void VRPN_CALLBACK _buttonCallback(void* userData,
                                              const vrpn_BUTTONCB b);

    /** Callback function to call by VRPN when the analog parameters are
        updated. */
    static void VRPN_CALLBACK _analogCallback(void* userData,
                                              const vrpn_ANALOGCB a);
};
}
}
}
#endif
