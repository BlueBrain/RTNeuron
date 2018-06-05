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

#ifndef RTNEURON_INTERSENSE_H
#define RTNEURON_INTERSENSE_H

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
class Intersense : public VRPNManipulatorDevice, protected VRPNMultiDeviceBase
{
public:
    /*--- Public contructor/destructor ---*/
    Intersense(const AttributeMap& configuration);

    ~Intersense();

    /*--- Public member functions ---*/

    /** @internal Trigger a reset. */
    void reset();

    void sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                osg::Quat& attitude, bool& reset);

private:
    /*--- Private declarations ---*/
    class SampleData
    {
    public:
        bool isTrackingAttitude; /*!< If true, apply attitude manipulator
                                   data; otherwise, ignore data */
        osg::Quat attitude;      //!< Attitude of the manipulator

        osg::Vec3f rotation;
        osg::Vec3f translation;
        bool reset;
    };

    /** Mapping modes between XY analog stick and 3D transforms. */
    enum AnalogTransformMode
    {
        TRANSFORM_MODE_TRANSLATE,
        XY_TranslateX_TranslateY = TRANSFORM_MODE_TRANSLATE,
        XY_None_TranslateZ,

        TRANSFORM_MODE_ROTATION,
        XY_RotateX_RotateY = TRANSFORM_MODE_ROTATION,
        XY_None_RotateZ,
    };

    /*--- Private member variables ---*/
    SampleData _transf; //!< OSG data

    bool _isTrackingInitialized;
    bool _isTrackingPosition;
    bool _isTrackingAttitude;

    AnalogTransformMode _analogMode;

    osg::Quat _attitude;            //!< Attitude of the tracker
    osg::Vec3f _position;           //!< Position of the tracker
    osg::Vec3f _prevPosition;       //!< Last measured position of the tracker
    osg::Vec3f _startPosition;      /*!< Position of the tracker when
                                      _isTrackingPosition triggered */
    osg::Vec3f _analogTranslation;  //!< Translation delta component from analog
    osg::Vec3f _trackerTranslation; /*!< Translation delta component from
                                      tracker */

    /*--- Private member functions ---*/

    /** Callback function to call by VRPN when the button parameters are
        updated. */
    static void VRPN_CALLBACK _buttonCallback(void* userdata,
                                              const vrpn_BUTTONCB b);

    /** Callback function to call by VRPN when the analog parameters
        are updated. */
    static void VRPN_CALLBACK _analogCallback(void* userData,
                                              const vrpn_ANALOGCB a);

    /** Callback function to call by VRPN when the tracker's position
        is updated. */
    static void VRPN_CALLBACK _trackerCallback(void* userData,
                                               const vrpn_TRACKERCB data);

    void _updateTrackerFromVRPN(const vrpn_TRACKERCB& data);
    void _updateAnalogFromVRPN(const vrpn_ANALOGCB& a);
};
}
}
}
#endif
