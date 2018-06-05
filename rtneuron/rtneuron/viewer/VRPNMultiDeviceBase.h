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

#ifndef RTNEURON_VRPNMULTIDEVICEBASE_H
#define RTNEURON_VRPNMULTIDEVICEBASE_H

#include "AttributeMap.h"

#include <vrpn_Analog.h>
#include <vrpn_Button.h>
#include <vrpn_Tracker.h>

#include <lunchbox/debug.h>

#include <eq/fabric/vmmlib.h>

#include <boost/scoped_ptr.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class VRPNMultiDeviceBase
{
protected:
    VRPNMultiDeviceBase(const AttributeMap& configuration);
    ~VRPNMultiDeviceBase();

    bool canUseSensor(const vrpn_TRACKERCB& data);

    boost::scoped_ptr<vrpn_Button_Remote> _button;   //!< VRPN button device
    boost::scoped_ptr<vrpn_Analog_Remote> _analog;   //!< VRPN axis device
    boost::scoped_ptr<vrpn_Tracker_Remote> _tracker; //!< VRPN tracking device

    eq::fabric::Matrix3f _positionTransform;
    eq::fabric::Matrix3f _attitudeTransform;

private:
    int32_t _trackerSensorID;
    int32_t _lowestSeenSensorID;

    void _init(const AttributeMap& configuration);
};
}
}
}
#endif
