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

#include <eq/types.h> /* Needs to be before VRPNMultiDeviceBase.h" */

#include "VRPNMultiDeviceBase.h"

#include "../ui/VRPNManipulator.h"

#include "util/attributeMapHelpers.h"

#include "util/log.h"

#include <boost/assign.hpp>
#include <boost/optional.hpp>
#include <boost/regex.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
/** Special VRPN Sensor ID values */
struct VRPNTrackerSensor
{
    enum SensorID
    {
        UseAnyValidSensorID = -1,
        UseLowestValidSensorID = -2
    };
};

bool _parseTransform(const std::string& transform, eq::Matrix3f& out)
{
    static const boost::regex s_axis_re("^([xyzXYZ])([xyzXYZ])([xyzXYZ])$");
    static std::map<char, eq::Vector3f> s_axisMap =
        boost::assign::map_list_of('X', eq::Vector3f::unitX())(
            'Y', eq::Vector3f::unitY())('Z', eq::Vector3f::unitZ());

    boost::smatch optResult;
    if (!boost::regex_search(transform, optResult, s_axis_re))
        return false;

    for (size_t r = 0; r < 3; ++r)
    {
        std::string optionValue(optResult.str(r + 1));
        LBASSERT(optionValue.size() == 1);
        const char axis = optionValue[0];
        const float scale = std::isupper(axis) ? 1.f : -1.f;
        const eq::Vector3f v = scale * s_axisMap[std::toupper(axis)];

        out.setRow(r, v);
    }
    return true;
}

void _tryParseTransform(const AttributeMap& attributes, const std::string& name,
                        eq::Matrix3f& matrix)
{
    try
    {
        using namespace AttributeMapHelpers;

        if (!getMatrix(attributes, name, matrix))
        {
            const std::string axis = attributes(name);
            if (!_parseTransform(axis, matrix))
            {
                LBWARN << "Failed to parse axis transform. Expected values: "
                          "string matching the regex [XYZxyz]{3} or a list"
                          " of 9 floats (row by row)"
                       << std::endl;
            }
        }
    }
    catch (std::runtime_error&)
    {
        /* This only happens if the attribute doesn't exist. In this case the
           matrix is unmodified */
    }
}

const int32_t MAX_SENSOR_ID = std::numeric_limits<int32_t>::max();
}

bool VRPNMultiDeviceBase::canUseSensor(const vrpn_TRACKERCB& data)
{
    LBLOG(LOG_TRACKER_DEVICES)
        << "Sensor " << data.sensor << ", position (" << data.pos[0] << ", "
        << data.pos[1] << "," << data.pos[2] << "), orientation ("
        << data.quat[0] << "," << data.quat[1] << "," << data.quat[2] << ","
        << data.quat[3] << ")" << std::endl;

    if (_trackerSensorID == VRPNTrackerSensor::UseLowestValidSensorID)
    {
        if (data.sensor > _lowestSeenSensorID)
            return false;

        _lowestSeenSensorID = data.sensor;
        return true;
    }
    return (_trackerSensorID == VRPNTrackerSensor::UseAnyValidSensorID ||
            _trackerSensorID == data.sensor);
}

VRPNMultiDeviceBase::VRPNMultiDeviceBase(const AttributeMap& configuration)
    : _lowestSeenSensorID(MAX_SENSOR_ID)
{
    _init(configuration);
}

VRPNMultiDeviceBase::~VRPNMultiDeviceBase()
{
}

void VRPNMultiDeviceBase::_init(const AttributeMap& configuration)
{
    using namespace boost;

    const std::string trackerDevName =
        VRPNManipulator::getDeviceURL(configuration, "tracker");
    if (trackerDevName.empty())
        LBTHROW(std::runtime_error("No URL provided for VRPN tracker device"));

    _trackerSensorID = VRPNTrackerSensor::UseAnyValidSensorID;
    _positionTransform = eq::Matrix3f();
    _attitudeTransform = eq::Matrix3f();

    const AttributeMapPtr trackerPtr =
        configuration("tracker", AttributeMapPtr());
    if (trackerPtr)
    {
        const AttributeMap& tracker = *trackerPtr;
        std::string trackerSensorIDStr;
        const int sensorIDResult = tracker.get("sensorid", trackerSensorIDStr);
        if (sensorIDResult == 1 && trackerSensorIDStr == "any")
            _trackerSensorID = VRPNTrackerSensor::UseAnyValidSensorID;
        else if (sensorIDResult == 1 && trackerSensorIDStr == "lowest")
            _trackerSensorID = VRPNTrackerSensor::UseLowestValidSensorID;
        else
        {
            if (tracker.get("sensorid", _trackerSensorID) != 1)
            {
                LBWARN << "Failed to parse sensor id from string. "
                       << "Expected: \"any\", \"lowest\", or positive integer"
                       << std::endl;
                _trackerSensorID = VRPNTrackerSensor::UseAnyValidSensorID;
            }
        }

        _tryParseTransform(tracker, "position_axis", _positionTransform);
        _tryParseTransform(tracker, "attitude_axis", _attitudeTransform);
    }

    const std::string buttonDevName =
        VRPNManipulator::getDeviceURL(configuration, "button");
    if (buttonDevName.empty())
        LBTHROW(std::runtime_error("No URL provided for VRPN button device"));

    const std::string analogDevName =
        VRPNManipulator::getDeviceURL(configuration, "analog");
    if (analogDevName.empty())
        LBTHROW(std::runtime_error("No URL provided for VRPN analog device"));

    LBLOG(LOG_TRACKER_DEVICES) << "tracker:" << trackerDevName << std::endl;
    LBLOG(LOG_TRACKER_DEVICES) << "tracker sensor id:" << _trackerSensorID
                               << std::endl;
    LBLOG(LOG_TRACKER_DEVICES) << "tracker positionTransform: " << std::endl
                               << _positionTransform << std::endl;
    LBLOG(LOG_TRACKER_DEVICES) << "tracker attitudeTransform: " << std::endl
                               << _attitudeTransform << std::endl;
    LBLOG(LOG_TRACKER_DEVICES) << "button:" << buttonDevName << std::endl;
    LBLOG(LOG_TRACKER_DEVICES) << "analog:" << analogDevName << std::endl;

    _tracker.reset(new vrpn_Tracker_Remote(trackerDevName.c_str()));
    _button.reset(new vrpn_Button_Remote(buttonDevName.c_str()));
    _analog.reset(new vrpn_Analog_Remote(analogDevName.c_str()));

    if (!_tracker->connectionPtr())
        throw std::runtime_error("Error opening tracker: " + trackerDevName);
    if (!_button->connectionPtr())
        throw std::runtime_error("Error opening button: " + buttonDevName);
    if (!_analog->connectionPtr())
        throw std::runtime_error("Error opening analog: " + analogDevName);
}
}
}
}
