/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Juan Hernando <juan.hernando@epfl.ch>
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

#include "Tracker.h"

#include "util/log.h"

#include <vmmlib/quaternion.hpp>

#include <boost/regex.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <locale>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
namespace
{
const int32_t MAX_SENSOR_ID = std::numeric_limits<int32_t>::max();
const int32_t USE_LOWEST_VALID_SENSOR_ID = std::numeric_limits<int32_t>::max();
const int32_t USE_ANY_VALID_SENSORID = std::numeric_limits<int32_t>::max() - 1;
}

/*
  Constructors/destructor
*/

Tracker::Tracker(const std::string& deviceName)
    : _sensorID(USE_LOWEST_VALID_SENSOR_ID)
    , _lowestSeenSensorID(MAX_SENSOR_ID)
{
    // <device_url> = ((<device_name>@<device_desc>)(&key-value)*
    // <device_name> = (non-space, non-@?/:;-symbols)+
    // <device_desc> = <ssep_word>@<ssep_word>[:<port>]
    static const boost::regex s_deviceURLRE(
        "^([^\\s@:;\\?&/]+@[^\\s@;\\?&/]+)(&\\w+=\\w+)?$");
    static const boost::regex s_sensoridRE("^&sensorid=(\\d+|any)$");

    std::string trackerDevName;

    boost::smatch deviceURL;
    if (boost::regex_search(deviceName, deviceURL, s_deviceURLRE))
    {
        // option index from s_deviceURLRE
        static const size_t s_optionIndexStart = 2;

        if (lunchbox::Log::topics & core::LOG_TRACKER_DEVICES)
        {
            for (size_t i = 0; i < deviceURL.size(); ++i)
            {
                LBLOG(core::LOG_TRACKER_DEVICES) << "deviceURL:" << i << " = '"
                                                 << deviceURL.str(i) << "'"
                                                 << std::endl;
            }
        }

        trackerDevName = deviceURL.str(1);
        for (size_t optionIdx = s_optionIndexStart;
             optionIdx < deviceURL.size(); ++optionIdx)
        {
            const std::string optionStr(deviceURL.str(optionIdx));

            if (optionStr.empty())
                continue;

            boost::smatch optionResult;

            if (boost::regex_search(optionStr, optionResult, s_sensoridRE))
            {
                std::string optionValue(optionResult.str(1));
                if (optionValue == "any")
                    _sensorID = USE_ANY_VALID_SENSORID;
                else
                    _sensorID = atoi(optionValue.c_str());
            }
            else
            {
                throw std::runtime_error(
                    "VRPN tracker couldn't process option: " + optionStr);
            }
        }
    }
    else
    {
        trackerDevName = deviceName;
    }

    LBLOG(core::LOG_TRACKER_DEVICES) << "tracker:" << trackerDevName
                                     << std::endl;

    _tracker.reset(new vrpn_Tracker_Remote(trackerDevName.c_str()));
    if (!_tracker->connectionPtr())
    {
        throw std::runtime_error(
            "osgEq::Tracker couldn't connect to tracker device" +
            trackerDevName);
    }

    /* Tell the tacker which callback handler function to use, and pass
       this as the user data (only sensor 1 is considered). */
    if (_tracker->register_change_handler(this, _trackerCallback) == -1)
    {
        throw std::runtime_error(
            "osgEq::Tracker couldn't connect to tracker device" +
            trackerDevName);
    }
}

Tracker::~Tracker()
{
}

/*
  Member functions
*/
eq::Matrix4f Tracker::sample()
{
    _tracker->mainloop();
    LBLOG(core::LOG_TRACKER_DEVICES) << "Tracker matrix\n"
                                     << _matrix << std::endl;

    return _matrix;
}

void VRPN_CALLBACK Tracker::_trackerCallback(void* userData,
                                             const vrpn_TRACKERCB data)
{
    Tracker* tracker = static_cast<Tracker*>(userData);

    if (!tracker->_canUseSensor(data))
        return;

    /* Update the matrix value */
    tracker->_updateMatrixFromVRPN(data);
}

bool Tracker::_canUseSensor(const vrpn_TRACKERCB& data)
{
    LBLOG(core::LOG_TRACKER_DEVICES)
        << "Sensor " << data.sensor << ", position (" << data.pos[0] << ", "
        << data.pos[1] << "," << data.pos[2] << "), orientation ("
        << data.quat[0] << "," << data.quat[1] << "," << data.quat[2] << ","
        << data.quat[3] << ")" << std::endl;

    if (_sensorID == USE_LOWEST_VALID_SENSOR_ID)
    {
        if (data.sensor > _lowestSeenSensorID)
            return false;

        _lowestSeenSensorID = data.sensor;
        return true;
    }
    return (_sensorID == USE_ANY_VALID_SENSORID || _sensorID == data.sensor);
}

void Tracker::_updateMatrixFromVRPN(const vrpn_TRACKERCB& data)
{
    typedef vmml::Quaternionf quaternion;
    /* \todo Check if the matrix is non-sensical */

    /* We exchange the x with z, y with x and z with y */
    quaternion quat(data.quat[0], data.quat[2], -data.quat[1], data.quat[3]);

    LBLOG(core::LOG_TRACKER_DEVICES)
        << "Position " << data.pos[0] << ' ' << data.pos[1] << ' '
        << data.pos[2] << std::endl
        << "Quaternion " << data.quat[0] << ' ' << data.quat[1] << ' '
        << data.quat[2] << ' ' << data.quat[3] << std::endl;

    _matrix = eq::Matrix4f(quat, eq::Vector3f(data.pos[0], data.pos[2],
                                              -data.pos[1]));
}
}
}
}
