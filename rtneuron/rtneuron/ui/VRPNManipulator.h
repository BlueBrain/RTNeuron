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

#ifndef RTNEURON_API_UI_VRPNMANIPULATOR_H
#define RTNEURON_API_UI_VRPNMANIPULATOR_H

#include "../AttributeMap.h"
#include "../CameraManipulator.h"

#include <string>

namespace bbp
{
namespace rtneuron
{
/**
 * A camera manipulator which is controlled via a VRPN device. The current
 * supported devices are space mouse and Wiimote.
 */
class VRPNManipulator : public CameraManipulator
{
public:
    /*--- Public declarations ---*/

    enum DeviceType
    {
        GYRATION_MOUSE,
        INTERSENSE_WAND,
        SPACE_MOUSE,
#ifdef RTNEURON_USE_WIIUSE
        WIIMOTE
#endif
    };

    /*--- Public constructor/destructor ---*/

    /**
       Create a VRPN manipulator using the given configuration

       @param type The device type.
       @param configuration An attribute map with the device-specific
              configuration. For GYRATION_MOUSE and INTERSENSE_WAND the
              configuration attributes are:
              - *tracker* (AttributeMap):
                - *url* (string):
                  The VPRN device URL.
                - *sensorid* (int | "any" | "lowest"):
                  The sensor ID for the tracker. Defaults to "any" if not
                  provided.
                - *attitude_axis* (string or floatx9):
                  A base change matrix from the device coordinates to world
                  coordinates for the attitude axis. Defaults to the identity
                  matrix is not given.
                - *position_axis* (string or floatx9):
                  A base change matrix from the device coordinates to world
                  coordinates for the tracker position. Defaults to the identity
                  matrix is not given
                .
              - *button* (string|AttributeMap):
                If it is a string, it contains the VRPN device URL
                If it is an attribute map, the map contains:
                - *url* (string):
                  The device specification for the device.
                .
              - *analog* (string|AttributeMap):
                If it is a string, it contains the VRPN device URL
                If it is an attribute map, the map contains:
                - *url* (string):
                  The device specification for the device.
                .
              - *url*
                The default URL for VRPN devices that are not specified
                explicitly or which do not have a *url* attribute.
              .
              Any of *tracker*, *analog* and *button* can be omitted if *url*
              is given. In that case, additional attributes take default
              values.

              For SPACE_MOUSE and WIIMOTE
              - *analog* (string|AttributeMap):
                If it is a string, it contains the VRPN device URL
                If it is an attribute map, the map contains:
                - *url* (string):
                  The device specification for the device.
                .
              - *url* (string):
                The URL of the VRPN device. Provided for consistency with
                GYRATION_MOUSE and INTERSENSE_WAND and with forward
                compatibility in mind.
              .
              If neither *analog* nor *url* are given, the URL will default to
              "WiiMote@localhost" for WIIMOTE and "device@localhost" for
              SPACE_MOUSE. The same device will be used for the device button.

     */
    VRPNManipulator(const DeviceType type, const AttributeMap& configuration);

    /** @deprecated */
    VRPNManipulator(const DeviceType type, const std::string& url = "");

    virtual ~VRPNManipulator();

    /*--- Public member functions C++ only ---*/

    /** @name C++ only public interface */
    ///@{

    /** @internal */
    virtual osgGA::CameraManipulator* osgManipulator();

    /** @internal */
    static std::string getDeviceURL(const AttributeMap& attributes,
                                    const std::string& name);

    ///@}

private:
    /*--- Private member attributes ---*/

    class Impl;
    Impl* _impl;
};
}
}
#endif
