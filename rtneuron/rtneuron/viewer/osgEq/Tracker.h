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

#ifndef RTNEURON_OSGEQ_TRACKER_H
#define RTNEURON_OSGEQ_TRACKER_H

#include <eq/eq.h>
#include <vrpn_Tracker.h>

#include <boost/scoped_ptr.hpp>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
//! Encapsulation of a VPRN tracker client.
class Tracker : public boost::noncopyable
{
public:
    /*--- Public constructors/destructor ---*/

    /**
       Creates a connection to a remote VRPN tracker server using the
       device name given.
     */
    Tracker(const std::string& deviceName);

    ~Tracker();

    /*--- Public member fuctions ---*/

    /**
       Returns the current matrix position and orientation
    */
    eq::Matrix4f sample();

private:
    /*--- Private member variables ---*/

    /** The transformation matrix */
    eq::Matrix4f _matrix;

    /** world to emitter transformation */
    eq::Matrix4f _worldToEmitter;
    /** sensor to object transformation */
    eq::Matrix4f _sensorToObject;

    /** VRPN tracker device */
    boost::scoped_ptr<vrpn_Tracker_Remote> _tracker; //!< VRPN tracking device
    int32_t _sensorID;
    int32_t _lowestSeenSensorID;

    /*--- Private member functions ---*/

    /** Callback function to call by VRPN when the tracker's position is
        updated. */
    static void VRPN_CALLBACK _trackerCallback(void* userData,
                                               const vrpn_TRACKERCB data);

    bool _canUseSensor(const vrpn_TRACKERCB& data);
    void _updateMatrixFromVRPN(const vrpn_TRACKERCB& data);
};
}
}
}
#endif
