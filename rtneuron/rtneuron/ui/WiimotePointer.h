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

#ifndef RTNEURON_API_UI_WIIMOTEPOINTER_H
#define RTNEURON_API_UI_WIIMOTEPOINTER_H

#include "Pointer.h"

#include <deque>

namespace bbp
{
namespace rtneuron
{
/**
 * A pointer device that uses the Wiimote to select scene objects, to change the
 * stereo correction and to reset the camera.
 */
class WiimotePointer : public Pointer
{
public:
    /*--- Public constructors/destructors ---*/

    WiimotePointer(const std::string& hostName);

    ~WiimotePointer();

    /*--- Public member functions ---*/

    //! Update state from Wiimote sensor data
    virtual void update();

private:
    /*--- Private member attributes ---*/
    class Impl;
    std::unique_ptr<Impl> _impl;

    /*--- Private member functions ---*/

    //! Kalman filtering
    void _kalmanPredict(const double gyroRate, const double dt);
    void _kalmanCorrect(const double actualAngle) const;

    //! Runge-Kutta 4 integration
    void _computeRK4(const osg::Vec3 val0);

    //! Low-pass filter
    osg::Vec3 _smooth(const std::deque<osg::Vec3>& data);

    //! Center Wiimote IR data
    osg::Vec3 _center(const osg::Vec3 pos) const;

    //! Emits the pick signal with the current pointing direction.
    void _select();
};
}
}

#endif /* RTNEURON_API_GUI_WIIMOTEPOINTER_H */
