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

#ifndef RTNEURON_VRPNMANIPULATORDEVICE_H
#define RTNEURON_VRPNMANIPULATORDEVICE_H

#include <osg/Quat>
#include <osg/Vec3d>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class VRPNManipulatorDevice
{
public:
    virtual ~VRPNManipulatorDevice() {}
    /**
       @param translationDelta Camera translation increment. Should be set
              to zero before calling
       @param rotationDelta Incremental rotation angles around the x, y, z
              axes. Should be set to zero before calling
       @param attitude Written to with the orientation matrix if the
              device tracks absoute orientation, otherwise left unmodified.
       @param reset Whether the camera manipulator should be reset to the
              home position or not before applying the other transformations.
     */
    virtual void sample(osg::Vec3d& translationDelta, osg::Vec3d& rotationDelta,
                        osg::Quat& attitude, bool& reset) = 0;
};
}
}
}
#endif
