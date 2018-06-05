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

#ifndef RTNEURON_API_CAMERAMANIPULATOR_H
#define RTNEURON_API_CAMERAMANIPULATOR_H

#include <boost/utility.hpp>

namespace osgGA
{
class CameraManipulator;
}

namespace bbp
{
namespace rtneuron
{
/**
   \brief Base class for all camera manipulators
*/
class CameraManipulator : public boost::noncopyable
{
public:
    /*--- Public constructors/destructor */

    /** @name C++ public interface wrapped in Python */
    ///@{

    virtual ~CameraManipulator() {}
    ///@}

    /*--- Public member functions C++ only ---*/

    /** @name C++ only public interface */
    ///@{

    /**
       @internal
     */
    virtual osgGA::CameraManipulator* osgManipulator() = 0;

    ///@}

protected:
    /*--- Protected constructors/destructor ---*/

    CameraManipulator() {}
};
}
}
#endif
