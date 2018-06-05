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

#ifndef RTNEURON_API_UI_TRACKBALLMANIPULATOR_H
#define RTNEURON_API_UI_TRACKBALLMANIPULATOR_H

#include "../CameraManipulator.h"
#include "../types.h"

#include <osgGA/TrackballManipulator>

namespace bbp
{
namespace rtneuron
{
/**
   \brief Default mouse-based manipulator from OpenSceneGraph.

   Use the right button to zoom, middle button to pan and left button to
   rotate.
 */
class TrackballManipulator : public CameraManipulator
{
public:
    /*--- Public constructors/destructor */

    /** @name C++ public interface wrapped in Python */
    ///@{

    TrackballManipulator();

/**
   @param eye The reference camera position
   @param center The reference pivot point for rotations
   @param up The direction of the up direction in the reference orientation.
   The "look at" vector is (center - eye).
   @version 2.3
 */
#ifdef DOXGEN_TO_BREATHE
    void setHomePosition(object eye, object center, object up) const;
#else
    void setHomePosition(const Vector3f& eye, const Vector3f& center,
                         const Vector3f& up);
#endif

/**
   \if pybind
   @return A tuple (eye, center, up) where each one is an [x, y, z] vector
   \else
   @see setHomePosition
   \endif
   @version 2.3
 */
#ifdef DOXGEN_TO_BREATHE
    typle getHomePosition();
#else
    void getHomePosition(Vector3f& eye, Vector3f& center, Vector3f& up);
#endif

    ///@}

    /*--- Public member functions C++ only ---*/

    /** @name C++ only public interface */
    ///@{

    virtual osgGA::TrackballManipulator* osgManipulator()
    {
        return _manipulator;
    }

    ///@}

private:
    osg::ref_ptr<osgGA::TrackballManipulator> _manipulator;
};
}
}
#endif
