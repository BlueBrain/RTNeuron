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

#ifndef RTNEURON_API_UI_CAMERAPATHMANIPULATOR_H
#define RTNEURON_API_UI_CAMERAPATHMANIPULATOR_H

#include "../CameraManipulator.h"
#include "../types.h"

namespace bbp
{
namespace rtneuron
{
/**
   \brief Camera manipulator to play a CameraPath.
 */
class CameraPathManipulator : public CameraManipulator
{
public:
    /*--- Public declarations ---*/

    /**
       The loop mode defines what to do when the end of the camera path is
       reached.
     */
    enum LoopMode
    {
        LOOP_NONE,   //!< Do nothing
        LOOP_REPEAT, //!< Start over the camera path
        LOOP_SWING   /**< Play the camera path in reserve until the start
                        is reached and repeat */
    };

    /*--- Public constructors/destructor ---*/

    CameraPathManipulator();

    ~CameraPathManipulator();

    /*--- Public member functions ---*/

    /** @name C++ public interface wrapped in Python */
    ///@{

    /**
       Loads a camera path from a file.

       If the camera path contains a single keyframe the loopmode is
       automatically set to LOOP_NONE.

       Throws if an error occurs reading the file.
     */
    void load(const std::string& fileName);

    void setPath(const CameraPathPtr& cameraPath);

    /**
       \brief Overrides the start time of the camera path
       @param start Milliseconds
     */
    void setPlaybackStart(float start);

    float getPlaybackStart() const;

    /**
       Overrides the stop time of the camera path

       @param end Milliseconds
     */
    void setPlaybackStop(float end);

    float getPlaybackStop() const;

    /**
       Overrides the start and stop time of the camera path.

       @param start Milliseconds
       @param end Milliseconds
     */
    void setPlaybackInterval(float start, float end);

    void getPlaybackInterval(float& start, float& end);

    /**
       Sets the delta time between keyframe samples (in milliseconds)

       Use a positive value to set a fixed delta between rendered frames.
       A value equal to 0 means that the camera path has to be played back
       in real-time.
    */
    void setFrameDelta(float milliseconds);

    /**
       Return the frame delta in milliseconds
    */
    float getFrameDelta() const;

    void setLoopMode(LoopMode loopMode);

    LoopMode getLoopMode() const;

/**
   Get a interpolated keyframe at the given timestamp

   \if pybind
   @return A tuple (position, (axis, angle), stereoCorection) where
   position and axis are [x, y, z] lists and the angle is in degrees.
   \endif
   @version 2.4
 */
#ifdef DOXGEN_TO_BREATHE
    object getKeyFrame(const float milliseconds) const;
#else
    void getKeyFrame(const float milliseconds, Vector3f& position,
                     Orientation& orientation, double& stereoCorrection) const;
#endif

    ///@}

    /*--- Public member functions (C++ only) ---*/

    /** @name C++ only public interface */
    ///@{

    osgGA::CameraManipulator* osgManipulator();

    ///@}

private:
    /*--- Private member attributes ---*/

    class Impl;
    Impl* _impl;
};
}
}
#endif
