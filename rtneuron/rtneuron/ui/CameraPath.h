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

#ifndef RTNEURON_API_UI_CAMERAPATH_H
#define RTNEURON_API_UI_CAMERAPATH_H

#include "../types.h"

#include <map>
#include <string>

namespace bbp
{
namespace rtneuron
{
/**
   \brief A sequence of camera keyframes with timestamps.
*/
class CameraPath
{
    friend class CameraPathManipulator;

public:
    /*--- Public declaration ---*/
    /** \brief Position, orientation and stereo correction of a given
        timestamp */
    class KeyFrame
    {
    public:
        friend class CameraPath;

        KeyFrame();
        KeyFrame(const Vector3f& position, const Orientation& orientation,
                 double stereoCorrection = 1);
        /** @version 2.3 */
        KeyFrame(View& view);
        KeyFrame(const KeyFrame&) = delete;
        KeyFrame& operator=(const KeyFrame&) = delete;
        ~KeyFrame();
        Vector3f getPosition() const;
        void setPosition(const Vector3f& position);
        Orientation getOrientation() const;
        void setOrientation(const Orientation& orientation);
        /** A scalar multiplicative factor for the interocular distance. */
        double getStereoCorrection() const;
        void setStereoCorrection(double correction);

    private:
        class Impl;
        Impl* _impl;
    };
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;

    /*--- Public constructors/destructor ---*/

    /**
       \brief Creates an empty camera path.
    */
    CameraPath();

    ~CameraPath();

    CameraPath(const CameraPath&) = delete;
    CameraPath& operator=(const CameraPath&) = delete;

    /*--- Public member functions ---*/

    /**
       \brief The time of the earliest key frame of NaN if the path is empty.
     */
    double getStartTime() const;

    /**
       \brief The time of the latest key frame of NaN if the path is empty.
     */
    double getStopTime() const;

    /**
       \brief Replaces the current path by a new one
       \if pybind
       @param frames A dictionary with time in seconds as keys and KeyFrames
       as values
       \endif
     */
    void setKeyFrames(std::map<double, KeyFramePtr> frames);

    /**
       \brief Adds a new key frame to the path.

       If there's a key frame with that exact timing, it is replaced.
       Changing the old key frame from an existing reference does not affect
       the camera path.
    */
    void addKeyFrame(double seconds, const KeyFramePtr& frame);

    /**
       \brief Adds a new key frame to the path from the camera and stereo
       correction of the given view.

       If there's a key frame with that exact timing, it is replaced.
       Changing the old key frame from an existing reference does not affect
       the camera path.
    */
    void addKeyFrame(double seconds, View& view);

    /**
       \brief Replaces a keyframe at a given position with a new one.

       Throws if the index is out of bounds.
    */
    void replaceKeyFrame(size_t index, const KeyFramePtr& frame);

    /**
       \brief Returns the key frame at a given position

       Throws if the index is out of bounds.
     */
    KeyFramePtr getKeyFrame(size_t index);

    /**
       \brief Removes the keyframe at the given position.

       Throws if the index is out of bounds.
    */
    void removeKeyFrame(size_t index);

    /**
       \if pybind
       \brief Return a list of tuples (times, KeyFrame).
       \else
       \brief Returns the list of key frames.
       \endif

       If key frames are modified the camera path will be updated.
     */
    std::vector<std::pair<double, KeyFramePtr>> getKeyFrames();

    /**
       \brief Clears the camera path.
    */
    void clear();

    /**
       Loads a camera path from the given file.
     */
    void load(const std::string& fileName);

    /**
       Writes this camera path to the given file.
     */
    void save(const std::string& fileName) const;

private:
    /*--- Private member attributes ---*/

    class Impl;
    Impl* _impl;
};
}
}
#endif
