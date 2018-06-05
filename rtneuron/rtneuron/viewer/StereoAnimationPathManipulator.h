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

#ifndef RTNEURON_STEREOANIMATIONPATHMANIPULATOR_H
#define RTNEURON_STEREOANIMATIONPATHMANIPULATOR_H

#include "../types.h"

#include <osg/Version>
#include <osgGA/CameraManipulator>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class StereoAnimationPath;

/*! \brief Animation path manipulator that extends osg::AnimationPath to add
  stereo fusion distance interpolation.
*/
class StereoAnimationPathManipulator : public osgGA::CameraManipulator
{
public:
    /*--- Declarations ---*/
    enum LoopMode
    {
        NO_LOOP = 0,
        LOOP,
        SWING
    };

    /*--- Public constructors/destructor ---*/

    StereoAnimationPathManipulator(StereoAnimationPath* animationPath = 0);

    StereoAnimationPathManipulator(const std::string& filename);

    ~StereoAnimationPathManipulator();

    /*--- Public member functions ---*/

    /**
       Loads a camera path from a file.

       If the camera path contains a single keyframe the loopmode is
       automatically set to NO_LOOP.

       Throw is an error occurs reading the file.
     */
    void load(const std::string& filename);

    /**
       Replaces the current path with a different one.

       The current matrix is not replaced but the loop mode is changed to
       NO_LOOP if the camera path contains a single keyframe.
     */
    void setPath(StereoAnimationPath* animationPath);

    /**
       Sets the delta time between keyframe samples (in milliseconds)

       Use a positive value to set a fixed delta between rendered frames.
       A value equal to 0 means that the camera path has to be played back
       in real-time.
     */
    void setFrameDelta(double milliseconds)
    {
        _frameDelta = milliseconds / 1000.0;
    }

    /**
      Return the frame delta in milliseconds.
    */
    double getFrameDelta() const { return _frameDelta * 1000.0; }
    /**
     */
    void setPlaybackStart(float milliseconds)
    {
        _playbackStart = milliseconds / 1000;
    }

    /**
     */
    float getPlaybackStart() const { return _playbackStart * 1000.0; }
    /**
     */
    void setPlaybackStop(float milliseconds)
    {
        _playbackStop = milliseconds / 1000.0;
    }

    /*+
     */
    float getPlaybackStop() const { return _playbackStop * 1000.0; }
    /**
     */
    void setPlaybackInterval(float start, float stop)
    {
        setPlaybackStart(start);
        setPlaybackStop(stop);
    }

    /**
     */
    void setLoopMode(LoopMode loopMode) { _loopMode = loopMode; }
    /**
     */
    LoopMode getLoopMode() const { return _loopMode; }
    virtual const char* className() const { return "AnimationPath"; }
    /**
        set the position of the matrix manipulator using a 4x4 Matrix.
    */
    virtual void setByMatrix(const osg::Matrixd& matrix) { _matrix = matrix; }
    /**
        set the position of the matrix manipulator using a 4x4 Matrix.
    */
    virtual void setByInverseMatrix(const osg::Matrixd& matrix)
    {
        _matrix.invert(matrix);
    }

    /**
        get the position of the manipulator as 4x4 Matrix.
    */
    virtual osg::Matrixd getMatrix() const { return _matrix; }
    float getStereoCorrection() const { return _stereoCorrection; }
    /**
        get the position of the manipulator as a inverse matrix of the
        manipulator, typically used as a model view matrix.
    */
    virtual osg::Matrixd getInverseMatrix() const
    {
        return osg::Matrixd::inverse(_matrix);
    }

    void setAnimationPath(StereoAnimationPath* animationPath);

    StereoAnimationPath* getAnimationPath() { return _animationPath.get(); }
    const StereoAnimationPath* getAnimationPath() const
    {
        return _animationPath.get();
    }

    bool valid() const { return _animationPath.valid(); }
    void init(const osgGA::GUIEventAdapter& ev,
              osgGA::GUIActionAdapter& actionAdapter)
    {
        home(ev, actionAdapter);
    }

    /**
       Sets the camera matrix at the beginning of the path.

       @param currentTime Reference time used when frame delta is different
       to 0 to now which is the time drift to consider for playback.
     */
    void home(double currentTime);

    /**
       The same as the function above but the time stamp is taken from the
       event adapter and a redraw is requested to the action adapter.
     */
    void home(const osgGA::GUIEventAdapter& ev,
              osgGA::GUIActionAdapter& actionAdapter);

    bool completed(const double currentTime);

    virtual bool handle(const osgGA::GUIEventAdapter& ev,
                        osgGA::GUIActionAdapter& actionAdapter);

    /** Get the keyboard and mouse usage of this manipulator.*/
    virtual void getUsage(osg::ApplicationUsage& usage) const;

protected:
    /*--- Protected member attributes ---*/

    osg::ref_ptr<StereoAnimationPath> _animationPath;

    double _frameDelta;
    double _playbackStart;
    double _playbackStop;
    LoopMode _loopMode;

    double _currentTime;
    double _realTimeReference;

    osg::Matrixd _matrix;
    double _stereoCorrection;

    /*--- Private member functions ---*/
private:
    double _adjustedTime(double time);

    /** Return True if the view matrix or stereo correction have changed */
    bool _handleFrame(const double time);
};
}
}
}
#endif
