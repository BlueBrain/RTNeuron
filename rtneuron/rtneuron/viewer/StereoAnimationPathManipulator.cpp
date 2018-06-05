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

#include <fstream>
#include <stdexcept>

#include <viewer/StereoAnimationPath.h>
#include <viewer/StereoAnimationPathManipulator.h>

#include <lunchbox/log.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Constructors
*/
StereoAnimationPathManipulator::StereoAnimationPathManipulator(
    StereoAnimationPath* animationPath)
    : _animationPath(animationPath)
    , _frameDelta(0)
    , _playbackStart(0)
    , _playbackStop(std::numeric_limits<double>::max())
    , _loopMode(LOOP)
    , _currentTime(0)
    , _realTimeReference(0)
    , _stereoCorrection(1)
{
}

StereoAnimationPathManipulator::StereoAnimationPathManipulator(
    const std::string& filename)
    : _frameDelta(0)
    , _playbackStart(0)
    , _playbackStop(std::numeric_limits<double>::max())
    , _loopMode(LOOP)
    , _currentTime(0)
    , _realTimeReference(0)
    , _stereoCorrection(1)
{
    load(filename);
}

/*
  Destructor
*/
StereoAnimationPathManipulator::~StereoAnimationPathManipulator()
{
}

/*
  Member functions
*/
void StereoAnimationPathManipulator::load(const std::string& filename)
{
    std::ifstream in(filename.c_str());

    if (!in)
        LBTHROW(std::runtime_error("Cannot open animation path file \"" +
                                   filename + "\".\n"));

    osg::ref_ptr<StereoAnimationPath> path(new StereoAnimationPath());
    path->read(in);

    if (!::getenv("RTNEURON_FORCE_LINEAR_PATH_INTERPOLATION") &&
        (path->getStopTime() - path->getStartTime()) /
                (float)path->getTimeControlPointMap().size() >
            0.2)
    {
        path->setPositionInterpolationMode(StereoAnimationPath::SPLINE);
    }

    if (path->getStartTime() == path->getStopTime())
        _loopMode = NO_LOOP;

    _animationPath = path;
}

void StereoAnimationPathManipulator::setPath(StereoAnimationPath* path)
{
    if (path->getStartTime() == path->getStopTime())
        _loopMode = NO_LOOP;

    _animationPath = path;
}

void StereoAnimationPathManipulator::setAnimationPath(
    StereoAnimationPath* animationPath)
{
    _animationPath = animationPath;
}

void StereoAnimationPathManipulator::home(
    const osgGA::GUIEventAdapter& ev, osgGA::GUIActionAdapter& actionAdapter)
{
    home(ev.getTime());
    actionAdapter.requestRedraw();
}

bool StereoAnimationPathManipulator::completed(double currentTime)
{
    if (_loopMode != NO_LOOP)
        return false;

    double time;
    if (_frameDelta == 0)
        time = currentTime - _realTimeReference;
    else
        time = _currentTime;

    return time > std::min(_animationPath->getStopTime(), _playbackStop);
}

void StereoAnimationPathManipulator::home(double currentTime)
{
    if (_animationPath.valid())
    {
        double start = std::max(_animationPath->getStartTime(), _playbackStart);

        _realTimeReference = currentTime - start;
        _currentTime = start;

        _handleFrame(start);
    }
}

bool StereoAnimationPathManipulator::handle(
    const osgGA::GUIEventAdapter& ev, osgGA::GUIActionAdapter& actionAdapter)
{
    if (!valid())
        return false;

    if (ev.getEventType() == osgGA::GUIEventAdapter::FRAME)
    {
        double time;
        if (_frameDelta == 0)
            time = ev.getTime() - _realTimeReference;
        else
            time = _currentTime;

        time = _adjustedTime(time);

        if (_handleFrame(time))
            actionAdapter.requestRedraw();

        if (_frameDelta != 0)
            _currentTime += _frameDelta;
    }
    return false;
}

void StereoAnimationPathManipulator::getUsage(
    osg::ApplicationUsage& usage) const
{
    usage.addKeyboardMouseBinding("AnimationPath: Space",
                                  "Reset the viewing position to start "
                                  "of animation");
    usage.addKeyboardMouseBinding("AnimationPath: p",
                                  "Pause/resume animation.");
}

double StereoAnimationPathManipulator::_adjustedTime(double time)
{
    double start = std::max(_animationPath->getStartTime(), _playbackStart);
    double stop = std::min(_animationPath->getStopTime(), _playbackStop);
    double period = stop - start;

    switch (_loopMode)
    {
    case SWING:
    {
        double modulatedTime = (time - start) / (period * 2);
        double fractionPart = (modulatedTime - floor(modulatedTime)) * 2;
        if (fractionPart > 1.0)
            fractionPart = 2.0 - fractionPart;
        time = start + fractionPart * period;
        break;
    }
    case LOOP:
    {
        double modulatedTime = (time - start) / period;
        double fractionPart = modulatedTime - floor(modulatedTime);
        time = start + fractionPart * period;
        break;
    }
    case NO_LOOP:
        /* No need to modulate the time. */
        break;
    }
    return time;
}

bool StereoAnimationPathManipulator::_handleFrame(const double time)
{
    if (completed(time))
        return false;

    StereoAnimationPath::ControlPoint point;
    _animationPath->getInterpolatedControlPoint(time, point);
    point.getMatrix(_matrix);

    _stereoCorrection = point.stereoCorrection;

    return true;
}
}
}
}
