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

#include "CameraPathManipulator.h"
#include "../AttributeMap.h"
#include "../Camera.h"
#include "../View.h"
#include "CameraPath.h"

#include "util/vec_to_vec.h"
#include "viewer/StereoAnimationPath.h"
#include "viewer/StereoAnimationPathManipulator.h"

#include <fstream>
#include <lunchbox/log.h>
#include <stdexcept>

typedef bbp::rtneuron::core::StereoAnimationPath::TimeControlPointMap
    ControlPointMap;

namespace std
{
template <>
struct less<ControlPointMap::iterator>
    : binary_function<ControlPointMap::iterator, ControlPointMap::iterator,
                      bool>
{
    bool operator()(const ControlPointMap::iterator& x,
                    const ControlPointMap::iterator& y) const
    {
        return x->first < y->first;
    }
};
}

namespace bbp
{
namespace rtneuron
{
/******************************************************************************
   CameraPath
 ******************************************************************************/

/* Forward definiton needed by KeyFrame */
class CameraPath::Impl : public core::StereoAnimationPath
{
public:
    void setKeyFrames(std::map<double, KeyFramePtr> frames);
    void addKeyFrame(double seconds, const KeyFramePtr& frame);
    void replaceKeyFrame(size_t index, const KeyFramePtr& frame);
    KeyFramePtr getKeyFrame(size_t index);
    void removeKeyFrame(size_t index);
    std::vector<std::pair<double, KeyFramePtr>> getKeyFrames();
    void clear();
    void load(const std::string& filename);
    void save(const std::string& filename) const;

    void controlPointsUpdated()
    {
        if (_interpolationMode == SPLINE && _timeControlPointMap.size() > 2)
            recalculateSplines();
    }

private:
    void _checkNotContained(const KeyFramePtr& frame) const
    {
        for (KeyFrames::const_iterator k = _keyframes.begin();
             k != _keyframes.end(); ++k)
        {
            if (k->second == frame)
                LBTHROW(std::runtime_error(
                    "KeyFrame already in camera path, "
                    "copy to a new object before inserting"));
        }
    }

    /* We are duplicating the information, but there's no other way we
       can ensure the correcness of modification methods than need to
       invalidate KeyFrame if we don't store reference to them. */
    typedef std::map<TimeControlPointMap::iterator, KeyFramePtr> KeyFrames;
    KeyFrames _keyframes;
};

/* KeyFrame */

class CameraPath::KeyFrame::Impl
{
public:
    Impl()
        : position(0.f, 0.f, 0.f)
        , orientation(1.f, 0.f, 0.f, 0.f)
        , stereoCorrection(1.f)
        , path(0)
    {
    }
    Vector3f position;
    Orientation orientation;
    double stereoCorrection;

    core::StereoAnimationPath::TimeControlPointMap::iterator keyframe;
    CameraPath::Impl* path;
};

CameraPath::KeyFrame::KeyFrame()
    : _impl(new Impl)
{
}

CameraPath::KeyFrame::KeyFrame(const Vector3f& position,
                               const Orientation& orientation,
                               double stereoCorrection)
    : _impl(new Impl)
{
    _impl->position = position;
    _impl->orientation = orientation;
    _impl->stereoCorrection = stereoCorrection;
}

CameraPath::KeyFrame::KeyFrame(View& view)
    : _impl(new Impl)
{
    const Camera& camera = *view.getCamera();
    Vector3f position;
    Orientation orientation;
    camera.getView(position, orientation);

    _impl->position = position;
    _impl->orientation = orientation;
    _impl->stereoCorrection = view.getAttributes()("stereo_correction", 1.0);
}

CameraPath::KeyFrame::~KeyFrame()
{
    delete _impl;
}

Vector3f CameraPath::KeyFrame::getPosition() const
{
    return _impl->position;
}

void CameraPath::KeyFrame::setPosition(const Vector3f& position)
{
    _impl->position = position;
    if (_impl->path)
    {
        assert(_impl->keyframe != _impl->path->getTimeControlPointMap().end());
        _impl->keyframe->second.position =
            osg::Vec3(position[0], position[1], position[2]);
        _impl->path->controlPointsUpdated();
    }
}

Orientation CameraPath::KeyFrame::getOrientation() const
{
    return _impl->orientation;
}

void CameraPath::KeyFrame::setOrientation(const Orientation& orientation)
{
    _impl->orientation = orientation;
    if (_impl->path)
    {
        assert(_impl->keyframe != _impl->path->getTimeControlPointMap().end());
        double angle = orientation[3] / 180 * M_PI;
        _impl->keyframe->second.rotation =
            osg::Quat(angle, osg::Vec3(orientation[0], orientation[1],
                                       orientation[2]));
        _impl->path->controlPointsUpdated();
    }
}

double CameraPath::KeyFrame::getStereoCorrection() const
{
    return _impl->stereoCorrection;
}

void CameraPath::KeyFrame::setStereoCorrection(const double correction)
{
    _impl->stereoCorrection = correction;
    if (_impl->path)
    {
        assert(_impl->keyframe != _impl->path->getTimeControlPointMap().end());
        _impl->keyframe->second.stereoCorrection = correction;
        _impl->path->controlPointsUpdated();
    }
}

/* CameraPath::Impl */

void CameraPath::Impl::setKeyFrames(std::map<double, KeyFramePtr> frames)
{
    /* Invalidating old keyframes */
    clear();
    for (std::map<double, KeyFramePtr>::iterator i = frames.begin();
         i != frames.end(); ++i)
        addKeyFrame(i->first, i->second);

    controlPointsUpdated();
}

void CameraPath::Impl::addKeyFrame(double seconds, const KeyFramePtr& frame)
{
    /* Checking if the key frame is already contained in the camera path
       and refusing to add it again in that case */
    _checkNotContained(frame);

    frame->_impl->path = this;

    /* Creating the control point */
    const Vector3f& p = frame->_impl->position;
    const Orientation& o = frame->_impl->orientation;
    StereoAnimationPath::ControlPoint point(
        osg::Vec3(p[0], p[1], p[2]),
        osg::Quat(o[3] / 180 * M_PI, osg::Vec3(o[0], o[1], o[2])),
        osg::Vec3(1, 1, 1), frame->_impl->stereoCorrection);

    /* Adding the control point */
    std::pair<TimeControlPointMap::iterator, bool> insertion =
        _timeControlPointMap.insert(std::make_pair(seconds, point));
    frame->_impl->keyframe = insertion.first;
    if (!insertion.second)
    {
        /* There's already a control point at that time.
           Invalidating the old keyframe and replacing the point */
        KeyFramePtr& k = _keyframes[insertion.first];
        k->_impl->path = 0;
        k = frame;
        insertion.first->second = point;
    }
    else
    {
        _keyframes[insertion.first] = frame;
    }

    controlPointsUpdated();
}

void CameraPath::Impl::replaceKeyFrame(size_t index, const KeyFramePtr& frame)
{
    _checkNotContained(frame);

    if (index >= _timeControlPointMap.size())
        LBTHROW(std::runtime_error("Index out of bounds"));

    frame->_impl->path = this;

    /* Creating the control point */
    const Vector3f& p = frame->_impl->position;
    const Orientation& o = frame->_impl->orientation;
    StereoAnimationPath::ControlPoint point(
        osg::Vec3(p[0], p[1], p[2]),
        osg::Quat(o[3] / 180 * M_PI, osg::Vec3(o[0], o[1], o[2])),
        osg::Vec3(1, 1, 1), frame->_impl->stereoCorrection);

    /* Finding the position */
    TimeControlPointMap::iterator iter = _timeControlPointMap.begin();
    for (size_t i = 0; i != index; ++i, ++iter)
        ;

    /* Invalidating the old keyframe and replacing the point */
    KeyFramePtr& k = _keyframes[iter];
    k->_impl->path = 0;
    k = frame;
    iter->second = point;
    frame->_impl->keyframe = iter;

    controlPointsUpdated();
}

CameraPath::KeyFramePtr CameraPath::Impl::getKeyFrame(size_t index)
{
    if (index >= _timeControlPointMap.size())
        LBTHROW(std::runtime_error("Index out of bounds"));

    /* Finding the position */
    TimeControlPointMap::iterator iter = _timeControlPointMap.begin();
    for (size_t i = 0; i != index; ++i, ++iter)
        ;
    return _keyframes[iter];
}

void CameraPath::Impl::removeKeyFrame(size_t index)
{
    if (index >= _timeControlPointMap.size())
        LBTHROW(std::runtime_error("Index out of bounds"));
    TimeControlPointMap::iterator iter = _timeControlPointMap.begin();
    for (size_t i = 0; i != index; ++i, ++iter)
        ;
    _timeControlPointMap.erase(iter);
    // cppcheck-suppress invalidIterator1 is not working in Ubuntu 16.04 for
    // an unknown reason. The following assignment silences the false positive.
    const TimeControlPointMap::iterator i = iter;
    _keyframes.erase(i);

    controlPointsUpdated();
}

std::vector<std::pair<double, CameraPath::KeyFramePtr>>
    CameraPath::Impl::getKeyFrames()
{
    std::vector<std::pair<double, KeyFramePtr>> out;
    for (TimeControlPointMap::iterator k = _timeControlPointMap.begin();
         k != _timeControlPointMap.end(); ++k)
        out.push_back(std::make_pair(k->first, _keyframes[k]));
    return out;
}

void CameraPath::Impl::clear()
{
    for (KeyFrames::iterator i = _keyframes.begin(); i != _keyframes.end(); ++i)
        i->second->_impl->path = 0;
    _keyframes.clear();

    _timeControlPointMap.clear();
}

void CameraPath::Impl::load(const std::string& filename)
{
    std::ifstream in(filename.c_str());
    if (!in)
        LBTHROW(std::runtime_error("Error opening camera path file \"" +
                                   filename + "\"."));
    read(in);

    /* Replacing the key frame refs */
    for (KeyFrames::iterator i = _keyframes.begin(); i != _keyframes.end(); ++i)
        i->second->_impl->path = 0;
    _keyframes.clear();
    /* Creating the new ones */
    for (TimeControlPointMap::iterator k = _timeControlPointMap.begin();
         k != _timeControlPointMap.end(); ++k)
    {
        KeyFramePtr frame(new KeyFrame());
        frame->_impl->path = this;
        frame->_impl->keyframe = k;

        osg::Vec3 pos = k->second.position;
        frame->_impl->position = Vector3f(pos[0], pos[1], pos[2]);
        osg::Quat::value_type angle;
        osg::Vec3 axis;
        k->second.rotation.getRotate(angle, axis);
        angle = angle * 180 / M_PI;
        frame->_impl->orientation =
            Orientation(axis[0], axis[1], axis[2], angle);
        frame->_impl->stereoCorrection = k->second.stereoCorrection;

        _keyframes[k] = frame;
    }
}

void CameraPath::Impl::save(const std::string& filename) const
{
    std::ofstream out(filename.c_str());
    if (!out)
        LBTHROW(std::runtime_error("Cannot open camera path file \"" +
                                   filename + "\" for writing."));
    write(out);
    if (out.fail())
        LBTHROW(std::runtime_error("Error writing camera path file \"" +
                                   filename + "\"."));
}

/* CameraPath */

CameraPath::CameraPath()
    : _impl(new Impl())
{
    _impl->ref();
}

CameraPath::~CameraPath()
{
    _impl->unref();
}

double CameraPath::getStartTime() const
{
    return _impl->getStartTime();
}

double CameraPath::getStopTime() const
{
    return _impl->getStopTime();
}

void CameraPath::setKeyFrames(std::map<double, KeyFramePtr> frames)
{
    _impl->setKeyFrames(frames);
}

void CameraPath::addKeyFrame(double seconds, const KeyFramePtr& frame)
{
    _impl->addKeyFrame(seconds, frame);
}

void CameraPath::addKeyFrame(double seconds, View& view)
{
    _impl->addKeyFrame(seconds, KeyFramePtr(new KeyFrame(view)));
}

void CameraPath::replaceKeyFrame(size_t index, const KeyFramePtr& frame)
{
    _impl->replaceKeyFrame(index, frame);
}

CameraPath::KeyFramePtr CameraPath::getKeyFrame(size_t index)
{
    return _impl->getKeyFrame(index);
}

void CameraPath::removeKeyFrame(size_t index)
{
    _impl->removeKeyFrame(index);
}

std::vector<std::pair<double, CameraPath::KeyFramePtr>>
    CameraPath::getKeyFrames()
{
    return _impl->getKeyFrames();
}

void CameraPath::clear()
{
    _impl->clear();
}

void CameraPath::load(const std::string& fileName)
{
    _impl->load(fileName);
}

void CameraPath::save(const std::string& fileName) const
{
    _impl->save(fileName);
}

/******************************************************************************
   CameraPathManipulator
 ******************************************************************************/

/*
  Nested classes
*/
class CameraPathManipulator::Impl : public core::StereoAnimationPathManipulator
{
};

/*
  Constructors/destructor
*/
CameraPathManipulator::CameraPathManipulator()
    : _impl(new Impl())
{
    _impl->ref();
}

CameraPathManipulator::~CameraPathManipulator()
{
    _impl->unref();
}

/*
  Member functions
*/
void CameraPathManipulator::load(const std::string& fileName)
{
    _impl->load(fileName);
}

void CameraPathManipulator::setPath(const CameraPathPtr& path)
{
    _impl->setPath(path->_impl);
}

void CameraPathManipulator::setPlaybackStart(float start)
{
    _impl->setPlaybackStart(start);
}

float CameraPathManipulator::getPlaybackStart() const
{
    return _impl->getPlaybackStart();
}

void CameraPathManipulator::setPlaybackStop(float stop)
{
    _impl->setPlaybackStop(stop);
}

float CameraPathManipulator::getPlaybackStop() const
{
    return _impl->getPlaybackStop();
}

void CameraPathManipulator::setPlaybackInterval(float start, float stop)
{
    _impl->setPlaybackInterval(start, stop);
}

void CameraPathManipulator::setFrameDelta(float milliseconds)
{
    _impl->setFrameDelta(milliseconds);
}

float CameraPathManipulator::getFrameDelta() const
{
    return _impl->getFrameDelta();
}

void CameraPathManipulator::setLoopMode(LoopMode loopMode)
{
    core::StereoAnimationPathManipulator::LoopMode mode;
    switch (loopMode)
    {
    case LOOP_REPEAT:
        mode = core::StereoAnimationPathManipulator::LOOP;
        break;
    case LOOP_SWING:
        mode = core::StereoAnimationPathManipulator::SWING;
        break;
    case LOOP_NONE:
    default:
        mode = core::StereoAnimationPathManipulator::NO_LOOP;
        break;
    }
    _impl->setLoopMode(mode);
}

CameraPathManipulator::LoopMode CameraPathManipulator::getLoopMode() const
{
    switch (_impl->getLoopMode())
    {
    case core::StereoAnimationPathManipulator::LOOP:
        return LOOP_REPEAT;
    case core::StereoAnimationPathManipulator::SWING:
        return LOOP_SWING;
    case core::StereoAnimationPathManipulator::NO_LOOP:
    default:
        return LOOP_NONE;
    }
}

void CameraPathManipulator::getKeyFrame(const float milliseconds,
                                        Vector3f& position,
                                        Orientation& orientation,
                                        double& stereoCorrection) const
{
    if (!_impl->getAnimationPath())
        throw std::runtime_error("Camera path not set");

    core::StereoAnimationPath::ControlPoint keyFrame;
    _impl->getAnimationPath()->getInterpolatedControlPoint(milliseconds,
                                                           keyFrame);
    position = core::vec_to_vec(keyFrame.position);
    const osg::Quat attitude = keyFrame.rotation;
    osg::Vec3 axis;
    double angle;
    attitude.getRotate(angle, axis);
    orientation.x() = axis[0];
    orientation.y() = axis[1];
    orientation.z() = axis[2];
    orientation.w() = angle / M_PI * 180.0;

    stereoCorrection = keyFrame.stereoCorrection;
}

osgGA::CameraManipulator* CameraPathManipulator::osgManipulator()
{
    return _impl;
}
}
}
