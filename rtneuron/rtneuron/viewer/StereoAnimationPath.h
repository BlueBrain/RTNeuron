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

#ifndef RTNEURON_STEREOANIMATIONPATH_H
#define RTNEURON_STEREOANIMATIONPATH_H

#include <util/Spline.h>

#include <OpenThreads/Mutex>
#include <osg/Matrixd>
#include <osg/Matrixf>
#include <osg/NodeCallback>
#include <osg/Quat>
#include <osg/Vec3>

#include <iostream>
#include <limits>
#include <map>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*! \brief Animation path with spline-based interpolation and stereo-awareness
  for cameras.

  This class uses splines for position interpolation, linear interpolatiomn
  for scale transformation, and slerp for orientation.

  It also includes control point for a stereo correction value. This value
  can be interpreted as a multiplying factor of the interocular distance.
*/
class StereoAnimationPath : public virtual osg::Object
{
    /* Declarations */
public:
    enum InterpolationMode
    {
        LINEAR,
        SPLINE
    };

    //! Control point with position, rotation scale and stereo correction.
    class ControlPoint
    {
    public:
        ControlPoint(const osg::Vec3& position_ = osg::Vec3(),
                     const osg::Quat& rotation_ = osg::Quat(),
                     const osg::Vec3& scale_ = osg::Vec3(1, 1, 1),
                     float stereoCorrection_ = 1)
            : position(position_)
            , rotation(rotation_)
            , scale(scale_)
            , stereoCorrection(stereoCorrection_)
        {
        }

        inline void getMatrix(osg::Matrixf& matrix) const
        {
            matrix.makeScale(scale);
            matrix.postMult(osg::Matrixf::rotate(rotation));
            matrix.postMult(osg::Matrixf::translate(position));
        }

        inline void getMatrix(osg::Matrixd& matrix) const
        {
            matrix.makeScale(scale);
            matrix.postMult(osg::Matrixd::rotate(rotation));
            matrix.postMult(osg::Matrixd::translate(position));
        }

        inline void getInverse(osg::Matrixf& matrix) const
        {
            matrix.makeScale(1 / scale.x(), 1 / scale.y(), 1 / scale.z());
            matrix.preMult(osg::Matrixf::rotate(rotation.inverse()));
            matrix.preMult(osg::Matrixf::translate(-position));
        }

        inline void getInverse(osg::Matrixd& matrix) const
        {
            matrix.makeScale(1 / scale.x(), 1 / scale.y(), 1 / scale.z());
            matrix.preMult(osg::Matrixd::rotate(rotation.inverse()));
            matrix.preMult(osg::Matrixd::translate(-position));
        }

        /* Member attributes */
    public:
        osg::Vec3d position;
        osg::Quat rotation;
        osg::Vec3d scale;
        float stereoCorrection;
    };

    typedef std::map<double, ControlPoint> TimeControlPointMap;

    /* Constructors */
public:
    StereoAnimationPath();

    StereoAnimationPath(const StereoAnimationPath& path,
                        const osg::CopyOp& op = osg::CopyOp::SHALLOW_COPY);

    META_Object(osg, StereoAnimationPath);

    /* Destructor */
protected:
    virtual ~StereoAnimationPath() {}
    /* Member functions */
public:
    /** Given a specific time, return the transformation matrix for a point. */
    bool getMatrix(double time, osg::Matrixf& matrix) const
    {
        ControlPoint point;
        if (!getInterpolatedControlPoint(time, point))
            return false;
        point.getMatrix(matrix);
        return true;
    }

    /** Given a specific time, return the transformation matrix for a point. */
    bool getMatrix(double time, osg::Matrixd& matrix) const
    {
        ControlPoint point;
        if (!getInterpolatedControlPoint(time, point))
            return false;
        point.getMatrix(matrix);
        return true;
    }

    /** \brief Given a specific time, return the inverse transformation
        matrix for a point. */
    bool getInverse(double time, osg::Matrixf& matrix) const
    {
        ControlPoint point;
        if (!getInterpolatedControlPoint(time, point))
            return false;
        point.getInverse(matrix);
        return true;
    }

    bool getInverse(double time, osg::Matrixd& matrix) const
    {
        ControlPoint point;
        if (!getInterpolatedControlPoint(time, point))
            return false;
        point.getInverse(matrix);
        return true;
    }

    /** \brief Given a specific time, return the local ControlPoint frame
        for a point. */
    virtual bool getInterpolatedControlPoint(double time,
                                             ControlPoint& controlPoint) const;

    /**
       Returns the maximum between the first time in the control point map
       and the user given first time.
     */
    double getStartTime() const
    {
        if (!_timeControlPointMap.empty())
            return _timeControlPointMap.begin()->first;
        else
            return std::numeric_limits<double>::quiet_NaN();
    }

    /**
       Returns the minimum between the last time in the control point map
       and the user given last time.
     */
    double getStopTime() const
    {
        if (!_timeControlPointMap.empty())
            return _timeControlPointMap.rbegin()->first;
        else
            return std::numeric_limits<double>::quiet_NaN();
    }

    double getPeriod() const { return getStartTime() - getStopTime(); }
    void setPositionInterpolationMode(InterpolationMode mode)
    {
        _interpolationMode = mode;
        if (mode == SPLINE && _timeControlPointMap.size() > 2)
            recalculateSplines();
    }

    void setTimeControlPointMap(TimeControlPointMap& points)
    {
        _timeControlPointMap = points;
        if (_interpolationMode == SPLINE && points.size() > 2)
            recalculateSplines();
    }

    const TimeControlPointMap& getTimeControlPointMap() const
    {
        return _timeControlPointMap;
    }

    bool empty() const { return _timeControlPointMap.empty(); }
    /** Read the animation path from a flat ASCII file stream. */
    void read(std::istream& in);

    /** Write the animation path to a flat ASCII file stream. */
    void write(std::ostream& out) const;

protected:
    void recalculateSplines() const;

    /* Member attributes */
protected:
    mutable OpenThreads::Mutex _mutex;

    mutable Spline _x, _y, _z;

    TimeControlPointMap _timeControlPointMap;
    InterpolationMode _interpolationMode;
};
}
}
}
#endif
