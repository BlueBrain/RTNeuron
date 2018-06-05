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

#include <viewer/StereoAnimationPath.h>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/io_utils>

#include <boost/lexical_cast.hpp>
#include <limits>
#include <lunchbox/log.h>
#include <stdexcept>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper functions
*/
namespace
{
template <typename T>
std::string lexical_cast(const T& t)
{
    return boost::lexical_cast<std::string>(t);
}

struct skip_white_space
{
    skip_white_space(size_t& line_count)
        : _line_count(&line_count)
    {
    }

    std::istream& operator()(std::istream& in) const
    {
        while (isspace(in.peek()))
        {
            if (in.get() == '\n')
            {
                ++(*_line_count);
            }
        }
        return in;
    }

    size_t* const _line_count;
};

std::istream& operator>>(std::istream& in, const skip_white_space& ws)
{
    return ws(in);
}
}

/*
  Constructors
*/
StereoAnimationPath::StereoAnimationPath()
    : _interpolationMode(LINEAR)
{
}

StereoAnimationPath::StereoAnimationPath(const StereoAnimationPath& path,
                                         const osg::CopyOp& op)
    : Object(path, op)
    , _timeControlPointMap(path._timeControlPointMap)
    , _interpolationMode(path._interpolationMode)
{
}

/*
  Member functions
*/

bool StereoAnimationPath::getInterpolatedControlPoint(double time,
                                                      ControlPoint& point) const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    if (_timeControlPointMap.empty())
        return false;

    /* Finding the neighbour control points */
    TimeControlPointMap::const_iterator second =
        _timeControlPointMap.lower_bound(time);
    if (second == _timeControlPointMap.begin())
    {
        point = second->second;
    }
    else if (second != _timeControlPointMap.end())
    {
        TimeControlPointMap::const_iterator first = second;
        --first;
        /* we have both a lower bound and the next item. */
        double delta_time = second->first - first->first;

        if (delta_time == 0.0)
        {
            point = first->second;
        }
        else
        {
            double ratio = (time - first->first) / delta_time;
            ControlPoint p1 = first->second;
            ControlPoint p2 = second->second;

            if (_interpolationMode == SPLINE && _timeControlPointMap.size() > 2)
                /* Spline interpolated position */
                point.position =
                    osg::Vec3(_x.eval(time), _y.eval(time), _z.eval(time));
            else
                point.position =
                    (p1.position * (1 - ratio) + p2.position * ratio);
            point.rotation.slerp(ratio, p1.rotation, p2.rotation);
            point.scale = p1.scale * (1 - ratio) + p2.scale * ratio;
            point.stereoCorrection = (p1.stereoCorrection * (1 - ratio) +
                                      p2.stereoCorrection * ratio);
        }
    }
    else
    {
        point = _timeControlPointMap.rbegin()->second;
    }

    return true;
}

void StereoAnimationPath::recalculateSplines() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    unsigned int size = _timeControlPointMap.size();

    double* t = new double[size];
    double* x = new double[size];
    double* y = new double[size];
    double* z = new double[size];

    /* Filling input data for spline calculation. */
    TimeControlPointMap::const_iterator iter;
    unsigned int i;
    for (i = 0, iter = _timeControlPointMap.begin();
         iter != _timeControlPointMap.end(); ++iter, ++i)
    {
        t[i] = iter->first;
        osg::Vec3 position = iter->second.position;
        x[i] = position[0];
        y[i] = position[1];
        z[i] = position[2];
    }

    /* Computing splines. */
    _x.setData(t, x, size);
    _y.setData(t, y, size);
    _z.setData(t, z, size);

    delete[] t;
    delete[] x;
    delete[] y;
    delete[] z;
}

void StereoAnimationPath::read(std::istream& in)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    _timeControlPointMap.clear();

    size_t lineNumber = 0;
    std::string str;
    in >> skip_white_space(lineNumber);
    while (!in.eof() && !std::getline(in, str).fail())
    {
        ++lineNumber;
        double time;
        osg::Vec3d position;
        osg::Quat rotation;
        double stereoCorrection;

        std::stringstream line(str);
        line >> time >> position.x() >> position.y() >> position.z() >>
            rotation.x() >> rotation.y() >> rotation.z() >> rotation.w() >>
            stereoCorrection;

        time /= 1000.0; /* Time is in milliseconds in file format */
        if (line.fail())
        {
            LBTHROW(std::runtime_error("Reading stereo animation path line: " +
                                       lexical_cast(lineNumber)));
        }
        _timeControlPointMap[time] =
            ControlPoint(position, rotation, osg::Vec3(1, 1, 1),
                         stereoCorrection);
        in >> skip_white_space(lineNumber);
    }

    if (_timeControlPointMap.size() < 3)
        _interpolationMode = LINEAR;
    if (_interpolationMode == SPLINE)
        recalculateSplines();
}

void StereoAnimationPath::write(std::ostream& out) const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    int precision = out.precision();
    out.precision(15);

    for (TimeControlPointMap::const_iterator i = _timeControlPointMap.begin();
         i != _timeControlPointMap.end(); ++i)
    {
        const ControlPoint& point = i->second;

        osg::Vec3 position(point.position);
        osg::Quat rotation(point.rotation);

        out << i->first * 1000.0 << ' ' /* Time is in milliseconds
                                             inside the file */
            << position[0] << ' ' << position[1] << ' ' << position[2] << ' '
            << rotation[0] << ' ' << rotation[1] << ' ' << rotation[2] << ' '
            << rotation[3] << ' ' << point.stereoCorrection << std::endl;
    }

    out.precision(precision);
}
}
}
}
