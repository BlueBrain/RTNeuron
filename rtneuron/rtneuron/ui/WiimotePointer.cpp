/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
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

#include "WiimotePointer.h"
#include "viewer/Wiimote.h"

#include <osg/Vec3>

namespace bbp
{
namespace rtneuron
{
static const int WM_ASPECT_X = 560;
static const int WM_ASPECT_Y = 420;

static const size_t SAMPLE_SIZE = 10;
/* Smoothing factor: the higher the smoother, increasing the lag though */
static const double SMOOTH_FACTOR = 0.6;

/*
  Helper functions
*/

static uint64_t getMilliSecs()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec * 0.001;
}

/*
  Implementation classes
*/

struct Kalman_t
{
    double angle;
    double bias;
    osg::Matrix2 P;
    // Q = Process covariance noise
    // Indicates how much we trust the accelerometer relative to the gyros
    double Q_angle;
    double Q_gyro;
    // Expected jitter from the accelerometer, in radians
    double R_angle;

    Kalman_t()
        : angle(0)
        , bias(0)
        , Q_angle(0.1f)
        , Q_gyro(0.002f)
        , R_angle(0.001f)
    {
    }
};

struct RK4_t
{
    osg::Vec3 prev;
    osg::Vec3 val3;
    osg::Vec3 val2;
    osg::Vec3 val1;
};

class WiimotePointer::Impl
{
public:
    WiimotePointer* _pointer;
    std::unique_ptr<core::Wiimote> _wiimote;
    uint64_t _lastUpdate;
    osg::Vec3 _lastFixedPos;
    osg::Vec3 _basePoint;
    core::Wiimote::PointingMode _lastMode;
    Kalman_t _kalman;
    RK4_t _orient;
    std::deque<osg::Vec3> _positionData;
    std::deque<osg::Vec3> _orientData;

    Impl(const std::string& hostName, WiimotePointer* pointer)
        : _pointer(pointer)
        , _wiimote(new core::Wiimote(hostName))
        , _lastUpdate(getMilliSecs())
        , _lastMode(core::Wiimote::ORIENT)
    {
    }

    void _onButtonPressed(core::Wiimote::Button button)
    {
        switch (button)
        {
        case core::Wiimote::A_BUTTON:
            _pointer->_select();
            break;
        case core::Wiimote::HOME_BUTTON:
            _orient.prev = osg::Vec3(0, 0, 0);
            break;
        case core::Wiimote::DOWN_BUTTON:
            _pointer->stereoCorrection(0.9);
            break;
        case core::Wiimote::UP_BUTTON:
            _pointer->stereoCorrection(1.1);
            break;
        default:;
        }
    }

    void _onSelectButton();
};

/*
  Constructors/destructor
*/

WiimotePointer::WiimotePointer(const std::string& hostName)
    : _impl(new Impl(hostName, this))
{
    _impl->_wiimote->buttonPressed.connect(
        boost::bind(&Impl::_onButtonPressed, _impl.get(), _1));
}

WiimotePointer::~WiimotePointer()
{
}

/*
  Member functions
*/

void WiimotePointer::update()
{
    if (_impl->_wiimote.get() == 0)
        return;

    const core::Wiimote::SampleData sample = _impl->_wiimote->sample();

    switch (_impl->_wiimote->getPointingMode())
    {
    /* Pointer position */
    case core::Wiimote::SHIFT:
    {
        osg::Vec3 wmPos(0, 0, 0);
        if (!_impl->_positionData.empty())
            wmPos = _impl->_positionData.back();

        if (_impl->_lastMode == core::Wiimote::ORIENT)
            /* Pointing mode just changed */
            _impl->_basePoint = _center(sample.ir);

        if (sample.ir.length()) /* IR value != (0,0,0) */
        {
            wmPos =
                _impl->_lastFixedPos + _center(sample.ir) - _impl->_basePoint;
        }

        _impl->_positionData.push_back(wmPos);

        if (_impl->_positionData.size() > SAMPLE_SIZE)
            _impl->_positionData.pop_front();

        break;
    }

    /* Pointer orientation */
    case core::Wiimote::ORIENT:
    {
        if (_impl->_lastMode == core::Wiimote::SHIFT)
        {
            /* Pointing mode just changed */
            if (sample.ir.length())
                _impl->_lastFixedPos += _center(sample.ir) - _impl->_basePoint;
        }

        osg::Vec3 gyroRate = sample.gyro;
        const osg::Vec3 accelOrient = sample.wmOrient;

        /* Stabilize values when small changes are registered */
        gyroRate[0] = fabs(gyroRate[0]) < 2.5f ? 0.0f : gyroRate[0];
        gyroRate[2] = fabs(gyroRate[2]) < 2.5f ? 0.0f : gyroRate[2];

        /* Measure delta time */
        const uint64_t now = getMilliSecs();
        const uint64_t deltaMs = now - _impl->_lastUpdate;
        _impl->_lastUpdate = now;

        const double deltaSec = deltaMs * 0.001f;

        /* Wiimote pointing near the middle (horizontally) of the sensor bar */
        if (fabs(sample.ir[0] - (WM_ASPECT_X / 2)) < 10.f)
            /* Reset */
            _impl->_orient.prev = osg::Vec3(0, 0, 0);
        else
        {
            /* Integrate direction */
            _computeRK4(gyroRate * deltaSec);

            _kalmanPredict(gyroRate.x(), deltaSec);
            _kalmanCorrect(osg::DegreesToRadians(accelOrient.x()));

            /* Clamping needed to keep the pointer oriented to the front:
            angle between -90 and 90 degrees.
            Without this, the value could be ridiculously high and corrupt
            every subsequent transformation applied to the geometry. */
            _impl->_kalman.angle =
                std::max(-M_PI, std::min(_impl->_kalman.angle, M_PI));

            _impl->_orientData.push_back(
                osg::Vec3(osg::RadiansToDegrees(_impl->_kalman.angle),
                          _impl->_orient.prev.y(), _impl->_orient.prev.z()));
        }

        if (_impl->_orientData.size() > SAMPLE_SIZE)
            _impl->_orientData.pop_front();

        break;
    }
    default:
        break;
    }

    _impl->_lastMode = _impl->_wiimote->getPointingMode();

    _updatePointerMatrix(_smooth(_impl->_positionData),
                         _smooth(_impl->_orientData), _manip);
}

void WiimotePointer::_kalmanPredict(const double gyroRate, const double dt)
{
    Kalman_t& kalman = _impl->_kalman;

    // Project the state ahead
    kalman.angle += (gyroRate - kalman.bias) * dt;

    // Project the error covariance ahead
    kalman.P.set(kalman.P(0, 0) + dt * dt * kalman.P(1, 1) -
                     dt * (kalman.P(1, 0) + kalman.P(0, 1)) + kalman.Q_angle,
                 kalman.P(0, 1) - dt * kalman.P(1, 1),
                 kalman.P(1, 0) - dt * kalman.P(1, 1),
                 kalman.P(1, 1) + kalman.Q_gyro);
}

void WiimotePointer::_kalmanCorrect(const double actualAngle) const
{
    Kalman_t& kalman = _impl->_kalman;

    // Compute Kalman gain
    const double S = kalman.P(0, 0) + kalman.R_angle;
    const osg::Vec2 K(kalman.P(0, 0) / S, kalman.P(1, 0) / S);

    // Update estimate with measurement from accelerometer
    const double m_diff = actualAngle - kalman.angle;
    kalman.angle += K[0] * m_diff;
    kalman.bias += K[1] * m_diff;

    // Update the error covariance
    kalman.P.set(kalman.P(0, 0) - kalman.P(0, 0) * K[0],
                 kalman.P(0, 1) - kalman.P(0, 1) * K[0],
                 kalman.P(1, 0) - kalman.P(0, 0) * K[1],
                 kalman.P(1, 1) - kalman.P(0, 1) * K[1]);
}

void WiimotePointer::_computeRK4(const osg::Vec3 val0)
{
    RK4_t& rk4 = _impl->_orient;

    rk4.prev += (rk4.val3 + rk4.val2 * 2 + rk4.val1 * 2 + val0) / 6.0f;
    rk4.val3 = rk4.val2;
    rk4.val2 = rk4.val1;
    rk4.val1 = val0;
}

osg::Vec3 WiimotePointer::_smooth(const std::deque<osg::Vec3>& data)
{
    osg::Vec3 smoothed(0, 0, 0);

    double weight = 1.0;
    for (size_t i = data.size(); i-- > 0;)
    {
        smoothed = smoothed * (1 - weight) + data[i] * weight;
        weight *= SMOOTH_FACTOR;
    }

    return smoothed;
}

osg::Vec3 WiimotePointer::_center(const osg::Vec3 pos) const
{
    return pos - osg::Vec3(WM_ASPECT_X, WM_ASPECT_Y, 0) / 2;
}

void WiimotePointer::_select()
{
    /** \todo JH: Is this the most up to date matrix when this slot is
        invoked ? What's the effect of the lag with heavy scenes. */
    const osg::Vec4 position4 = osg::Vec4(0, 0, 0, 1) * _pointerMatrix;
    osg::Vec3 position = osg::Vec3(position4[0], position4[1], position4[2]);

    osg::Vec3 look = osg::Vec3(0, 0, -1) * _pointerMatrix;

    /* From camera to world coordinates */
    transformCoord(position);
    transformCoord(look);

    const osg::Vec3 direction = look - position;
    pick(position, direction);
}
}
}
