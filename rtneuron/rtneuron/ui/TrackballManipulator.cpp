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

#include "TrackballManipulator.h"
#include "util/vec_to_vec.h"

namespace bbp
{
namespace rtneuron
{
namespace
{
const double MAX_DISTANCE_FACTOR = 8;
const double MIN_MAX_DISTANCE = 100000.0;
}

class Manipulator : public osgGA::TrackballManipulator
{
public:
    Manipulator()
        : _maximumDistance(MIN_MAX_DISTANCE)
    {
    }

    void zoomModel(const float dy, bool pushForwardIfNeeded) final
    {
        const float scale = 1.0f + dy;

        float minDist = _minimumDistance;
        if (getRelativeFlag(_minimumDistanceFlagIndex))
            minDist *= _modelSize;

        /* Computing the distance */
        if (_distance * scale > minDist)
        {
            _distance *= scale;
        }
        else
        {
            if (pushForwardIfNeeded)
            {
                /* Pushing the camera pivot forward. */
                osg::Matrixd rotation_matrix(_rotation);
                const osg::Vec3d dv = (osg::Vec3d(0, 0, -1) * rotation_matrix) *
                                      (dy * -_distance);
                _center += dv;
            }
            else
            {
                /* Clamping distance to the minimum */
                _distance = minDist;
            }
        }
        if (_distance > _maximumDistance)
            _distance = _maximumDistance;
    }

    bool handleKeyDown(const osgGA::GUIEventAdapter& ev,
                       osgGA::GUIActionAdapter& actions) final
    {
        const int modifiers = ev.getModKeyMask();
        const float scaling = 0.02;
        auto key = ev.getKey();

        /* For some reason, the keycode in CTRL + key is translated into a
           number between 1 and 26 for alphabetical keys. */
        if (modifiers & osgGA::GUIEventAdapter::MODKEY_CTRL && key > 0 &&
            key < 27)
        {
            key += 'a' - 1;
        }

        switch (key)
        {
        case osgGA::GUIEventAdapter::KEY_Left:
        case 'a':
        {
            if (modifiers & osgGA::GUIEventAdapter::MODKEY_CTRL)
            {
                performMouseDeltaMovement(0.1, 0);
            }
            else
                panModel(scaling * _distance, 0, 0);

            actions.requestRedraw();
            return true;
        }
        case osgGA::GUIEventAdapter::KEY_Right:
        case 'd':
        {
            if (modifiers & osgGA::GUIEventAdapter::MODKEY_CTRL)
            {
                performMouseDeltaMovement(-0.1, 0);
            }
            else
                panModel(-scaling * _distance, 0, 0);

            actions.requestRedraw();
            return true;
        }
        case osgGA::GUIEventAdapter::KEY_Up:
        case 'w':
        {
            if (modifiers & osgGA::GUIEventAdapter::MODKEY_CTRL)
            {
                performMouseDeltaMovement(0, -0.1);
            }
            else if (modifiers & osgGA::GUIEventAdapter::MODKEY_ALT)
            {
                zoomModel(-0.1, true);
            }
            else
                panModel(0, -scaling * _distance, 0);

            actions.requestRedraw();
            return true;
        }
        case osgGA::GUIEventAdapter::KEY_Down:
        case 's':
        {
            if (modifiers & osgGA::GUIEventAdapter::MODKEY_CTRL)
            {
                performMouseDeltaMovement(0, 0.1);
            }
            else if (modifiers & osgGA::GUIEventAdapter::MODKEY_ALT)
            {
                zoomModel(0.1, true);
            }
            else
                panModel(0, scaling * _distance, 0);

            actions.requestRedraw();
            return true;
        }
        default:;
        }
        return osgGA::TrackballManipulator::handleKeyDown(ev, actions);
    }

    void setHomePosition(const osg::Vec3d& eye, const osg::Vec3d& center,
                         const osg::Vec3d& up,
                         bool autoComputeHomePosition) final
    {
        _maximumDistance =
            std::max((eye - center).length() * MAX_DISTANCE_FACTOR,
                     MIN_MAX_DISTANCE);
        osgGA::TrackballManipulator::setHomePosition(eye, center, up,
                                                     autoComputeHomePosition);
    }

private:
    float _maximumDistance;
};

TrackballManipulator::TrackballManipulator()
    : _manipulator(new Manipulator)
{
    _manipulator->setVerticalAxisFixed(false);
}

void TrackballManipulator::setHomePosition(const Vector3f& eye,
                                           const Vector3f& center,
                                           const Vector3f& up)
{
    using core::vec_to_vec;
    _manipulator->setHomePosition(vec_to_vec(eye), vec_to_vec(center),
                                  vec_to_vec(up), false);
}

void TrackballManipulator::getHomePosition(Vector3f& eye, Vector3f& center,
                                           Vector3f& up)
{
    osg::Vec3d e;
    osg::Vec3d c;
    osg::Vec3d u;
    _manipulator->getHomePosition(e, c, u);
    eye = Vector3f(e[0], e[1], e[2]);
    center = Vector3f(c[0], c[1], c[2]);
    up = Vector3f(u[0], u[1], u[2]);
}
}
}
