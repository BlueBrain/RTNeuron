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

#include "CameraImpl.h"

#include "util/math.h"
#include "util/vec_to_vec.h"
#include "viewer/osgEq/View.h"

#include <osg/io_utils>

namespace bbp
{
namespace rtneuron
{
using core::vec_to_vec;

// Helper functions -----------------------------------------------------------

namespace
{
int _uniqueNonZero(const osg::Vec3& v)
{
    if (v[0] != 0 && v[1] == 0 && v[2] == 0)
        return 0;
    if (v[0] == 0 && v[1] != 0 && v[2] == 0)
        return 1;
    if (v[0] == 0 && v[1] == 0 && v[2] != 0)
        return 2;
    return -1;
}

float _computeAspectRatio(const float angle_a, const float angle_b)
{
    return std::tan(angle_a / 180 * M_PI / 2) /
           std::tan(angle_b / 180 * M_PI / 2);
}

void _getWallSides(const eq::Wall& wall, float& left, float& right,
                   float& bottom, float& top)
{
    Vector3f u = wall.bottomRight - wall.bottomLeft;
    Vector3f v = wall.topLeft - wall.bottomLeft;
    const float halfWidth = u.normalize() * 0.5;
    const float halfHeight = v.normalize() * 0.5;
    const Vector3f center = (wall.bottomRight + wall.topLeft) * 0.5f;
    left = center.x() - halfWidth;
    right = center.x() + halfWidth;
    top = center.y() + halfHeight;
    bottom = center.y() - halfHeight;
}
}

// Camera::_Impl --------------------------------------------------------------

/*
  Constructor
*/

Camera::_Impl::_Impl(osgEq::View* view, Camera* parent)
    : _view(view)
    , _parent(parent)
{
    _view->modelviewDirty.connect(boost::bind(&_Impl::_onModelViewDirty, this));
}

/*
  Destructor
*/

Camera::_Impl::~_Impl()
{
    /* osgEq::View survives the lifetime of this object */
    _view->modelviewDirty.disconnect(
        boost::bind(&_Impl::_onModelViewDirty, this));
}

/*
  Member functions
*/
void Camera::_Impl::setViewLookAt(const Vector3f& eye, const Vector3f& center,
                                  const Vector3f& up, const bool emitDirty)
{
    _view->setHomePosition(vec_to_vec(eye), vec_to_vec(center), vec_to_vec(up));
    osg::Matrix v;
    if (eye == center)
    {
        std::cerr << "Warning: the eye and center points are equal"
                  << std::endl;
    }
    if (_uniqueNonZero(vec_to_vec(eye - center)) ==
        _uniqueNonZero(vec_to_vec(up)))
    {
        std::cerr << "Warning: the up vector is aligned to the view direction"
                  << std::endl;
    }
    v.makeLookAt(vec_to_vec(eye), vec_to_vec(center), vec_to_vec(up));
    _view->setModelMatrix(v, emitDirty);
}

void Camera::_Impl::setView(const Vector3f& position,
                            const Orientation& orientation,
                            const bool emitDirty)
{
    osg::Matrix v =
        (osg::Matrix::rotate(orientation.w() / 180 * M_PI,
                             vec_to_vec(orientation.get_sub_vector<3, 0>())) *
         osg::Matrix::translate(vec_to_vec(position)));
    v.invert(v);
    _view->setModelMatrix(v, emitDirty);
}

void Camera::_Impl::getView(Vector3f& position, Orientation& orientation) const
{
    core::decomposeMatrix(_view->getModelMatrix(), position, orientation);
}

Matrix4f Camera::_Impl::getViewMatrix() const
{
    auto data = _view->getModelMatrix().ptr();
    Matrix4f result;
    for (size_t i = 0; i < 16; ++i)
        result(i & 3, i >> 2) = data[i];
    return result;
}

void Camera::_Impl::makeOrtho()
{
    _view->setUseOrtho(true);
}

void Camera::_Impl::makePerspective()
{
    _view->setUseOrtho(false);
}

bool Camera::_Impl::isOrtho() const
{
    return _view->isOrtho();
}

void Camera::_Impl::setProjectionPerspective(const float verticalFOV)
{
    switch (_view->getCurrentType())
    {
    case eq::fabric::Frustum::TYPE_WALL:
    {
        eq::Projection projection;
        projection = _view->getWall();
        eq::Wall wall = _view->getWall();
        const float alpha = _computeAspectRatio(verticalFOV, projection.fov[1]);
        wall.resizeVertical(alpha);
        wall.resizeHorizontal(alpha);
        _view->setPerspectiveWall(wall);
        break;
    }
    case eq::fabric::Frustum::TYPE_PROJECTION:
    {
        eq::Projection projection = _view->getProjection();
        std::cout << projection << std::endl;
        const float alpha = _computeAspectRatio(verticalFOV, projection.fov[1]);
        projection.resizeVertical(alpha);
        projection.resizeHorizontal(alpha);
        _view->setProjection(projection);
        break;
    }
    default: // eq::fabric::frustum::TYPE_NONE
        /* Can't update the frustum because there's no frustum */
        throw std::runtime_error(
            "Can't change vertical FOV in a camera "
            " with no frustum intialized");
    }
}

void Camera::_Impl::setProjectionFrustum(float left, float right, float bottom,
                                         float top, float near)
{
    eq::Wall wall;
    wall.bottomLeft = eq::Vector3f(left, bottom, -near);
    wall.bottomRight = eq::Vector3f(right, bottom, -near);
    wall.topLeft = eq::Vector3f(left, top, -near);
    wall.type = eq::Wall::TYPE_FIXED;
    _view->setPerspectiveWall(wall);
}

void Camera::_Impl::setProjectionOrtho(const float left, const float right,
                                       const float bottom, const float top,
                                       const float near)
{
    eq::Wall wall;
    wall.bottomLeft = eq::Vector3f(left, bottom, -near);
    wall.bottomRight = eq::Vector3f(right, bottom, -near);
    wall.topLeft = eq::Vector3f(left, top, -near);
    wall.type = eq::Wall::TYPE_FIXED;
    _view->setOrthoWall(wall);
}

void Camera::_Impl::getProjectionOrtho(float& left, float& right, float& bottom,
                                       float& top) const
{
    if (_view->getCurrentType() != eq::fabric::Frustum::TYPE_WALL)
        throw std::runtime_error(
            "Cannot extract an orthographic frustum from "
            "a camera using perspective projection");
    const eq::Wall& wall = _view->getWall();
    _getWallSides(wall, left, right, bottom, top);
}

void Camera::_Impl::getProjectionFrustum(float& left, float& right,
                                         float& bottom, float& top,
                                         float& near) const
{
    if (_view->getCurrentType() == eq::fabric::Frustum::TYPE_WALL)
    {
        const eq::Wall& wall = _view->getWall();
        _getWallSides(wall, left, right, bottom, top);
        near = -wall.bottomRight.z();
    }
    else
    {
        const eq::Projection& projection = _view->getProjection();
        right = std::tan(projection.fov[0] / 180 * M_PI / 2);
        left = -right;
        top = std::tan(projection.fov[1] / 180 * M_PI / 2);
        bottom = -right;
        near = -projection.distance;
    }
}

void Camera::_Impl::getProjectionPerspective(float& verticalFOV,
                                             float& aspectRatio) const
{
    eq::Projection projection;
    switch (_view->getCurrentType())
    {
    case eq::fabric::Frustum::TYPE_WALL:
        projection = _view->getWall();
        break;
    case eq::fabric::Frustum::TYPE_PROJECTION:
        projection = _view->getProjection();
        break;
    default: // eq::fabric::frustum::TYPE_NONE
        /* Can't retrieve the frustum because there's no frustum */
        throw std::runtime_error(
            "Cannot get vertical FOV in a camera "
            " with no frustum intialized");
    }

    verticalFOV = projection.fov[1];
    aspectRatio = _computeAspectRatio(projection.fov[0], projection.fov[1]);
}

Matrix4f Camera::_Impl::getProjectionMatrix() const
{
    return _getProjectionMatrix(0.01, 100000);
}

Vector3f Camera::_Impl::projectPoint(const Vector3f& point) const
{
    const osg::Vec3 p = vec_to_vec(point) * _view->getModelMatrix();

    Vector3f projected;
    projected[2] = (p[2] < 0) - (p[2] > 0);

    float l, r, b, t;
    float factor;
    if (_view->isOrtho())
    {
        getProjectionOrtho(l, r, b, t);
        factor = 1 / _view->getModelUnit();
    }
    else
    {
        if (projected[2] == 0)
            return projected;

        float n;
        getProjectionFrustum(l, r, b, t, n);
        factor = -n / p[2];
    }
    projected[0] = (p[0] * factor - l) / (r - l) * 2 - 1;
    projected[1] = (p[1] * factor - b) / (t - b) * 2 - 1;

    return projected;
}

Vector3f Camera::_Impl::unprojectPoint(const vmml::Vector2f& point,
                                       const float z) const
{
    const auto& modelMatrix = _view->getModelMatrix();

    if (z == 0)
    {
        return -Vector3f(modelMatrix(3, 0), modelMatrix(3, 1),
                         modelMatrix(3, 2));
    }

    /* We use the camera z given as the near plane, so we can easily compute
       the unprojected point later. */
    const auto projection =
        osg::Matrix(_getProjectionMatrix(-z, -z * 2).data());
    const auto MVP = _view->getModelMatrix() * projection;
    osg::Matrix inverseMVP;
    inverseMVP.invert(MVP);
    const auto worldPoint = osg::Vec3(point.x(), point.y(), -1) * inverseMVP;
    return vec_to_vec(worldPoint);
}

Matrix4f Camera::_Impl::_getProjectionMatrix(const float near,
                                             const float far) const
{
    typedef vmml::Frustumf Frustum;
    if (_view->isOrtho())
    {
        Frustum frustum;
        getProjectionOrtho(frustum.left(), frustum.right(), frustum.bottom(),
                           frustum.top());
        /* This values are arbitrary */
        frustum.nearPlane() = near;
        frustum.farPlane() = far;
        return frustum.computeOrthoMatrix();
    }

    Frustum frustum;
    getProjectionFrustum(frustum.left(), frustum.right(), frustum.bottom(),
                         frustum.top(), frustum.nearPlane());
    const float correction = near / frustum.nearPlane();
    frustum.left() *= correction;
    frustum.right() *= correction;
    frustum.top() *= correction;
    frustum.bottom() *= correction;
    frustum.nearPlane() *= correction;
    frustum.farPlane() = far;
    return frustum.computePerspectiveMatrix();
}

void Camera::_Impl::_onModelViewDirty() const
{
    _parent->viewDirty();
}
}
}
