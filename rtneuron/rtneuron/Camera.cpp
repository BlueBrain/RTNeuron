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

#include <vmmlib/matrix.hpp>

#include <lunchbox/debug.h>

namespace bbp
{
namespace rtneuron
{
/*
  Constructors
*/
Camera::Camera(osgEq::View* view)
    : _impl(new _Impl(view, this))
{
}

/*
  Destructor
*/
Camera::~Camera()
{
}

/*
  Member functions
*/
void Camera::setProjectionPerspective(const float verticalFOV)
{
    checkValid();
    _impl->setProjectionPerspective(verticalFOV);
}

void Camera::makeOrtho()
{
    checkValid();
    _impl->makeOrtho();
}

void Camera::makePerspective()
{
    checkValid();
    _impl->makePerspective();
}

bool Camera::isOrtho() const
{
    checkValid();
    return _impl->isOrtho();
}

void Camera::setProjectionFrustum(const float left, const float right,
                                  const float bottom, const float top,
                                  const float near)
{
    checkValid();
    _impl->setProjectionFrustum(left, right, bottom, top, near);
}

void Camera::getProjectionPerspective(float& verticalFOV,
                                      float& aspectRatio) const
{
    checkValid();
    _impl->getProjectionPerspective(verticalFOV, aspectRatio);
}

void Camera::getProjectionFrustum(float& left, float& right, float& bottom,
                                  float& top, float& near) const
{
    checkValid();
    _impl->getProjectionFrustum(left, right, bottom, top, near);
}

void Camera::setProjectionOrtho(const float left, const float right,
                                const float bottom, const float top,
                                const float near)
{
    checkValid();
    _impl->setProjectionOrtho(left, right, bottom, top, near);
}

void Camera::getProjectionOrtho(float& left, float& right, float& top,
                                float& bottom) const
{
    checkValid();
    _impl->getProjectionOrtho(left, right, top, bottom);
}

Matrix4f Camera::getProjectionMatrix() const
{
    checkValid();
    return _impl->getProjectionMatrix();
}

void Camera::setViewLookAt(const Vector3f& eye, const Vector3f& center,
                           const Vector3f& up)
{
    checkValid();
    _impl->setViewLookAt(eye, center, up);
}

void Camera::setView(const Vector3f& position, const Orientation& orientation)
{
    checkValid();
    _impl->setView(position, orientation);
}

void Camera::getView(Vector3f& position, Orientation& orientation) const
{
    checkValid();
    _impl->getView(position, orientation);
}

Matrix4f Camera::getViewMatrix() const
{
    checkValid();
    return _impl->getViewMatrix();
}

Vector3f Camera::projectPoint(const Vector3f& point) const
{
    checkValid();
    return _impl->projectPoint(point);
}

Vector3f Camera::unprojectPoint(const vmml::Vector2f& point,
                                const float z) const
{
    checkValid();
    return _impl->unprojectPoint(point, z);
}

void Camera::setViewLookAtNoDirty(const Vector3f& eye, const Vector3f& center,
                                  const Vector3f& up)
{
    checkValid();
    _impl->setViewLookAt(eye, center, up, false);
}

void Camera::setViewNoDirty(const Vector3f& position,
                            const Orientation& orientation)
{
    checkValid();
    _impl->setView(position, orientation, false);
}

void Camera::invalidate()
{
    _impl.reset();
}

void Camera::checkValid() const
{
    if (_impl.get() == 0)
    {
        LBTHROW(std::runtime_error(
            "This View associanted with this camera has been destroyed."));
    }
}
}
}
