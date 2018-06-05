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

#ifndef RTNEURON_API_CAMERA_IMPL_H
#define RTNEURON_API_CAMERA_IMPL_H

#include "Camera.h"

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class View;
}

/**
   At this moment this object is nothing but a proxy to a osgEq::View for
   the frustum and its model matrix.
*/
class Camera::_Impl
{
public:
    /*--- Public constructors/destructor ---*/
    _Impl(osgEq::View* view, Camera* parent);

    ~_Impl();

    /*--- Public member functions ---*/

    /**
       @sa Camera::setViewLookAt
     */
    void setViewLookAt(const Vector3f& eye, const Vector3f& center,
                       const Vector3f& up, const bool emitDirty = true);
    /**
       @sa Camera::setView
     */
    void setView(const Vector3f& position, const Orientation& orientation,
                 const bool emitDirty = true);
    /**
       @sa Camera::getView
     */
    void getView(Vector3f& position, Orientation& orientation) const;

    /**
       @sa Camera::getViewMatrix
     */
    Matrix4f getViewMatrix() const;

    /**
       @sa Camera::makeOrtho
    */
    void makeOrtho();

    /**
       @sa Camera::makePerspective
    */
    void makePerspective();

    /**
       @sa Camera::isOrtho
    */
    bool isOrtho() const;

    /**
       @sa Camera::setProjectionPerspective
       @todo Add aspect ratio
    */
    void setProjectionPerspective(const float verticalFOV);

    /**
       @sa Camera::setProjectionFrustum
     */
    void setProjectionFrustum(const float left, const float right,
                              const float bottom, const float top,
                              const float near);
    /**
       @sa Camera::setProjectionOrtho
     */
    void setProjectionOrtho(const float left, const float right,
                            const float bottom, const float top,
                            const float near);

    /**
       @sa Camera::getProjectionPerspective
     */
    void getProjectionPerspective(float& verticalFOV, float& aspectRatio) const;

    /**
       @sa Camera::getProjectionFrustum
     */
    void getProjectionFrustum(float& left, float& right, float& bottom,
                              float& top, float& near) const;

    /**
       @sa Camera::getProjectionOrtho
     */
    void getProjectionOrtho(float& left, float& right, float& bottom,
                            float& top) const;

    /**
       @sa Camera::getProjectionMatrix
     */
    Matrix4f getProjectionMatrix() const;

    /**
       @sa Camera::projectPoint
    */
    Vector3f projectPoint(const Vector3f& point) const;

    /**
       @sa Camera::unprojectPoint
    */
    Vector3f unprojectPoint(const vmml::Vector2f& point, float z) const;

private:
    /*--- Private member attributes ---*/

    /* For the moment a single view per camera is allowed, being the View
       the owner of the camera. */
    osgEq::View* _view;

    Camera* _parent;

    /*--- Private member functions ---*/

    Matrix4f _getProjectionMatrix(const float near, const float far) const;
    void _onModelViewDirty() const;
};
}
}
#endif
