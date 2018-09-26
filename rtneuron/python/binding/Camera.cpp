/* Copyright (c) 2006-2018, Ecole Polytechnique Federale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politecnica de Madrid (UPM)
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

#include "boost_signal_connect_wrapper.h"

#include "rtneuron/Camera.h"

#include "docstrings.h"
#include "helpers.h"

#include <brain/python/arrayHelpers.h>

#include <vmmlib/matrix.hpp>
#include <vmmlib/vector.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

tuple Camera_getProjectionPerspective(const Camera* camera)
{
    float verticalFOV, aspectRatio;
    camera->getProjectionPerspective(verticalFOV, aspectRatio);
    return make_tuple(verticalFOV, aspectRatio);
}

tuple Camera_getProjectionOrtho(const Camera* camera)
{
    float left, right, top, bottom;
    camera->getProjectionOrtho(left, right, bottom, top);
    return make_tuple(left, right, bottom, top);
}

tuple Camera_getProjectionFrustum(const Camera* camera)
{
    float left, right, top, bottom, near;
    camera->getProjectionFrustum(left, right, bottom, top, near);
    return make_tuple(left, right, bottom, top, near);
}

object Camera_getProjectionMatrix(const Camera* camera)
{
    return brain_python::toNumpy(camera->getProjectionMatrix());
}

void Camera_setViewLookAt(Camera* camera, object& eye, object& center,
                          object& up)
{
    camera->setViewLookAt(extract_Vector3(eye), extract_Vector3(center),
                          extract_Vector3(up));
}

void Camera_setView(Camera* camera, object& position, object& orientation)
{
    camera->setView(extract_Vector3(position),
                    extract_Orientation(orientation));
}

object Camera_getView(Camera* camera)
{
    Vector3f position;
    Orientation orientation;
    camera->getView(position, orientation);

    list position_;
    position_.append(position.x());
    position_.append(position.y());
    position_.append(position.z());

    list axis;
    axis.append(orientation.x());
    axis.append(orientation.y());
    axis.append(orientation.z());

    return make_tuple(position_, make_tuple(axis, orientation.w()));
}

object Camera_getViewMatrix(const Camera* camera)
{
    return brain_python::toNumpy(camera->getViewMatrix());
}

object Camera_projectPoint(const Camera* camera, object& point)
{
    const Vector3f projected = camera->projectPoint(extract_Vector3(point));
    return make_tuple(projected.x(), projected.y(), projected.z());
}

object Camera_unprojectPoint(const Camera* camera, object& point, const float z)
{
    const Vector3f unprojected =
        camera->unprojectPoint(extract_Vector2(point), z);
    return make_tuple(unprojected.x(), unprojected.y(), unprojected.z());
}

// export_Camera ---------------------------------------------------------------

void export_Camera()
// clang-format off
{

class_<Camera, CameraPtr, boost::noncopyable> cameraWrapper(
    "Camera", DOXY_CLASS(bbp::rtneuron::Camera), no_init);

scope cameraScope = cameraWrapper;

/* Nested classes */
class_<Camera::DirtySignal, boost::noncopyable>("__DirtySignal__", no_init)
    .def("connect", signal_connector<Camera::DirtySignalSignature>::connect)
    .def("disconnect",
         signal_connector<Camera::DirtySignalSignature>::disconnect);

cameraWrapper
    .def("setProjectionPerspective", &Camera::setProjectionPerspective,
         (arg("verticalFOV")),
         DOXY_FN(bbp::rtneuron::Camera::setProjectionPerspective))
    .def("getProjectionPerspective", Camera_getProjectionPerspective,
         DOXY_FN(bbp::rtneuron::Camera::getProjectionPerspective))
    .def("getProjectionFrustum", Camera_getProjectionFrustum,
         DOXY_FN(bbp::rtneuron::Camera::getProjectionFrustum))
    .def("setProjectionFrustum",
         (void (Camera::*)(float, float, float, float, float)) &
             Camera::setProjectionFrustum,
         (arg("left"), arg("right"), arg("bottom"), arg("top"),
          arg("near") = 0.1),
         DOXY_FN(bbp::rtneuron::Camera::setProjectionFrustum))
    .def("setProjectionOrtho",
         (void (Camera::*)(float, float, float, float, float)) &
             Camera::setProjectionOrtho,
         (arg("left"), arg("right"), arg("bottom"), arg("top"),
          arg("near") = 0.1),
         DOXY_FN(bbp::rtneuron::Camera::setProjectionOrtho))
    .def("getProjectionOrtho", Camera_getProjectionOrtho,
         DOXY_FN(bbp::rtneuron::Camera::getProjectionOrtho))
    .def("getProjectionMatrix", Camera_getProjectionMatrix,
         DOXY_FN(bbp::rtneuron::Camera::getProjectionMatrix))
    .def("makeOrtho", &Camera::makeOrtho,
         DOXY_FN(bbp::rtneuron::Camera::makeOrtho))
    .def("makePerspective", &Camera::makePerspective,
         DOXY_FN(bbp::rtneuron::Camera::makePerspective))
    .def("isOrtho", &Camera::isOrtho,
         DOXY_FN(bbp::rtneuron::Camera::isOrtho))
    .def("setViewLookAt", Camera_setViewLookAt,
         (arg("eye"), arg("center"), arg("up")),
         DOXY_FN(bbp::rtneuron::Camera::setViewLookAt))
    .def("setView", Camera_setView, (arg("position"), arg("orientation")),
         DOXY_FN(bbp::rtneuron::Camera::setView))
    .def("getView", Camera_getView, DOXY_FN(bbp::rtneuron::Camera::getView))
    .def("getViewMatrix", Camera_getViewMatrix,
         DOXY_FN(bbp::rtneuron::Camera::getViewMatrix))
    .def("projectPoint", Camera_projectPoint,
         DOXY_FN(bbp::rtneuron::Camera::projectPoint))
    .def("unprojectPoint", Camera_unprojectPoint,
         DOXY_FN(bbp::rtneuron::Camera::unprojectPoint))
    .def_readonly("viewDirty", &Camera::viewDirty,
                  DOXY_VAR(bbp::rtneuron::Camera::viewDirty));
}
