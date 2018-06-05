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

#ifndef RTNEURON_API_CAMERA_H
#define RTNEURON_API_CAMERA_H

#include "View.h"

#include <boost/signals2/signal.hpp>

#include <memory>

namespace bbp
{
namespace rtneuron
{
// clang-format off

/**
   A camera represents the frustum and the model part of the model-view
   transformation used for a single view.

   The view part of the transformation will be handled internally by the
   Equalizer backend as this is required in cluster configurations.
*/
class Camera
{
    friend class View::_Impl;
    friend class Testing;

public:
    /*--- Public declarations ---*/

    typedef void DirtySignalSignature();
    typedef boost::signals2::signal<DirtySignalSignature> DirtySignal;

    /*--- Public member constructors/destructor ---*/

    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    /*--- Public member functions (available in python) ---*/

    /** @name C++ public interface wrapped in Python */
    ///@{

    /**
       Change the vertical field of view of the perspective projection.

       The aspect ratio is inferred from the current projection matrix.

       @param verticalFOV Angle in degrees.
    */
    void setProjectionPerspective(float verticalFOV);

    /**
       Gets the parameters of a perspective projection.

       The results are undefined if the camera is set as orthographic.

       \if pybind
       @return A tuple with the vertical field of view and the aspect ratio.
       \else
       @param verticalFOV Vertical field of view.
       @param aspectRatio The ratio between the horizontal and vertical fields
                          of view.
       \endif
    */
#ifdef DOXYGEN_TO_BREATHE
    tuple getProjectionPerspective() const;
#else
    void getProjectionPerspective(float& verticalFOV, float& aspectRatio) const;
#endif

    /**
       Sets the camera frustum for perspective projection.

       Near and far are autoadjusted by the renderer. The near value provided
       here is used to infer the field of view. No auto aspect ratio
       conservation is performed.
    */
    void setProjectionFrustum(float left, float right, float bottom, float top,
                              float near);

    /**
       Gets the frustum definition of a perspective projection.

       The near parameters returned is just meant to indicate the field of
       view the actual parameters used for rendering are adjusted to the
       scene being displayed.

       The results are undefined if the camera is set as orthographic.

       \if pybind
       @return A tuple with left, right, top, bottom, near
       \endif
    */
#ifdef DOXYGEN_TO_BREATHE
    tuple getProjectionFrustum() const;
#else
    void getProjectionFrustum(float& left, float& right, float& bottom,
                              float& top, float& near) const;
#endif

    /**
       Sets the camera frustum for orthographic projections.
    */
    void setProjectionOrtho(float left, float right, float bottom, float top,
                            float near);

    /**
       Gets the camera frustum for orthographic projection.

       The results are undefined if the camera is using perspective projection.

       \if pybind
       @return left, right, bottom and top
       \endif
    */
#ifdef DOXYGEN_TO_BREATHE
    tuple getProjectionOrtho() const;
#else
    void getProjectionOrtho(float& left, float& right, float& bottom,
                            float& top) const;
#endif

    /**
       Get the camera projection matrix.

       \if pybind
       @return An OpenGL ready 4x4 numpy matrix
       \endif
    */
#ifdef DOXYGEN_TO_BREATHE
    typle getProjectionMatrix() const;
#else
    Matrix4f getProjectionMatrix() const;
#endif

    /**
       Sets the camera to do orthographic projection preserving the
       current frustum.
    */
    void makeOrtho();

    /**
       Sets the camera to do perspective projection preserving the
       current frustum.
    */
    void makePerspective();

    /**
       @return True if the camera is applying orthographic projection, false
               otherwise
    */
    bool isOrtho() const;

    /**
       The same as gluLookAt.

       This method also sets the home position and pivotal point for
       manipulators that take it into account.
     */
    void setViewLookAt(const Vector3f& eye, const Vector3f& center,
                       const Vector3f& up);

    /**
       Sets the camera position

       @param position The world position of the camera
       \if pybind
       @param orientation A tuple ((x, y, z), angle) with a rotation to be
              applied to the camera (the initial view direction is looking
              down the negative z axis). The angle is in degrees.
       \else
       @param orientation A rotation to be applied to the camera (the initial
              view direction is looking down the negative z axis).
       \endif
     */
    void setView(const Vector3f& position, const Orientation& orientation);

    /**
       Get the camera position.

       \if pybind
       @return A tuple (position, (axis, angle)) where position and axis are
       [x, y, z] lists and the angle is in degrees.
       \else
       @param position The camera position in world coordinates
       @param orientation The rotation of the camera with regard to the
       default orientation (y up, x right, negative z front).
       \endif
     */
#ifdef DOXYGEN_TO_BREATHE
    object getView() const;
#else
    void getView(Vector3f& position, Orientation& orientation) const;
#endif

    /**
       Get the camera modelview matrix.

       \if pybind
       @return An OpenGL ready 4x4 numpy matrix
       \endif
    */
#ifdef DOXYGEN_TO_BREATHE
    object getViewMatrix() const;
#else
    Matrix4f getViewMatrix() const;
#endif

    /**
       Returns the 2D projected coordinates of a 3D point in world coordinates.

       The third coordinate just represents if the point is in front of (+1),
       coincident (0) or behind (-1) the camera. If any coordinate is
       <-1 or >1 that means that the point is outside the frustum.
     */
    Vector3f projectPoint(const Vector3f& position) const;

    /**
       Return the 3D world coordinates of a projected point.

       @param point The 2D normalized device coordinates of the point
       @param z The z value of the 2D point in camera coordinates. Note that
                for points in front of the camera this value is negative.
     */
    Vector3f unprojectPoint(const vmml::Vector2f& point, float z) const;

    /*--- Public signals ---*/

    /**
       \brief Emitted whenever the modelview matrix is modified by the
       rendering engine.
    */
    DirtySignal viewDirty;

    ///@}

    /*--- Public member functions (C++ only) ---*/

    /** @name C++ only public interface */
    ///@{

    /**
       The same as setViewLookAt but the dirty signal is not emitted.
    */
    void setViewLookAtNoDirty(const Vector3f& eye, const Vector3f& center,
                              const Vector3f& up);

    /**
       The same as setView but the dirty signal is not emitted.
    */
    void setViewNoDirty(const Vector3f& position,
                        const Orientation& orientation);

    /**
       @internal
       Called when the backend Equalizer object is going to be released.

       If the wrapping is holding this object after this occurs, any
       operation will throw an exception to handle on the python side.

       This function deallocates any resources held by this object.
    */
    void invalidate();

    /**
       @internal
       Checks if invalidate was called and throws a Bad_Operation
       exception in that case.
     */
    void checkValid() const;

    ///@}

protected:
    /*--- Protected members attributes ---*/

    class _Impl;
    std::unique_ptr<_Impl> _impl;

private:
    /*--- Private constructors/destructor ---*/

    /**
       This prototype may change to allow sharing cameras between views.
     */
    Camera(osgEq::View* view);
};
}
}
#endif
