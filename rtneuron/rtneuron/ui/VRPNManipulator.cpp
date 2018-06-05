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

#include "VRPNManipulator.h"

#include "viewer/GyrationMouse.h"
#include "viewer/Intersense.h"
#include "viewer/SpaceMouse.h"
#include "viewer/VRPNManipulatorDevice.h"
#ifdef RTNEURON_USE_WIIUSE
#include "viewer/Wiimote.h"
#endif

#include <osgGA/CameraManipulator>

#include <boost/scoped_ptr.hpp>

#include <iostream>
#include <memory>

namespace bbp
{
namespace rtneuron
{
/*
  Nested classes
*/
class VRPNManipulator::Impl : public osgGA::CameraManipulator
{
public:
    Impl(const DeviceType type, const std::string& url)
        : _modelScale(0.01)
        , _minimumZoomScale(0.05)
        , _distance(0)
    {
        switch (type)
        {
        case GYRATION_MOUSE:
        case INTERSENSE_WAND:
            LBTHROW(std::runtime_error(
                "Cannot create VRPN manipulator using only an URL"));
            break;
        case SPACE_MOUSE:
            _device.reset(new core::SpaceMouse(url));
            break;
#ifdef RTNEURON_USE_WIIUSE
        case WIIMOTE:
            _device.reset(new core::Wiimote(url));
            break;
#endif
        default:
            std::cerr << "Unreachable code " << __FILE__ << ':' << __LINE__
                      << std::endl;
        }
    }

    Impl(const DeviceType type, const AttributeMap& configuration)
        : _modelScale(0.01)
        , _minimumZoomScale(0.05)
        , _distance(0)
    {
        switch (type)
        {
        case GYRATION_MOUSE:
            _device.reset(new core::GyrationMouse(configuration));
            break;
        case INTERSENSE_WAND:
            _device.reset(new core::Intersense(configuration));
            break;
        case SPACE_MOUSE:
        {
            _device.reset(
                new core::SpaceMouse(getDeviceURL(configuration, "analog")));
            break;
        }
#ifdef RTNEURON_USE_WIIUSE
        case WIIMOTE:
        {
            _device.reset(
                new core::Wiimote(getDeviceURL(configuration, "analog")));
            break;
        }
#endif
        default:
            std::cerr << "Unreachable code " << __FILE__ << ':' << __LINE__
                      << std::endl;
        }
    }

    virtual const char* className() const { return "VRPNManipulator"; }
    virtual void setByMatrix(const osg::Matrixd& matrix)
    {
        _center = osg::Vec3(0.0f, 0.0f, -_distance) * matrix;
        _attitude = matrix.getRotate();
    }

    virtual void setByInverseMatrix(const osg::Matrixd& matrix)
    {
        setByMatrix(osg::Matrixd::inverse(matrix));
    }

    virtual osg::Matrixd getMatrix() const
    {
        return osg::Matrixd::translate(0.0, 0.0, _distance) *
               osg::Matrixd::rotate(_attitude) *
               osg::Matrixd::translate(_center);
    }

    virtual osg::Matrixd getInverseMatrix() const
    {
        return osg::Matrixd::translate(-_center) *
               osg::Matrixd::rotate(_attitude.inverse()) *
               osg::Matrixd::translate(0.0, 0.0, -_distance);
    }

    virtual void init(const osgGA::GUIEventAdapter&, osgGA::GUIActionAdapter&)
    {
        const osg::Vec3f eye = osg::Vec3(0.0, 0.0, 5.0);
        const osg::Vec3f center = osg::Vec3(0.0, 0.0, 0.0);
        const osg::Vec3f up = osg::Vec3(0.0, 1.0, 0.0);

        setHomePosition(eye, center, up);
        home(0);

        _modelScale = 0.01f;
        _minimumZoomScale = 0.05f;
    }

    virtual bool handle(const osgGA::GUIEventAdapter& ev,
                        osgGA::GUIActionAdapter& actionAdapter)
    {
        switch (ev.getEventType())
        {
        case (osgGA::GUIEventAdapter::FRAME):
            if (_calcMovement())
                actionAdapter.requestRedraw();
            return false;
        default:
            break;
        }

        return false;
    }

    virtual void home(double)
    {
        if (getAutoComputeHomePosition())
            computeHomePosition();
        _computePosition(_homeEye, _homeCenter, _homeUp);
    }

private:
    double _modelScale;
    double _minimumZoomScale;

    osg::Vec3d _center;  //!< Position of the manipulator
    double _distance;    //!< Distance of the manipulator
    osg::Quat _attitude; //!< Attitude of the manipulator

    boost::scoped_ptr<core::VRPNManipulatorDevice> _device;

    /*
      This function queries the manipulators and  calculates the resulting
      movement of the manipulator.

      Returns true if the calculation was successful.
    */
    bool _calcMovement()
    {
        if (!_device)
            return false;

        osg::Vec3d dT;
        osg::Vec3d dR;
        bool reset;
        _device->sample(dT, dR, _attitude, reset);

        if (reset)
            home(0);

        osg::Matrix rotation_matrix;
        rotation_matrix.makeRotate(_attitude);

        const osg::Vec3d upVector = getFrontVector(rotation_matrix);
        const osg::Vec3d sideVector = -getSideVector(rotation_matrix);
        osg::Vec3d lookVector = -getUpVector(rotation_matrix);

        /* Attitude */
        _attitude *= osg::Quat(dR.y(), upVector);
        _attitude *= osg::Quat(dR.x(), sideVector);
        _attitude *= osg::Quat(dR.z(), lookVector);

        /* To prevent the camera from rotating around X too much, so it doesn't
           leave Y axis upside down...
         --> If the inclination angle is greater than 90 degrees, undo last
           rotation */
        rotation_matrix.makeRotate(_attitude);
        if (fabs(atan2(-rotation_matrix(1, 2), rotation_matrix(1, 1))) >
            osg::DegreesToRadians(90.0f))
            _attitude /= osg::Quat(dR.x(), sideVector);

        /* Pan */
        float scale = -0.3f * _distance;
        rotation_matrix.makeRotate(_attitude);

        const osg::Vec3 dv(dT.x() * scale, -dT.y() * scale, 0.0f);
        _center += dv * rotation_matrix;

        /* Zoom */
        scale = 1.0f - dT.z();
        if (_distance * scale > _modelScale * _minimumZoomScale)
            _distance *= scale;

        return true;
    }

    void _computePosition(const osg::Vec3& eye, const osg::Vec3& center,
                          const osg::Vec3& up)
    {
        _center = center;

        osg::Vec3 f(center - eye);
        _distance = f.normalize();
        osg::Vec3 s(f ^ up);
        s.normalize();
        osg::Vec3 u(s ^ f);
        u.normalize();

        const osg::Matrix rotation_matrix(s[0], u[0], -f[0], 0.0f, s[1], u[1],
                                          -f[1], 0.0f, s[2], u[2], -f[2], 0.0f,
                                          0.0f, 0.0f, 0.0f, 1.0f);
        _attitude = rotation_matrix.getRotate().inverse();
    }
};

/*
  Constructors/destructor
*/

VRPNManipulator::VRPNManipulator(const DeviceType type,
                                 const std::string& hostName)
    : _impl(new Impl(type, hostName))
{
    _impl->ref();
}

VRPNManipulator::VRPNManipulator(const DeviceType type,
                                 const AttributeMap& configuration)
    : _impl(new Impl(type, configuration))
{
    _impl->ref();
}

VRPNManipulator::~VRPNManipulator()
{
    _impl->unref();
}

/*
  Member functions
*/

osgGA::CameraManipulator* VRPNManipulator::osgManipulator()
{
    return _impl;
}

std::string VRPNManipulator::getDeviceURL(const AttributeMap& attributes,
                                          const std::string& name)
{
    const std::string defaultURL = attributes("url", "");
    const AttributeMapPtr fieldPtr = attributes(name, AttributeMapPtr());
    if (!fieldPtr)
        return attributes(name, defaultURL);
    else
    {
        const AttributeMap& field = *fieldPtr;
        return field("url", defaultURL);
    }
}
}
}
