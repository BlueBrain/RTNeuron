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

#include "docstrings.h"
#include "helpers.h"

#include "rtneuron/AttributeMap.h"
#include "rtneuron/View.h"
#include "rtneuron/types.h"
#include "rtneuron/ui/CameraPath.h"
#include "rtneuron/ui/CameraPathManipulator.h"
#include "rtneuron/ui/TrackballManipulator.h"
#ifdef RTNEURON_USE_VRPN
#include "rtneuron/ui/VRPNManipulator.h"
#endif

#include <boost/python.hpp>

#include <vmmlib/vector.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

// export_CameraManipulators ---------------------------------------------------

void TrackballManipulator_setHomePosition(TrackballManipulator* manipulator,
                                          object& eye, object& center,
                                          object& up)
{
    manipulator->setHomePosition(extract_Vector3(eye), extract_Vector3(center),
                                 extract_Vector3(up));
}

tuple TrackballManipulator_getHomePosition(TrackballManipulator* manipulator)
{
    Vector3f eye, center, up;
    manipulator->getHomePosition(eye, center, up);
    return make_tuple(make_tuple(eye.x(), eye.y(), eye.z()),
                      make_tuple(center.x(), center.y(), center.z()),
                      make_tuple(up.x(), up.y(), up.z()));
}

CameraPath::KeyFramePtr KeyFrame_init(object& position, object& orientation,
                                      double stereoCorrection = 1)
{
    return CameraPath::KeyFramePtr(
        new CameraPath::KeyFrame(extract_Vector3(position),
                                 extract_Orientation(orientation),
                                 float(stereoCorrection)));
}

void KeyFrame_setPosition(CameraPath::KeyFrame* keyframe, object p)
{
    keyframe->setPosition(extract_Vector3(p));
}

tuple KeyFrame_getPosition(CameraPath::KeyFrame* keyframe)
{
    Vector3f position = keyframe->getPosition();
    return make_tuple(position.x(), position.y(), position.z());
}

void KeyFrame_setOrientation(CameraPath::KeyFrame* keyframe, object o)
{
    keyframe->setOrientation(extract_Orientation(o));
}

tuple KeyFrame_getOrientation(CameraPath::KeyFrame* keyframe)
{
    Orientation orientation = keyframe->getOrientation();
    return make_tuple(make_tuple(orientation.x(), orientation.y(),
                                 orientation.z()),
                      orientation.w());
}

str KeyFrame_repr(const CameraPath::KeyFrame* keyframe)
{
    std::stringstream string;
    Vector3f p = keyframe->getPosition();
    Orientation o = keyframe->getOrientation();
    string << "CameraPath.KeyFrame([" << p[0] << ", " << p[1] << ", " << p[2]
           << "], "
           << "([" << o[0] << ", " << o[1] << ", " << o[2] << "], " << o[3]
           << "), " << keyframe->getStereoCorrection() << ")";
    return str(string.str());
}

void CameraPath_setKeyFrames(CameraPath* path, dict d)
{
    std::map<double, CameraPath::KeyFramePtr> keyframes;
    list frames = d.items();
    for (int i = 0; i != len(frames); ++i)
    {
        tuple timeKeyframe = extract<tuple>(frames[i]);
        double time = extract<double>(timeKeyframe[0]);
        CameraPath::KeyFramePtr keyframe =
            extract<CameraPath::KeyFramePtr>(timeKeyframe[1]);
        keyframes[time] = keyframe;
    }
    path->setKeyFrames(keyframes);
}

list CameraPath_getKeyFrames(CameraPath* path)
{
    typedef std::vector<std::pair<double, CameraPath::KeyFramePtr>> KeyFrames;
    KeyFrames keyframes = path->getKeyFrames();
    list out;
    for (KeyFrames::iterator k = keyframes.begin(); k != keyframes.end(); ++k)
        out.append(make_tuple(object(k->first), object(k->second)));
    return out;
}

tuple CameraPathManipulator_getKeyFrame(
    const CameraPathManipulator* manipulator, const double timestamp)
{
    Vector3f position;
    Orientation orientation;
    double stereoCorrection;
    manipulator->getKeyFrame(timestamp, position, orientation,
                             stereoCorrection);
    return make_tuple(make_tuple(position.x(), position.y(), position.z()),
                      make_tuple(make_tuple(orientation.x(), orientation.y(),
                                            orientation.z()),
                                 orientation.w()),
                      stereoCorrection);
}

void export_CameraManipulators()
// clang-format off
{

class_<CameraManipulator, CameraManipulatorPtr, boost::noncopyable>
("CameraManipulator", DOXY_CLASS(bbp::rtneuron::CameraManipulator), no_init)
;

class_<TrackballManipulator, std::shared_ptr<TrackballManipulator>,
       bases<CameraManipulator>, boost::noncopyable>
("TrackballManipulator",
 DOXY_CLASS(bbp::rtneuron::TrackballManipulator), init<>())
    .def("setHomePosition", TrackballManipulator_setHomePosition,
         (arg("eye"), arg("center"), arg("up")),
         DOXY_FN(bbp::rtneuron::TrackballManipulator::setHomePosition))
    .def("getHomePosition", TrackballManipulator_getHomePosition,
         DOXY_FN(bbp::rtneuron::TrackballManipulator::getHomePosition))
;

class_<CameraPath, CameraPathPtr, boost::noncopyable>
cameraPathWrapper(
    "CameraPath", DOXY_CLASS(bbp::rtneuron::CameraPath), init<>());

{
    scope cameraPathScope = cameraPathWrapper;

    class_<CameraPath::KeyFrame, CameraPath::KeyFramePtr, boost::noncopyable>
        ("KeyFrame", DOXY_CLASS(bbp::rtneuron::CameraPath::KeyFrame), init<>())
        /* It seems that is not possible to add argument names and default
           arguments when using make_constructor */
        .def("__init__", make_constructor(KeyFrame_init))
        .def(init<View&>())
        .add_property(
            "position", KeyFrame_getPosition, KeyFrame_setPosition,
            "An (x, y, z) tuple.")
        .add_property(
            "orientation",
            KeyFrame_getOrientation, KeyFrame_setOrientation,
            "An (x, y, z, w) tuple where x, y, z represents the rotation axis"
            " and w the rotation angle in degrees.")
        .add_property(
            "stereoCorrection", &CameraPath::KeyFrame::getStereoCorrection,
            &CameraPath::KeyFrame::setStereoCorrection,
            DOXY_FN(bbp::rtneuron::CameraPath::KeyFrame::getStereoCorrection))
        .def("__repr__", KeyFrame_repr);
}

void (CameraPath::*CameraPath_addKeyFrame1)(double,
                                            const CameraPath::KeyFramePtr&) =
    &CameraPath::addKeyFrame;
void (CameraPath::*CameraPath_addKeyFrame2)(double, View&) =
    &CameraPath::addKeyFrame;

cameraPathWrapper
    .add_property("startTime", &CameraPath::getStartTime,
                  DOXY_FN(bbp::rtneuron::CameraPath::getStartTime))
    .add_property("stopTime", &CameraPath::getStopTime,
                  DOXY_FN(bbp::rtneuron::CameraPath::getStopTime))
    .def("setKeyFrames", CameraPath_setKeyFrames, (arg("frames")),
         DOXY_FN(bbp::rtneuron::CameraPath::setKeyFrames))
    .def("addKeyFrame", CameraPath_addKeyFrame1,
         (arg("seconds"), arg("keyframe")),
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::CameraPath::addKeyFrame(double, const KeyFramePtr&)))
    .def("addKeyFrame", CameraPath_addKeyFrame2,
         (arg("seconds"), arg("view")),
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::CameraPath::addKeyFrame(double, View&)))
    .def("replaceKeyFrame", &CameraPath::replaceKeyFrame,
         (arg("index"), arg("frame")),
         DOXY_FN(bbp::rtneuron::CameraPath::replaceKeyFrame))
    .def("getKeyFrame", &CameraPath::getKeyFrame, (arg("index")),
         DOXY_FN(bbp::rtneuron::CameraPath::getKeyFrame))
    .def("removeKeyFrame", &CameraPath::removeKeyFrame, (arg("index")),
         DOXY_FN(bbp::rtneuron::CameraPath::removeKeyFrame))
    .def("getKeyFrames", CameraPath_getKeyFrames,
         DOXY_FN(bbp::rtneuron::CameraPath::getKeyFrames))
    .def("clear", &CameraPath::clear,
         DOXY_FN(bbp::rtneuron::CameraPath::clear))
    .def("load", &CameraPath::load, (arg("filename")),
         DOXY_FN(bbp::rtneuron::CameraPath::load))
    .def("save", &CameraPath::save, (arg("filename")),
         DOXY_FN(bbp::rtneuron::CameraPath::save))
;

class_<CameraPathManipulator, CameraPathManipulatorPtr,
       bases<CameraManipulator>, boost::noncopyable>
cameraPathManipulatorWrapper("CameraPathManipulator", init<>());
{
    scope cameraPathManipulatorScope = cameraPathManipulatorWrapper;

    enum_<CameraPathManipulator::LoopMode>(
        "LoopMode", DOXY_ENUM(bbp::rtneuron::CameraPathManipulator::LoopMode))
    .value("NONE", CameraPathManipulator::LOOP_NONE)
    .value("REPEAT", CameraPathManipulator::LOOP_REPEAT)
    .value("SWING", CameraPathManipulator::LOOP_SWING)
    ;

    cameraPathManipulatorWrapper
    .def("load", &CameraPathManipulator::load, (arg("fileName")),
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::load))
    .def("setPath", &CameraPathManipulator::setPath)
    .add_property("frameDelta", &CameraPathManipulator::getFrameDelta,
         &CameraPathManipulator::setFrameDelta,
        (std::string("Get: ") +
        DOXY_FN(bbp::rtneuron::CameraPathManipulator::getFrameDelta) +
         "\nSet: " +
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::setFrameDelta)).c_str())
    .add_property("playbackStop", &CameraPathManipulator::getPlaybackStop,
         &CameraPathManipulator::setPlaybackStop,
        (std::string("Get: ") +
        DOXY_FN(bbp::rtneuron::CameraPathManipulator::getPlaybackStop) +
         "\nSet: " +
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::setPlaybackStop)).c_str())
    .add_property("playbackStart", &CameraPathManipulator::getPlaybackStart,
         &CameraPathManipulator::setPlaybackStart,
        (std::string("Get: ") +
        DOXY_FN(bbp::rtneuron::CameraPathManipulator::getPlaybackStart) +
         "\nSet: " +
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::setPlaybackStart)).c_str())
    .def("setPlaybackInterval", &CameraPathManipulator::setPlaybackInterval,
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::setPlaybackInterval))
    .add_property("loopMode", &CameraPathManipulator::getLoopMode,
         &CameraPathManipulator::setLoopMode,
        (std::string("Get: ") +
        DOXY_FN(bbp::rtneuron::CameraPathManipulator::getLoopMode) +
         "\nSet: " +
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::setLoopMode)).c_str())
    .def("getKeyFrame", CameraPathManipulator_getKeyFrame,
         (arg("milliseconds")),
         DOXY_FN(bbp::rtneuron::CameraPathManipulator::getKeyFrame))
    ;
}

#ifdef RTNEURON_USE_VRPN

class_<VRPNManipulator, std::shared_ptr<VRPNManipulator>,
       bases<CameraManipulator>,
       boost::noncopyable>
vrpnManipulator("VRPNManipulator", DOXY_CLASS(bbp::rtneuron::VRPNManipulator),
                init<VRPNManipulator::DeviceType,
                     optional<const std::string &> >((arg("types"),
                                                      arg("url"))));
{
    vrpnManipulator
        .def(init<VRPNManipulator::DeviceType, const AttributeMap&>(
                 (arg("type"), arg("attributes")),
             /* Do not wrap this line or change whitespace */
             DOXY_FN(bbp::rtneuron::VRPNManipulator::VRPNManipulator(const DeviceType, const AttributeMap&))));

    scope vrpnManipulatorScope = vrpnManipulator;
    enum_<VRPNManipulator::DeviceType>("DeviceType")
    .value("GYRATION_MOUSE", VRPNManipulator::GYRATION_MOUSE)
    .value("INTERSENSE_WAND", VRPNManipulator::INTERSENSE_WAND)
    .value("SPACE_MOUSE", VRPNManipulator::SPACE_MOUSE)
#  ifdef RTNEURON_USE_WIIUSE
    .value("WIIMOTE", VRPNManipulator::WIIMOTE)
#  endif
    ;
}

#endif
// clang-format off
}
