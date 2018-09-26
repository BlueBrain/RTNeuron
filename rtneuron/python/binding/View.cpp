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

#include "docstrings.h"
#include "gil.h"

#include "rtneuron/Camera.h"
#include "rtneuron/CameraManipulator.h"
#include "rtneuron/ColorMap.h"
#include "rtneuron/Scene.h"
#include "rtneuron/View.h"
#include "rtneuron/ui/Pointer.h"

#ifdef override
#undef override
#endif
#include <boost/python.hpp>

#include <osgGA/TrackballManipulator>

using namespace boost::python;
using namespace bbp::rtneuron;

// export_View -----------------------------------------------------------------

AttributeMap& View_getAttributes(View& view)
{
    AttributeMap& attributes = view.getAttributes();
    /** \todo Move this assignment to the class constructor so it's only
        done once */
    attributes.setExtraDocstring(DOXY_FN(bbp::rtneuron::View::getAttributes));
    return attributes;
}

void View_snapshot(View* view, const std::string& filename,
                   const bool waitForCompletion)
{
    ReleaseGIL release;
    view->snapshot(filename, waitForCompletion);
}

void View_snapshot1(View* view, const std::string& filename, const float scale)
{
    ReleaseGIL release;
    view->snapshot(filename, scale);
}

void View_snapshot2(View* view, const std::string& filename,
                    const tuple& resolution)
{
    const size_t resX = extract<size_t>(resolution[0]);
    const size_t resY = extract<size_t>(resolution[1]);

    ReleaseGIL release;
    view->snapshot(filename, resX, resY);
}

void View_computeHomePosition(View* view)
{
    ReleaseGIL release;
    view->computeHomePosition();
}

void export_View()
// clang-format off
{

class_<View, ViewPtr, boost::noncopyable>
("View", DOXY_CLASS(bbp::rtneuron::View), no_init)
    .def("record", &View::record, (arg("enable")),
         DOXY_FN(bbp::rtneuron::View::record))
    .def("snapshot", View_snapshot,
         (arg("fileName"), arg("waitForCompletion") = true),
         /* Don't wrap this line of change whitespace */
         DOXY_FN(bbp::rtneuron::View::snapshot(const std::string&, bool)))
    .def("snapshot", View_snapshot1, (arg("fileName"), arg("scale")),
         /* Don't wrap this line of change whitespace */
         DOXY_FN(bbp::rtneuron::View::snapshot(const std::string&, float)))
    .def("snapshot", View_snapshot2, (arg("fileName"), arg("resolution")),
         /* Don't wrap this line of change whitespace */
         DOXY_FN(bbp::rtneuron::View::snapshot(const std::string&, const tuple&)))
    .add_property("attributes", make_function(View_getAttributes,
                                              return_internal_reference<>()),
        DOXY_FN(bbp::rtneuron::View::getAttributes))
    .add_property("scene", &View::getScene, &View::setScene,
        (std::string("Get: ") + DOXY_FN(bbp::rtneuron::View::getScene) +
         "\nSet: " + DOXY_FN(bbp::rtneuron::View::setScene)).c_str())
    .add_property("camera", &View::getCamera,
        (std::string("Get only: ") +
         DOXY_FN(bbp::rtneuron::View::getCamera)).c_str())
    .add_property("cameraManipulator", &View::getCameraManipulator,
         &View::setCameraManipulator,
        (std::string("Get: ") +
        DOXY_FN(bbp::rtneuron::View::getCameraManipulator) +
         "\nSet: " +
         DOXY_FN(bbp::rtneuron::View::setCameraManipulator)).c_str())
    .def("computeHomePosition", View_computeHomePosition,
         DOXY_FN(bbp::rtneuron::View::computeHomePosition))
    .add_property("pointer", &View::getPointer, &View::setPointer,
        (std::string("Get: ") + DOXY_FN(bbp::rtneuron::View::getPointer) +
         "\nSet: " + DOXY_FN(bbp::rtneuron::View::setPointer)).c_str())
;

}
// clang-format on
