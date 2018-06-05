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

#include "rtneuron/ColorMap.h"
#include "docstrings.h"
#include "rtneuron/types.h"

#include "helpers.h"

#include <osg/Vec4>

#include <vmmlib/vector.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/stl_iterator.hpp>
#include <rtneuron/detail/attributeMapTypeRegistration.h>

#include <boost/serialization/export.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

namespace
{
void ColorMap_setPoints(ColorMap* colorMap, const dict& d)
{
    ColorMap::ColorPoints points;

    list pointList = d.items();
    while (len(pointList) != 0)
    {
        tuple valueColor = extract<tuple>(pointList.pop());
        double value = extract<double>(valueColor[0]);
        Vector4f c = extract_Vector4(valueColor[1]);
        points[value] = osg::Vec4(c[0], c[1], c[2], c[3]);
    }
    colorMap->setPoints(points);
}

dict ColorMap_getPoints(const ColorMap* colorMap)
{
    const ColorMap::ColorPoints& points = colorMap->getPoints();
    dict d;
    for (ColorMap::ColorPoints::const_iterator point = points.begin();
         point != points.end(); ++point)
    {
        const osg::Vec4& color = point->second;
        d[point->first] = make_tuple(color[0], color[1], color[2], color[3]);
    }

    return d;
}

tuple ColorMap_getRange(const ColorMap* colorMap)
{
    float min, max;
    colorMap->getRange(min, max);
    return make_tuple(min, max);
}

tuple ColorMap_getColor(const ColorMap* colorMap, float value)
{
    osg::Vec4 color = colorMap->getColor(value);
    return make_tuple(color[0], color[1], color[2], color[3]);
}

void (ColorMap::*ColorMap_load)(const std::string&) = &ColorMap::load;

void (ColorMap::*ColorMap_save)(const std::string&) = &ColorMap::save;
}

namespace bbp
{
namespace rtneuron
{
ATTRIBUTE_MAP_HAS_DIRTY_SIGNAL(ColorMap)
}
}
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<ColorMapPtr>, "a:ColorMap");

void export_ColorMap()
// clang-format off
{
AttributeMap::registerType<ColorMapPtr>("ColorMap");

class_<ColorMap, ColorMapPtr, boost::noncopyable>("ColorMap")
    .def("setPoints", ColorMap_setPoints, (arg("colorPoints")),
         DOXY_FN(bbp::rtneuron::ColorMap::setPoints))
    .def("getPoints", ColorMap_getPoints,
         DOXY_FN(bbp::rtneuron::ColorMap::getPoints))
    .def("setRange", &ColorMap::setRange, (arg("min"), arg("max")),
         DOXY_FN(bbp::rtneuron::ColorMap::setRange))
    .def("getRange", ColorMap_getRange,
         DOXY_FN(bbp::rtneuron::ColorMap::getRange))
    .def("getColor", ColorMap_getColor, (arg("value")),
         DOXY_FN(bbp::rtneuron::ColorMap::getColor))
    .def("load", ColorMap_load, DOXY_FN(bbp::rtneuron::ColorMap::load))
    .def("save", ColorMap_save, DOXY_FN(bbp::rtneuron::ColorMap::save))
    .add_property("textureSize", &ColorMap::getTextureSize,
                  &ColorMap::setTextureSize,
                  DOXY_FN(bbp::rtneuron::ColorMap::setTextureSize));
}
