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

#include "GeometryObject.h"
#include "render/SceneStyle.h"

#include <osg/LineWidth>
#include <osg/MatrixTransform>
#include <osg/PolygonOffset>
#include <osg/PrimitiveSet>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
bool _isLines(const osg::PrimitiveSet& primitive)
{
    switch (primitive.getMode())
    {
    case osg::PrimitiveSet::LINES:
    case osg::PrimitiveSet::LINE_STRIP:
    case osg::PrimitiveSet::LINE_LOOP:
        return true;
    default:
        return false;
    }
}

osg::BoundingBox _computeBoundingBox(osg::Array& vertices, const float radius)
{
    osg::BoundingBox bbox;
    if (vertices.getType() == osg::Array::Vec3ArrayType)
    {
        if (radius != 0)
            for (auto&& v : static_cast<osg::Vec3Array&>(vertices))
                bbox.expandBy(osg::BoundingSphere(v, radius));
        else
            for (auto&& v : static_cast<osg::Vec3Array&>(vertices))
                bbox.expandBy(v);
    }
    else
    {
        for (auto&& v : static_cast<osg::Vec4Array&>(vertices))
            bbox.expandBy(
                osg::BoundingSphere(osg::Vec3(v[0], v[1], v[2]), v[3]));
    }
    return bbox;
}
}

GeometryObject::GeometryObject(
    const osg::ref_ptr<osg::Array>& vertices,
    const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
    const osg::ref_ptr<osg::Vec4Array>& colors,
    const osg::ref_ptr<osg::Vec3Array>& normals, const AttributeMap& attributes,
    Scene::_Impl* parent)
    : ModelObject(attributes, parent)
    , _geometry(new osg::Geometry())
    , _geode(new osg::Geode())
{
    if (vertices->getType() != osg::Array::Vec3ArrayType &&
        (primitive || vertices->getType() != osg::Array::Vec4ArrayType))
    {
        throw std::runtime_error(
            "Invalid number of components in vertex array");
    }

    _geometry->setVertexArray(vertices);
    _geometry->setUseVertexBufferObjects(true);
    _geometry->setUseDisplayList(false);

    _assignOrCreatePrimitiveSet(primitive, attributes);

    if (colors)
    {
        _geometry->setColorArray(colors);
        assert(colors->size() == vertices->getNumElements() ||
               colors->size() == 1);
        if (colors->size() == 1)
            _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
        else
            _geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    }
    else
    {
        osg::ref_ptr<osg::Vec4Array> white = new osg::Vec4Array();
        white->push_back(osg::Vec4(1, 1, 1, 1));
        _geometry->setColorArray(white);
        _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    }

    if (normals)
    {
        _geometry->setNormalArray(normals);
        assert(normals->size() == vertices->getNumElements());
        _geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    _geode->addDrawable(_geometry);
    _node = _geode;
}

void GeometryObject::_assignOrCreatePrimitiveSet(
    const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
    const AttributeMap& attributes)
{
    const auto& vertices = _geometry->getVertexArray();
    const size_t vertexCount = vertices->getNumElements();

    if (primitive)
    {
        _geometry->addPrimitiveSet(primitive);
        _type = _isLines(*primitive) ? Type::lines : Type::mesh;
        if (_type == Type::lines)
        {
            osg::StateSet* stateSet = _geometry->getOrCreateStateSet();
            const double width = attributes("line_width", 1.0);
            stateSet->setAttributeAndModes(new osg::LineWidth(width));
        }
        if (_type == Type::mesh)
        {
            double factor = 0.0, units = 0.0;
            if (attributes.get("polygon_offset", factor, units) == 2)
            {
                osg::StateSet* stateSet = _geometry->getOrCreateStateSet();
                stateSet->setAttributeAndModes(
                    new osg::PolygonOffset(factor, units));
            }
        }
    }
    else
    {
        _geometry->addPrimitiveSet(
            new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertexCount));
        const std::string style = attributes("point_style", "points");
        if (style == "spheres")
            _type = Type::spheres;
        else if (style == "points")
            _type = Type::points;
        else if (style == "circles")
            _type = Type::circles;
        else
            throw std::runtime_error("Invalid point style");

        /* Adding the global point size or sphere radius uniform if
           necessary. */
        const float pointSize = (double)attributes("point_size", 1.0);
        if (vertices->getType() == osg::Array::Vec3ArrayType)
        {
            _geometry->getOrCreateStateSet()->addUniform(new osg::Uniform(
                _type == Type::spheres ? "radius" : "pointSize", pointSize));
        }

        if (_type == Type::spheres)
        {
            /* Computing a better bounding box and making sure that OSG doesn't
               try to adjust it later. */
            _geometry->setInitialBound(
                _computeBoundingBox(*vertices, pointSize));
            _geometry->setComputeBoundingBoxCallback(
                new osg::Drawable::ComputeBoundingBoxCallback());
        }
        else if (_type == Type::points || _type == Type::circles)
        {
            /* In these cases there's nothing simple we can do as the bounding
               box depends on the camera positions. Since this geometry is
               easy to render we simply disable culling in this case.
               We still provide a bounding box for computing the camera home
               position. */
            /* The radius can't be 0, or the drawable will still be culled. */
            _geometry->setInitialBound(_computeBoundingBox(*vertices, 1));
            _geometry->setComputeBoundingBoxCallback(
                new osg::Drawable::ComputeBoundingBoxCallback());
        }
    }
}

void GeometryObject::applyStyle(const SceneStylePtr& style)
{
    _style = style;
    AttributeMap extra;

    /* Initialized to avoid spurious warning in release build */
    core::SceneStyle::StateType stateType = core::SceneStyle::StateType();
    switch (_type)
    {
    case Type::mesh:
        stateType = _current("flat", false) ? core::SceneStyle::FLAT_MESH
                                            : core::SceneStyle::MESH;
        break;
    case Type::lines:
        stateType = core::SceneStyle::LINES;
        break;
    case Type::circles:
        extra.set("circles", true);
    /* falls through */
    case Type::points:
        if (_geometry->getVertexArray()->getType() == osg::Array::Vec3ArrayType)
            extra.set("use_point_size_uniform", true);
        stateType = core::SceneStyle::POINTS;
        break;
    case Type::spheres:
        if (_geometry->getVertexArray()->getType() == osg::Array::Vec3ArrayType)
            extra.set("use_radius_uniform", true);
        stateType = core::SceneStyle::SPHERES;
        break;
    }

    _node->setStateSet(style->getStateSet(stateType, extra));
}
}
}
}
