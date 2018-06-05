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

#include "ModelObject.h"
#include "render/SceneStyle.h"
#include "util/attributeMapHelpers.h"

#include <osg/MatrixTransform>

#include <osgDB/ReadFile>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
class ModelPropertiesVisitor : public osg::NodeVisitor
{
public:
    bool hasVertexColors = false;

#ifndef OSG_GL3_AVAILABLE
    virtual void apply(osg::Geode& geode)
    {
        for (auto drawable : geode.getDrawableList())
        {
            auto* geometry = drawable->asGeometry();
            if (geometry)
                apply(*geometry);
        }
    }
#endif

    void apply(osg::Geometry& geometry)
    {
        if (geometry.getColorArray() != 0)
            hasVertexColors = true;
    }
};

#ifdef OSG_GL3_AVAILABLE
const char* colorUniformName = "osg_Color";
#else
const char* colorUniformName = "globalColor";
#endif

osg::Uniform* _getColorGlobalUniform(osg::Node* parent)
{
    osg::Group* group = parent->asGroup();
    osg::Node* node = group ? group->getChild(0) : nullptr;
    osg::StateSet* stateSet = node ? node->getStateSet() : nullptr;
    return stateSet ? stateSet->getUniform(colorUniformName) : nullptr;
}
}

ModelObject::ModelObject(const std::string& filename,
                         const osg::Matrix& transform,
                         const AttributeMap& attributes, Scene::_Impl* parent)
    : SceneObject(parent, attributes)
{
    using namespace AttributeMapHelpers;

    osg::ref_ptr<osg::MatrixTransform> node(new osg::MatrixTransform);
    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(filename);
    if (!model)
    {
        throw std::runtime_error("Could not load any model from file: " +
                                 filename);
    }

    node->addChild(model);
    node->setMatrix(transform);
    node->setName(filename);
    _node = node;

    ModelPropertiesVisitor properties;
    node->accept(properties);
    if (!properties.hasVertexColors)
    {
        /* Checking if the user provided a color or assigning white by
           default. */
        AttributeMap& attr = getAttributes();
        blockAttributeMapSignals();
        osg::Vec4 color(1, 1, 1, 1);
        if (!getColor(attr, "color", color))
            attr.set("color", 1, 1, 1, 1);
        unblockAttributeMapSignals();

        model->getOrCreateStateSet()->addUniform(
            new osg::Uniform(colorUniformName, color));
    }
}

ModelObject::~ModelObject()
{
}

void ModelObject::applyStyle(const SceneStylePtr& style)
{
    _style = style;

    AttributeMap extra;
    if (_getColorGlobalUniform(_node))
        extra.set("color_uniform", true);

    if (_current("flat", false))
        _node->setStateSet(
            style->getStateSet(core::SceneStyle::FLAT_MESH, extra));
    else
        _node->setStateSet(style->getStateSet(core::SceneStyle::MESH, extra));
}

void ModelObject::onAttributeChangingImpl(
    const AttributeMap&, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    if (name == "color")
    {
        if (!_getColorGlobalUniform(_node))
        {
            std::cerr << "Warning: Changing the overall color will"
                         " have no effect in this model"
                      << std::endl;
        }
        osg::Vec4 color;
        AttributeMapHelpers::getRequiredColor(parameters, color);
    }
    else
        throw std::runtime_error("Unknown or inmmutable attribute: " + name);
}

void ModelObject::applyAttribute(const AttributeMap& attributes,
                                 const std::string& name)
{
    if (name == "color")
    {
        osg::Vec4 color;
        AttributeMapHelpers::getColor(attributes, "color", color);
        osg::Uniform* uniform = _getColorGlobalUniform(_node);
        if (uniform)
            uniform->set(color);
    }
}
}
}
}
