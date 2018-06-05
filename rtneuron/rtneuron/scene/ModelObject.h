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

#ifndef RTNEURON_SCENE_MODEL_OBJECT_H
#define RTNEURON_SCENE_MODEL_OBJECT_H

#include "SceneObject.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class ModelObject : public SceneObject
{
public:
    /*--- Public Constructors/destructor ---*/

    ModelObject(const std::string& filename, const osg::Matrix& transform,
                const AttributeMap& attributes, Scene::_Impl* parent);

    ~ModelObject();

    /*--- Public member functions ---*/

    boost::any getObject() const override
    {
        return boost::any(_node->getName());
    }

    osg::Node* getNode() { return _node.get(); }
    void cleanup() final { _parent->_toRemove.push_back(_node); }
    virtual void applyStyle(const SceneStylePtr& style);

    void onAttributeChangingImpl(const AttributeMap&, const std::string& name,
                                 const AttributeMap::AttributeProxy&) override;

protected:
    /*--- Protected member variables ---*/

    SceneStylePtr _style;
    osg::ref_ptr<osg::Node> _node;

    /*--- Protected Constructors/destructor ---*/

    ModelObject(const AttributeMap& attributes, Scene::_Impl* parent)
        : SceneObject(parent, attributes)
    {
    }

    /*--- Protected member functions ---*/

    void applyAttribute(const AttributeMap&, const std::string&);

    bool preUpdateImplementation() final { return false; }
};
}
}
}
#endif
