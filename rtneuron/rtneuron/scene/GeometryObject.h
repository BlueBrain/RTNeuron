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

#ifndef RTNEURON_API_SCENE_GEOMETRY_OBJECT_H
#define RTNEURON_API_SCENE_GEOMETRY_OBJECT_H

#include "ModelObject.h"

#include <osg/Geode>
#include <osg/Geometry>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class GeometryObject : public ModelObject
{
public:
    /*--- Public Constructors/destructor ---*/
    GeometryObject(const osg::ref_ptr<osg::Array>& vertices,
                   const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
                   const osg::ref_ptr<osg::Vec4Array>& colors,
                   const osg::ref_ptr<osg::Vec3Array>& normals,
                   const AttributeMap& attributes, Scene::_Impl* parent);

    /*--- Public member functions ---*/

    boost::any getObject() const final
    {
        /* This class doesn't have a corresponding Python object */
        return boost::any();
    }

    void applyStyle(const SceneStylePtr& style) override;

    void onAttributeChangingImpl(const AttributeMap&, const std::string& name,
                                 const AttributeMap::AttributeProxy&) final
    {
        throw std::runtime_error("Unknown or inmmutable attribute: " + name);
    }

private:
    /*--- Private member variables ---*/
    enum class Type
    {
        mesh,
        lines,
        points,
        circles,
        spheres
    };
    Type _type;
    osg::ref_ptr<osg::Geometry> _geometry;
    osg::ref_ptr<osg::Geode> _geode;

    void _assignOrCreatePrimitiveSet(
        const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
        const AttributeMap& attributes);
};
}
}
}
#endif
