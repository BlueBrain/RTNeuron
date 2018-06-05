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
#ifndef RTNEURON_VIEWSTYLE_H
#define RTNEURON_VIEWSTYLE_H

#include "AttributeMap.h"
#include "types.h"

#include "viewer/osgEq/SceneDecorator.h"

#include <osg/Referenced>

#include <boost/serialization/export.hpp>

/* Forward references */
namespace boost
{
namespace serialization
{
class access;
}
}

namespace osg
{
class Node;
class Group;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
class AppInterface;
class ViewStyleData;

/**
   This class implements all the logic to setup and change global rendering
   style parameters that are view dependent.

   Handles dynamic attribute map changes related to the rendering style,
   osg::StateSet, osg::Uniform and and other callbacks that affect how
   the scene should be rendered under this view.

   Valid attributes are:
   - clod_threshold (float)
   - lod_bias (float)
   - display_simulation (bool)
   - inflation_factor (float)
   - probe_threshold (float)
   - probe_color (floatx4)
   - spike_tail (float)
*/
class ViewStyle : public osgEq::SceneDecorator
{
public:
    /*--- Public constructors/destructor ---*/

    ViewStyle(const AttributeMap& attributes = AttributeMap());

    ~ViewStyle();

    /*--- Public Member functions ---*/

    /**
       Given the root scenegraph node for a view, returns a root
       with the view dependent mutable style that contains the scene as
       a subgraph.
    */
    osg::Group* decorateScenegraph(osg::Node* root) final;

    /**
       To be called only from osgEq::Channel.

       @sa osgEq::SceneDecorator::updateCamera
     */
    void updateCamera(osg::Camera* camera) const final;

    /**
       To be called only from osgEq::Channel.

       @sa osgEq::SceneDecorator::update
     */
    void update() final;

    osgEq::SceneDecoratorPtr clone() const final;

    /**
       Returns the attribute map with the runtime configurable attributes.

       Attribute changes are applied during the execution of the update
       callback attached to the osg::Node returned by stylizedScenegraph
       in a thread-safe manner.
     */
    AttributeMap& getAttributes();
    const AttributeMap& getAttributes() const;

    void setGUICallbacks(AppInterface* gui);

    ViewStyleData* getData();

    /**
       \brief Validates an attribute change and throws is the attribute has
       bad type or it is unknown.

       To be called externally from View when an attribute is being changed.
    */
    void validateAttributeChange(
        const AttributeMap& attributes, const std::string& name,
        const AttributeMap::AttributeProxy& parameters);

    /** \brief Get the default attributes with their values */
    void getDefaultAttributes(AttributeMap& attributes);

private:
    /*--- Private member attributes ---*/

    class Impl;
    Impl* _impl;

    /*--- Private member functions ---*/

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version);

    template <class Archive>
    void load(Archive& ar, const unsigned int version);
    /**
       Only serializes the attribute updates since the last save.

       Declared const by requirement but this operation is not idempotent.
    */
    template <class Archive>
    void save(Archive& ar, const unsigned int version) const;
};
}
}
}

#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
BOOST_CLASS_EXPORT_KEY(bbp::rtneuron::core::ViewStyle)

#endif
