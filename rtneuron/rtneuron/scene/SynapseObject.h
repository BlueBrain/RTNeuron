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

#ifndef RTNEURON_SCENE_SYNAPSEOBJECT_H
#define RTNEURON_SCENE_SYNAPSEOBJECT_H

#include "SceneObject.h"

#include "../SceneImpl.h"
#include "../types.h"

#include <brain/synapses.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class SynapseObject : public SceneObject
{
public:
    /*--- Public declarations ---*/
    enum Type
    {
        AFFERENT,
        EFFERENT
    };

    /*--- Public constructors/destructor ---*/

    SynapseObject(const brain::Synapses& synapses, Type type,
                  const AttributeMap& attributes, Scene::_Impl* parent);

    /*--- Public member functions ---*/
    const brain::Synapses& getSynapses() const { return _synapses; }
    /** Get the synapse locations according to the attributes and type. */
    const Vector3fs& getLocations() const;

    Type getType() const { return _type; }
    virtual boost::any getObject() const
    {
        /** \bug It's not possible to distinguish between afferent and
            efferent synapses at the Python side. */
        /* This is not very efficient, but there's no way to make a shallow
           copy and it may not even be desirable for thread safety-reasons. */
        return boost::any(_synapses);
    }

    virtual void cleanup();

    virtual void onAttributeChangingImpl(
        const AttributeMap& attributes, const std::string& name,
        const AttributeMap::AttributeProxy& parameters);

    void update(SubScene& subscene, const AttributeMap& attributes) final;

    void update(Scene::_Impl&, const AttributeMap&) final{};

    using SceneObject::update;

private:
    /*--- Private member attributes ---*/
    brain::Synapses _synapses;
    Type _type;
    bool _surface;
    bool _visible;
    double _radius; /* Needs to be double to get it from AttributeMaps created
                       in Python. */
    mutable Vector3fs _locations;
};
}
}
}
#endif
