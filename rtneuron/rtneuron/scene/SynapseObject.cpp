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

#include "SynapseObject.h"

#include "SceneOperation.h"
#include "SubScene.h"

#include "config/Globals.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "util/attributeMapHelpers.h"
#include "util/vec_to_vec.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>
#include <brain/synapse.h>
#include <brain/synapses.h>
#include <brain/synapsesIterator.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
inline osg::Vec3 _center(const brain::Vector4f& sample)
{
    return osg::Vec3(sample[0], sample[1], sample[2]);
}

/* Assumes the positions is not in the soma, otherwise it will throw */
Vector3f _computePosition(const Neuron& neuron, const size_t sectionID,
                          const size_t segmentID, const float segmentDistance)
{
    /* Obtaining the synapse location in local coordinates */
    const auto& morphology = *neuron.getMorphology();
    const auto section = morphology.getSection(sectionID);
    assert(segmentID < section.getNumSamples());
    const auto& start = _center(section[segmentID]);
    const auto& end = _center(section[segmentID + 1]);
    const float normDistance = segmentDistance / (end - start).length();
    auto local = end * normDistance + start * (1 - normDistance);
    /* Moving to global */
    const auto& rotation = neuron.getOrientation();
    const auto global = rotation * local + neuron.getPosition();
    return vec_to_vec(global);
}

Vector3f _computeSomaSynapseLocation(const Neuron& post, const Neuron& pre,
                                     const bool surface,
                                     const brain::Synapse& synapse)
{
    const auto soma = post.getMorphology()->getSoma();
    auto position = post.getPosition() + vec_to_vec(soma.getCentroid());

    if (surface)
    {
        const auto& bouton =
            _computePosition(pre, synapse.getPresynapticSectionID(),
                             synapse.getPresynapticSegmentID(),
                             synapse.getPresynapticDistance());
        auto direction = vec_to_vec(bouton) - position;
        direction.normalize();
        position += direction * soma.getMaxRadius();
    }
    return vec_to_vec(position);
}

Vector3fs _computeSynapseLocations(const brain::Synapses& synapses,
                                   Scene::_Impl* parent,
                                   SynapseObject::Type type, const bool surface)
{
    /* First of all collecting the GIDs of the neurons for which we need the
       data and creating the necessary Neurons containers. */
    Neurons neurons;
    Neurons opposite;
    if (type == SynapseObject::AFFERENT)
    {
        GIDSet postGIDs(synapses.postGIDs(),
                        synapses.postGIDs() + synapses.size());
        neurons = Neurons(postGIDs, parent->getCircuit());
        if (surface)
        {
            GIDSet preGIDs;
            auto hint = preGIDs.end();
            /* Get only the GIDs of the cells that are forming a synapse
               in the soma. */
            for (auto synapse : synapses)
            {
                if (synapse.getPostsynapticSectionID() == 0)
                    hint = preGIDs.insert(hint, synapse.getPresynapticGID());
            }
            opposite = Neurons(preGIDs, parent->getCircuit());
        }
    }
    else
    {
        GIDSet preGIDs(synapses.preGIDs(),
                       synapses.preGIDs() + synapses.size());
        neurons = Neurons(preGIDs, parent->getCircuit());
    }

    using Syn = brain::Synapse;
    const auto getGID = type == SynapseObject::AFFERENT
                            ? &Syn::getPostsynapticGID
                            : &Syn::getPresynapticGID;

    const auto getSection = type == SynapseObject::AFFERENT
                                ? &Syn::getPostsynapticSectionID
                                : &Syn::getPresynapticSectionID;
    const auto getSegment = type == SynapseObject::AFFERENT
                                ? &Syn::getPostsynapticSegmentID
                                : &Syn::getPresynapticSegmentID;
    const auto getDistance = type == SynapseObject::AFFERENT
                                 ? &Syn::getPostsynapticDistance
                                 : &Syn::getPresynapticDistance;

    Vector3fs positions;
    for (const auto& synapse : synapses)
    {
        const auto gid = (synapse.*getGID)();
        const auto& neuron = *neurons.find(gid);
        const auto sectionID = (synapse.*getSection)();

        if (sectionID == 0 && type == SynapseObject::AFFERENT)
        {
            const auto preGID = synapse.getPresynapticGID();
            positions.push_back(
                _computeSomaSynapseLocation(neuron, *opposite.find(preGID),
                                            surface, synapse));
            continue;
        }

        positions.push_back(_computePosition(neuron, sectionID,
                                             (synapse.*getSegment)(),
                                             (synapse.*getDistance)()));
    }
    return positions;
}
}

SynapseObject::SynapseObject(const brain::Synapses& synapses,
                             const Type synapseType,
                             const AttributeMap& attributes,
                             Scene::_Impl* parent)
    : SceneObject(parent, attributes)
    , _synapses(synapses)
    , _type(synapseType)
    , _radius(Globals::getDefaultSynapseRadius())
{
    using namespace AttributeMapHelpers;
    AttributeMap& attr = getAttributes();
    blockAttributeMapSignals();

    osg::Vec4 dummy1;
    if (!getColor(attr, "color", dummy1))
    {
        osg::Vec4 c;
        if (synapseType == SynapseObject::EFFERENT)
            c = Globals::getDefaultEfferentSynapseColor();
        else
            c = Globals::getDefaultAfferentSynapseColor();
        attr.set("color", c[0], c[1], c[2], c[3]);
    }

    if (attr.get("radius", _radius) != 1)
        attr.set("radius", _radius);

    if (attr.get("surface", _surface) != 1)
    {
        attr.set("surface", true);
        _surface = true;
    }

    if (attr.get("visible", _visible) != 1)
        attr.set("visible", true);

    unblockAttributeMapSignals();
    _parent->_synapsesBound.init();
}

const Vector3fs& SynapseObject::getLocations() const
{
    if (!_locations.empty())
        return _locations;

    const auto functor =
        _surface ? (_type == SynapseObject::EFFERENT
                        ? &brain::Synapse::getPresynapticSurfacePosition
                        : &brain::Synapse::getPostsynapticSurfacePosition)
                 : (_type == SynapseObject::EFFERENT
                        ? &brain::Synapse::getPresynapticCenterPosition
                        : &brain::Synapse::getPostsynapticCenterPosition);
    try
    {
        _locations.reserve(_synapses.size());
        for (const auto& synapse : _synapses)
            _locations.emplace_back((synapse.*functor)());
    }
    catch (...)
    {
        /* Position files not available. */
        _locations =
            _computeSynapseLocations(_synapses, _parent, _type, _surface);
    }
    return _locations;
}

void SynapseObject::cleanup()
{
    lunchbox::ScopedWrite mutex(_parentLock);
    _parent->_synapsesBound.init();
    /* As long as subscenes are deleted when objects are removed there's
       no need to do anything else here. */
}

void SynapseObject::onAttributeChangingImpl(
    const AttributeMap&, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    using namespace AttributeMapHelpers;

    if (name == "radius")
    {
        (void)(double) parameters;
    }
    else if (name == "color")
    {
        osg::Vec4 color;
        getColor(parameters, color);
    }
    else if (name == "visible")
    {
        (bool)parameters;
    }
    else
        throw std::runtime_error("Unknown or inmmutable attribute: " + name);
}

void SynapseObject::update(SubScene& subScene, const AttributeMap& attributes)
{
    core::SphereSet::SubSetID id = subScene.findSynapseSubset(getID());
    if (!id)
        return;
    SphereSet& spheres = subScene.getSynapses();

    for (const auto& attribute : attributes)
    {
        const std::string& name = attribute.first;
        if (name == "radius")
        {
            _radius = (double)attribute.second;
            if (_visible)
                spheres.updateRadius(id, _radius);
        }
        else if (name == "color")
        {
            using namespace AttributeMapHelpers;
            osg::Vec4 color;
            getColor(attribute.second, color);
            spheres.updateColor(id, color);
        }
        else if (name == "visible")
        {
            _visible = (bool)attribute.second;
            if (_visible)
                spheres.updateRadius(id, _radius);
            else
                spheres.updateRadius(id, 0);
        }
    }
}
}
}
}
