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

#include "NeuronObject.h"
#include "ClientSceneObject.h"
#include "SubScene.h"
#include "data/Neuron.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "render/NeuronColoring.h"
#include "util/attributeMapHelpers.h"

#include <boost/serialization/export.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper functions
*/

typedef std::vector<ColorMapPtr> ColorMaps;

namespace
{
bool _needsUpgrade(const Neuron& neuron, const RepresentationMode mode)
{
    const RepresentationMode restrictedMode =
        neuron.getRestrictedRepresentationMode();
    if (restrictedMode == mode)
        return false;
    return restrictedMode == RepresentationMode::NO_DISPLAY ||
           (restrictedMode == RepresentationMode::SOMA &&
            mode != RepresentationMode::SOMA &&
            mode != RepresentationMode::NO_DISPLAY);
}

boost::any _gidsFromNeurons(const Neurons& neurons)
{
    uint32_ts gids;
    gids.reserve(neurons.size());
    for (const auto& neuron : neurons)
        gids.push_back(neuron->getGID());
    return boost::any(std::move(gids));
}

/* Applies all except color scheme related attributes.
   Coloring attributes are handled separately using NeuronColoring. */
void _applyAttribute(const AttributeMap& attributes, const std::string& name,
                     const Neurons& neurons, SubScene& scene,
                     const NeuronColoring& coloring)
{
    using namespace AttributeMapHelpers;
    if (name == "mode")
    {
        const RepresentationMode mode =
            getEnum<RepresentationMode>(attributes, "mode");

        std::stringstream progressMessage;
        progressMessage << "Upgrading representation mode of " << neurons.size()
                        << " neurons";
        size_t progress = 0;

        bool reportProgress = false;
        for (const auto& neuron : neurons)
        {
            reportProgress = _needsUpgrade(*neuron, mode);
            if (reportProgress)
            {
                scene.reportProgress(progressMessage.str(), 0, neurons.size());
                break;
            }
        }

#pragma omp parallel for
        for (size_t i = 0; i < neurons.size(); ++i)
        {
            Neuron& neuron = *neurons[i];

            if (_needsUpgrade(neuron, mode))
            {
                neuron.upgradeRepresentationMode(mode, scene);
                /* The coloring needs to be reapplied because the new models
                   lack it. */
                neuron.applyColoring(coloring);
                /* The same goes for offsets for simulation mappping. */
                scene.updateSimulationMapping(neuron);
            }
            else
                neuron.setRepresentationMode(mode);

            if (reportProgress)
            {
#pragma omp atomic update
                ++progress;
#ifdef RTNEURON_USE_OPENMP
                if (omp_get_thread_num() == 0 && progress < neurons.size())
#endif
                    scene.reportProgress(progressMessage.str(), progress,
                                         neurons.size());
            }
        }
        if (reportProgress)
            scene.reportProgress(progressMessage.str(), progress,
                                 neurons.size());
    }
    else if (name == "max_visible_branch_order")
    {
        int order = attributes("max_visible_branch_order");
        for (const auto& neuron : neurons)
            neuron->setMaximumVisibleBranchOrder(
                order < 0 ? std::numeric_limits<unsigned int>::max() : order);
    }
}

GIDSet _toGIDSet(const uint32_ts& ids)
{
    GIDSet set;
    set.insert(ids.begin(), ids.end());
    return set;
}

void _checkTargetIsSubset(const GIDSet& target, const Neurons& neurons)
{
    const GIDSet missing = target - neurons;
    if (missing.empty())
        return;

    std::stringstream msg;
    msg << "Query ids not found in source object:";
    size_t count = 0;
    for (const uint32_t id : missing)
    {
        msg << ' ' << id;
        if (++count == 10)
        {
            msg << "...";
            break;
        }
    }
    LBTHROW(std::runtime_error(msg.str()));
}

void _verifyAttributeParameters(const std::string& name,
                                const AttributeMap::AttributeProxy& parameters)
{
    using namespace AttributeMapHelpers;

    if (name == "color_scheme")
    {
        /** \todo Range check */
        getEnum<ColorScheme>(parameters);
    }
    else if (name == "color" || name == "secondary_color" ||
             name == "primary_color")
    {
        osg::Vec4 color;
        getRequiredColor(parameters, color);
    }
    else if (name == "mode")
    {
        getEnum<RepresentationMode>(parameters);
    }
    else if (name == "max_visible_branch_order")
    {
        (void)(int) parameters;
    }
    else if (name == "extra")
    {
        /* (void)(AttributeMapPtr)parameters; doesn't compile */
        AttributeMapPtr extra = parameters;
        (void)extra;
    }
    else if (!name.compare(0, 6, "extra."))
    {
        /* Unverified for simplicity */
    }
    else if (name == "colormaps")
    {
        try
        {
            const AttributeMapPtr maps = parameters;
            (void)maps;
        }
        catch (...)
        {
            throw std::runtime_error(
                "The attribute colormaps must be an AttributeMap");
        }
    }
    else if (!name.compare(0, 10, "colormaps."))
    {
        try
        {
            const ColorMapPtr map = parameters;
            (void)map;
        }
        catch (...)
        {
            throw std::runtime_error("Attribute must be a ColorMap");
        }
    }
    else
        throw std::runtime_error("Unknown or inmmutable attribute: " + name);
}

/**
   Upgrade a display mode based on an incoming parameter.
   Throws if the mode upgrade is not possible.
*/
void _upgradeRestrictedMode(const AttributeMap::AttributeProxy& parameters,
                            RepresentationMode& restrictedMode)
{
    using namespace AttributeMapHelpers;

    const auto mode = getEnum<RepresentationMode>(parameters);

    if (restrictedMode == RepresentationMode::SOMA &&
        mode != RepresentationMode::SOMA &&
        mode != RepresentationMode::NO_DISPLAY)
    {
        restrictedMode = mode;
    }
    else if (restrictedMode == RepresentationMode::NO_AXON &&
             mode == RepresentationMode::WHOLE_NEURON)
    {
        std::stringstream msg;
        std::string str = lexical_cast<std::string>(mode);
        str[0] = std::toupper(str[0]);
        msg << "Cannot swith to representation mode " << str
            << " for neuron target";
        throw std::runtime_error(msg.str());
    }
}
}

struct NeuronObject::Helpers
{
    typedef Scene::_Impl::SubSceneMap SubSceneMap;

    static void applyAttributes(const AttributeMap& attributes,
                                Scene::_Impl& scene, const Neurons& filter,
                                ClientMembers& client, NeuronObject* origin)
    {
        Neurons allNeurons;

        for (auto& s : scene._subScenes)
        {
            SubScene& subScene = *s.second.subScene;
            const auto subset = subScene.getNeurons() & filter;
            allNeurons += subset;

            /* Applying all attributes except the coloring attributes
               on the neurons from this subscene */
            for (const auto& i : attributes)
                _applyAttribute(attributes, i.first, subset, subScene,
                                client._coloring);
        }

        /* Apply all coloring related attributes at once to minimize state
           changes. This is done last to ensure that when the representation
           mode has been upgraded, the new neuron models also get the
           coloring. */
        if (client._coloring.update(attributes) || client._coloringDirty)
        {
            client._coloringDirty = false;

            const ColorMaps colorMaps = client._coloring.getColorMapsForAtlas();
            for (const auto& colorMap : colorMaps)
                scene.addColorMapToAtlas(colorMap);

            for (const auto& neuron : allNeurons)
                neuron->applyColoring(client._coloring);

            if (origin)
            {
                /* If this function is called from a subset handler, ensure
                   that the coloring is udpated next time update
                   is called on the parent object.
                   Doing it here ensures that the client side of the object
                   handlers is consistent with the order in which update
                   operations are queued in the master side. */
                origin->_client._coloringDirty = true;
            }
        }
    }

    static void clearInvalidSubsetPointers(NeuronObject& handler)
    {
        SubSets& subsets = handler._subsets;
        for (auto i = subsets.begin(); i != subsets.end();)
        {
            if (!i->lock())
                i = subsets.erase(i);
            else
                ++i;
        }
    }
};

class NeuronObject::SubSet : public SceneObject
{
public:
    /* Scene operations to make it possible the propagation of object
       handlers to rendering clients and make it possible to change the
       attributes there. */
    class CreateOperation;
    class RemoveOperation;
    class DirtyColoring;
    class ClientHandler;

    SubSet(NeuronObject& origin, const GIDSet& target);

    SubSet(const SubSet& other, const GIDSet& target);

    ~SubSet();

    void cleanup() final
    {
        /* Nothing to do. The neurons are removed from the scene by the
           NeuronObject to which this subset belongs. */
    }

    void onAttributeChangingImpl(
        const AttributeMap&, const std::string& name,
        const AttributeMap::AttributeProxy& parameters) final
    {
        _verifyAttributeParameters(name, parameters);
        /* Ensuring that update will queue an UpdateOperation even when
           the attributes haven't been changed in the original handler. */
        _origin._subsetDirty = true;

        if (name == "mode")
            _upgradeRestrictedMode(parameters, _restrictedMode);
    }

    void onAttributeChangedImpl(const AttributeMap& attributes,
                                const std::string& name) final
    {
        /* Keeping consistency between color and primary_color */
        if (name == "color")
        {
            blockAttributeMapSignals();
            getAttributes().set("primary_color", attributes("color"));
            unblockAttributeMapSignals();
        }
        else if (name == "primary_color")
        {
            blockAttributeMapSignals();
            getAttributes().set("color", attributes("primary_color"));
            unblockAttributeMapSignals();
        }
    }

    rtneuron::Scene::ObjectPtr query(const uint32_ts& ids, const bool checkIds)
    {
        lunchbox::ScopedWrite mutex(_parentLock);

        if (_parent == 0)
            throw std::runtime_error("Invalid object handler.");

        auto target = _toGIDSet(ids);
        if (checkIds)
            _checkTargetIsSubset(target, _neurons);

        SubSetPtr subset(new SubSet(*this, target));

        std::unique_lock<std::mutex> lock(_origin._subsetsMutex);
        _origin._subsets.push_back(SubSetWeakPtr(subset));

        return subset;
    }

    boost::any getObject() const final { return _gidsFromNeurons(_neurons); }
    /* Copy the value of an attribute from an origin attribute map without
       triggering signals. */
    void resetAttribute(const AttributeMap& attributes,
                        const std::string& name);

    /* Copy the value of all attributes from an origin attribute map without
       triggering signals. */
    void resetAttributes(const AttributeMap& attributes);

    bool preUpdateImplementation() final;

    void applyAttribute(const AttributeMap&, const std::string&) final
    {
        /* This function mustn't be called because ClientHandler should be
           used by the update operation. */
        LBUNREACHABLE;
    }

    void dirty()
    {
        /* We assume the coloring has also been tainted */
        _dirtyByOverlap = true;
        _dirtyClientHandlerColoring();
    }

private:
    NeuronObject& _origin;
    Neurons _neurons;

    RepresentationMode _restrictedMode;
    bool _dirtyByOverlap;

    void _dirtyClientHandlerColoring();
};

/** A subset handler for storing subsets inside a Scene::_Impl.

    This object is merely used to make update operations on subsets of
    neuron objects work with multi-node configurations.

    The lifetime of this object is tied to the NeuronObject::SubSet that
    triggers its creation.
*/
class NeuronObject::SubSet::ClientHandler : public ClientSceneObject
{
public:
    ClientHandler(Scene::_Impl* scene, unsigned int id, unsigned int originID,
                  const GIDSet& target)
        : ClientSceneObject(scene, id)
    {
        _origin = scene->getObjectByID(originID);
        if (!_origin)
        {
            LBERROR << "Invalid scene object ID" << std::endl;
            return;
        }
        const auto& origin = static_cast<NeuronObject&>(*_origin);
        _neurons = origin._neurons & target;
    }

    void cleanup() final {}
    /* Sets the dirty flag that will make update() reapply the coloring scheme.

       This is needed when the coloring has been changed by another handler.
       In this case, this handlers's attributes may have not changed, and
       NeuronColoring::update() won't notice that the coloring must be
       reapplied.

       This function is called by DirtyColoring and CreateOperation
    */
    void dirtyColoring() { _client._coloringDirty = true; }
    void update(SubScene&, const AttributeMap&) final {}
    void update(Scene::_Impl& scene, const AttributeMap& attributes) final
    {
        /* Currently all attributes are reapplied when update is called in
           a SubSet because UpdateOperation is not selective on changed
           attributes and current attributes are not tracked here either.
           There are several ways in which this can be avoided, but all
           make the code more complicated. */

        NeuronObject* origin = static_cast<NeuronObject*>(_origin.get());
        Helpers::applyAttributes(attributes, scene, _neurons, _client, origin);

        if (origin)
        {
            /* Ensuring that the attributes that have changed in the subset
               will be reapplied the next time update is called on the
               parent object.
               Doing it here ensures that the client side of the object
               handlers is consistent with the order in which update
               operations are queued in the master side. */
            for (const auto& attribute : attributes)
                origin->_current.unset(attribute.first);
        }
    }

private:
    Neurons _neurons;
    ClientMembers _client;
    Scene::_Impl::BaseObjectPtr _origin;
};

#define INSTANTIATE_OPERATION_SERIALIZATION(ClassName)           \
    template void ClassName::serialize<net::DataOStreamArchive>( \
        net::DataOStreamArchive&, const unsigned int);           \
    template void ClassName::serialize<net::DataIStreamArchive>( \
        net::DataIStreamArchive&, const unsigned int);           \
    }                                                            \
    }                                                            \
    }                                                            \
    BOOST_CLASS_EXPORT(bbp::rtneuron::core::ClassName)           \
    namespace bbp                                                \
    {                                                            \
    namespace rtneuron                                           \
    {                                                            \
    namespace core                                               \
    {
class NeuronObject::SubSet::CreateOperation : public SceneOperation
{
public:
    /* Default constructor needed for serialization */
    CreateOperation()
        : _objectID(-1)
        , _originID(-1)
    {
    }

    CreateOperation(const SubSet& subset)
        : _objectID(subset.getID())
        , _originID(subset._origin.getID())
        , _target(subset._neurons)
        , _isDirty(subset._dirtyByOverlap)
    {
    }

    void operator()(SubScene&) const {}
    void operator()(Scene::_Impl& scene) const
    {
        auto handler = std::make_shared<ClientHandler>(&scene, _objectID,
                                                       _originID, _target);
        if (_isDirty)
            handler->dirtyColoring();
        scene.insertObject(handler);
    }

    template <class Archive>
    void load(Archive& archive, const unsigned int /* version */)
    {
        archive& _objectID;
        archive& _originID;
        size_t size = 0;
        archive& size;
        _target.clear();
        for (size_t i = 0; i != size; ++i)
        {
            uint32_t id;
            archive >> id;
            _target.insert(id);
        }
        archive& _isDirty;
    }

    template <class Archive>
    void save(Archive& archive, const unsigned int /* version */) const
    {
        archive& _objectID;
        archive& _originID;
        const size_t size = _target.size();
        archive& size;
        for (const uint32_t id : _target)
            archive& id;
        archive& _isDirty;
    }

    template <class Archive>
    void serialize(Archive& archive, const unsigned int version)
    {
        archive& boost::serialization::base_object<SceneOperation>(*this);
        boost::serialization::split_member(archive, *this, version);
    }

    unsigned int _objectID;
    unsigned int _originID;
    GIDSet _target;
    bool _isDirty;
};
INSTANTIATE_OPERATION_SERIALIZATION(NeuronObject::SubSet::CreateOperation)

class NeuronObject::SubSet::RemoveOperation : public SceneOperation
{
public:
    /* Default constructor needed for serialization */
    RemoveOperation()
        /* This operation needs to be run after subscene updates. Otherwise,
           when an operation queue contains a create-update-remove sequence
           the update operation will fail in the subscenes because the object
           has already been removed from the scene where the queue runs
           first. */
        : SceneOperation(Order::postUpdate),
          _objectID(-1)
    {
    }

    RemoveOperation(const unsigned int id)
        : _objectID(id)
    {
    }

    void operator()(SubScene&) const {}
    void operator()(Scene::_Impl& scene) const
    {
        scene.removeObject(_objectID);
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
        ar& _objectID;
    }

    unsigned int _objectID;
    AttributeMap _attributes;
};
INSTANTIATE_OPERATION_SERIALIZATION(NeuronObject::SubSet::RemoveOperation)

class NeuronObject::SubSet::DirtyColoring : public SceneOperation
{
public:
    /* Default constructor needed for serialization */
    DirtyColoring()
        : _objectID(-1)
    {
    }

    DirtyColoring(const SubSet& subset)
        : _objectID(subset.getID())
    {
    }

    void operator()(SubScene&) const {}
    void operator()(Scene::_Impl& scene) const
    {
        Scene::_Impl::BaseObjectPtr object = scene.getObjectByID(_objectID);
        if (object)
            static_cast<ClientHandler&>(*object).dirtyColoring();
    }

    template <class Archive>
    void serialize(Archive& archive, const unsigned int)
    {
        archive& boost::serialization::base_object<SceneOperation>(*this);
        archive& _objectID;
    }

    unsigned int _objectID;
};
INSTANTIATE_OPERATION_SERIALIZATION(NeuronObject::SubSet::DirtyColoring)

NeuronObject::SubSet::SubSet(NeuronObject& origin, const GIDSet& target)
    : SceneObject(origin._parent, origin.getAttributes())
    , _origin(origin)
    , _neurons(_origin._neurons & target)
    , _restrictedMode(origin._restrictedMode)
    /* A subset handler can be already dirty at creation. This occurs when
       it's created from a NeuronObject that has been tainted by another
       subset. The safest in this case is to assume that this target may
       have been touched by the other subset. */
    , _dirtyByOverlap(origin._subsetDirty)
{
    _parent->pushOperation(SceneOperationPtr(new CreateOperation(*this)));
}

NeuronObject::SubSet::SubSet(const SubSet& other, const GIDSet& target)
    : SceneObject(other._parent, other.getAttributes())
    , _origin(other._origin)
    , _neurons(other._neurons & target)
    , _restrictedMode(other._restrictedMode)
    , _dirtyByOverlap(other._dirtyByOverlap)
{
    _parent->pushOperation(SceneOperationPtr(new CreateOperation(*this)));
}

NeuronObject::SubSet::~SubSet()
{
    lunchbox::ScopedWrite mutex(_parentLock);

    AttributeMapPtr colormaps = getAttributes()("colormaps", AttributeMapPtr());
    if (colormaps && !colormaps->empty())
    {
        LBWARN
            << "Deleting a temporary neuron object handler with"
               " one or more colormaps assigned has undefined results in the"
               " coloring of somas. If you really want to delete this object"
               " assign an empty AttributeMap to the colormaps attribute to get"
               " rid of this message"
            << std::endl;
    }

    if (_parent)
        _parent->pushOperation(SceneOperationPtr(new RemoveOperation(getID())));
}

void NeuronObject::SubSet::resetAttribute(const AttributeMap& attributes,
                                          const std::string& name)
{
    blockAttributeMapSignals();
    getAttributes().set(name, attributes(name));
    _currentHash = getAttributes().hash();
    /* For simplicity the neuron coloring is always considered tainted
       at origin */
    _dirtyClientHandlerColoring();
    unblockAttributeMapSignals();
}

void NeuronObject::SubSet::resetAttributes(const AttributeMap& attributes)
{
    blockAttributeMapSignals();
    getAttributes().merge(attributes);
    _currentHash = getAttributes().hash();
    /* For simplicity the neuron coloring is always considered tainted
       at origin */
    _dirtyClientHandlerColoring();
    unblockAttributeMapSignals();
}

bool NeuronObject::SubSet::preUpdateImplementation()
{
    if (_currentHash == getAttributes().hash() && !_dirtyByOverlap)
        return false;

    /* If attributes haven't really changed but we reach this point,
        the neurons have been updated on other subset (which set _dirtyByOverlap
        to true). Neurons are going to be updated by this subset, so now we
        have to make sure than update will work for the overlapping subets
        afterwards. */
    std::unique_lock<std::mutex> lock(_origin._subsetsMutex);
    Helpers::clearInvalidSubsetPointers(_origin);
    for (const auto& i : _origin._subsets)
    {
        SubSet& other = *i.lock();
        if (&other != this && !(other._neurons & _neurons).empty())
            other.dirty(); /* The subsets overlap. */
    }

    if (_dirtyByOverlap)
    {
        _dirtyByOverlap = false;
        return true;
    }
    return false;
}

void NeuronObject::SubSet::_dirtyClientHandlerColoring()
{
    _origin._parent->pushOperation(
        SceneOperationPtr(new SubSet::DirtyColoring(*this)));
}

// NeuronObject ----------------------------------------------------------------

NeuronObject::NeuronObject(Neurons&& neurons, const AttributeMap& attributes,
                           Scene::_Impl* parent)
    : SceneObject(parent, attributes)
    , _neurons(std::move(neurons))
    , _restrictedMode(
          AttributeMapHelpers::getEnum(attributes, "mode",
                                       RepresentationMode::WHOLE_NEURON))
    , _subsetDirty(false)
    , _client(attributes)
{
    Neurons intersection;
    if (!parent->_neurons.empty() &&
        !(intersection = neurons & parent->_neurons).empty())
    {
        std::stringstream error;
        error << "Trying to insert " << intersection.size() << " neuron"
              << (intersection.size() > 1 ? "s" : "")
              << " already present in this scene:";
        size_t i = 0;
        auto n = intersection.begin();
        for (; n != intersection.end() && i < 10; ++n, ++i)
            error << ' ' << (*n)->getGID();
        if (n != intersection.end())
            error << "...";
        throw std::runtime_error(error.str());
    }

    /* Assigning default values to some attributes if they don't have any
       already. This way they can be queried from Python */
    blockAttributeMapSignals();
    NeuronColoring::assignDefaults(getAttributes());
    ColorScheme dummy;
    if (!AttributeMapHelpers::getEnum(getAttributes(), "mode", dummy))
        /* The attribute is stored as integer for simplicity. Otherwise the
           enum type needs to be registered for AttributeMap::hash to work. */
        getAttributes().set("mode", int(RepresentationMode::WHOLE_NEURON));
    unblockAttributeMapSignals();

    parent->_neurons += neurons;
    parent->_somasBound.init();
}

NeuronObject::~NeuronObject()
{
    _invalidateQueriedHandlers();
}

bool NeuronObject::preUpdateImplementation()
{
    if (!_subsetDirty)
        return false;

    _subsetDirty = false;

    std::unique_lock<std::mutex> lock(_subsetsMutex);

    Helpers::clearInvalidSubsetPointers(*this);

    /* Reset the attributes on all the subsets */
    for (const auto& i : _subsets)
    {
        SubSetPtr subset = i.lock();
        subset->resetAttributes(getAttributes());
    }
    return true;
}

boost::any NeuronObject::getObject() const
{
    return _gidsFromNeurons(_neurons);
}

void NeuronObject::cleanup()
{
    lunchbox::ScopedWrite mutex(_parentLock);
    _parent->_neurons = _parent->_neurons - _neurons;
    _parent->_somasBound.init();
}

void NeuronObject::onAttributeChangingImpl(
    const AttributeMap&, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    _verifyAttributeParameters(name, parameters);

    if (name == "mode")
        _upgradeRestrictedMode(parameters, _restrictedMode);
}

void NeuronObject::onAttributeChangedImpl(const AttributeMap& attributes,
                                          const std::string& name)
{
    /* Ensuring consistency of attributes */
    if (name == "color")
    {
        blockAttributeMapSignals();
        getAttributes().set("primary_color", attributes("color"));
        unblockAttributeMapSignals();
    }
    else if (name == "primary_color")
    {
        blockAttributeMapSignals();
        getAttributes().set("color", attributes("primary_color"));
        unblockAttributeMapSignals();
    }

    std::unique_lock<std::mutex> lock(_subsetsMutex);

    Helpers::clearInvalidSubsetPointers(*this);
    for (const auto& i : _subsets)
    {
        SubSetPtr subset = i.lock();
        subset->resetAttribute(attributes, name);
    }
}

void NeuronObject::update(Scene::_Impl& scene, const AttributeMap& attributes)
{
    /* The base implementation is used for all attributes which are not
       coloring related . */
    SceneObject::update(scene, attributes);

    /* Dealing with coloring attributes from here because reparing a coloring
       scheme touched by a subset handler needs access to
       _client._coloringDirty.
       This is done after applying the other attributes to ensure that if the
       representation mode has been upgraded, the new models are given the
       new coloring. */
    Neurons neurons;
    /* This occurs during node sync, so it's safe to search the neurons
       in the scene */
    for (const auto& s : _parent->_subScenes)
    {
        /* Subscene neurons go first because they are those that go in the
           result and get the attribute changes applied. */
        neurons += s.second.subScene->getNeurons() & _neurons;
    }

    if (_client._coloring.update(attributes) || _client._coloringDirty)
    {
        for (const auto& neuron : neurons)
            neuron->applyColoring(_client._coloring);
        _client._coloringDirty = false;
    }
}

void NeuronObject::dirtySubsets()
{
    Helpers::clearInvalidSubsetPointers(*this);
    for (const auto& i : _subsets)
        i.lock()->dirty();
}

void NeuronObject::applyAttribute(const AttributeMap& attributes,
                                  const std::string& name)
{
    for (const auto& i : _parent->_subScenes)
    {
        SubScene& subScene = *i.second.subScene;
        /* Subscene neurons go first because they are those that go in the
           result and get the attribute changes applied. */
        const auto& subset = subScene.getNeurons() & _neurons;
        _applyAttribute(attributes, name, subset, subScene, _client._coloring);
    }
}

void NeuronObject::invalidateImplementation()
{
    std::unique_lock<std::mutex> lock(_subsetsMutex);
    _invalidateQueriedHandlers();
}

rtneuron::Scene::ObjectPtr NeuronObject::query(const uint32_ts& ids,
                                               const bool checkIds)
{
    std::unique_lock<std::mutex> lock(_subsetsMutex);
    lunchbox::ScopedWrite mutex(_parentLock);

    if (_parent == 0)
        throw std::runtime_error("Invalid object handler.");

    Helpers::clearInvalidSubsetPointers(*this);

    const auto target = _toGIDSet(ids);
    if (checkIds)
        _checkTargetIsSubset(target, _neurons);

    SubSetPtr subset(new SubSet(*this, target));
    _subsets.push_back(subset);
    return subset;
}

void NeuronObject::_invalidateQueriedHandlers()
{
    for (const auto& i : _subsets)
    {
        SubSetPtr subset = i.lock();
        if (subset)
            subset->invalidate();
    }
}
}
}
}
