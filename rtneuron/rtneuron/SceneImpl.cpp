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

#include "SceneImpl.h"
#include "scene/GeometryObject.h"
#include "scene/NeuronObject.h"
#include "scene/SceneOperations.h"
#include "scene/SubScene.h"
#include "scene/SynapseObject.h"

#include "config/Globals.h"
#include "config/constants.h"
#include "data/SimulationDataMapper.h"
#include "data/SpikeReport.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "render/CameraData.h"
#include "render/ColorMapAtlas.h"
#include "render/RenderBinManager.h"
#include "render/SceneStyle.h"
#include "scene/CircuitScene.h"
#include "util/attributeMapHelpers.h"
#include "util/log.h"
#include "util/transformString.h"
#include "util/vec_to_vec.h"

#define EQ_IGNORE_GLEW
#include "viewer/osgEq/Config.h"
#include "viewer/osgEq/FrameData.h"

#include <brain/circuit.h>
#include <brain/synapse.h>
#include <brain/synapses.h>
#include <brain/synapsesIterator.h>

#include <brion/blueConfig.h>

#include <lunchbox/refPtr.h>

#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

namespace eq
{
namespace fabric
{
bool operator<(const Range& a, const Range& b)
{
    return a.start < b.start || (a.start == b.start && a.end < b.end);
}
}
}

namespace bbp
{
namespace rtneuron
{
using namespace core;
using core::vec_to_vec;

/*
  Helper functions
*/
namespace
{
Vector4f _getSomaSphere(const Neuron& neuron)
{
    return Vector4f(vec_to_vec(neuron.getPosition()),
                    neuron.getSomaRenderRadius());
}

osg::BoundingSphere _calculateBoundingSphere(
    const std::vector<osg::Vec3>& positions)
{
    if (positions.empty())
        return osg::BoundingSphere();

    osg::BoundingBox box;
    for (const auto& position : positions)
        box.expandBy(position);

    const osg::Vec3& center = box.center();
    float maxDistance = 0;
    for (const auto& position : positions)
        maxDistance = std::max(maxDistance, (position - center).length2());

    return osg::BoundingSphere(center, std::sqrt(maxDistance));
}

bool _isSceneStyleAttribute(const std::string& name)
{
    std::set<std::string> s_names;
    if (s_names.empty())
    {
        s_names.insert("alpha_blending");
        s_names.insert("em_shading");
        s_names.insert("inflatable_neurons");
        s_names.insert("show_spikes");
        s_names.insert("show_soma_spikes");
    }
    return s_names.find(name) != s_names.end();
}
}

/*
  Helper classes
*/

#define SCENE_OPERATION_SERIALIZERS(operation)                                 \
    template void Scene::_Impl::operation::serialize<net::DataOStreamArchive>( \
        net::DataOStreamArchive&, const unsigned int);                         \
    template void Scene::_Impl::operation::serialize<net::DataIStreamArchive>( \
        net::DataIStreamArchive&, const unsigned int);

class Scene::_Impl::HighlightCells : public SceneOperation
{
public:
    HighlightCells() {}
    HighlightCells(const uint32_t gid, const bool highlight)
        : _gids({gid})
        , _highlight(highlight)
    {
    }

    HighlightCells(const uint32_ts& gids, const bool highlight)
        : _gids(gids)
        , _highlight(highlight)
    {
    }

    void operator()(SubScene& subScene) const final
    {
        subScene.highlightCells(GIDSet(_gids.begin(), _gids.end()), _highlight);
    }

    void operator()(rtneuron::Scene::_Impl&) const final {}
private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
        ar& _gids;
        ar& _highlight;
    }

    uint32_ts _gids;
    bool _highlight;
};
SCENE_OPERATION_SERIALIZERS(HighlightCells)

class Scene::_Impl::Clear : public SceneOperation
{
    void operator()(SubScene&) const final {}
    void operator()(rtneuron::Scene::_Impl& scene) const final
    {
        scene._clearSubScenes();
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
    }
};
SCENE_OPERATION_SERIALIZERS(Clear);

class Scene::_Impl::UpdateClipPlane : public SceneOperation
{
public:
    UpdateClipPlane() {}
    UpdateClipPlane(const unsigned int index, const Vector4f& plane)
        : _index(index)
        , _plane(plane)
    {
    }

    void operator()(SubScene& subscene) const final
    {
        subscene.setClipPlane(_index, vec_to_vec(_plane));
    }

    void operator()(rtneuron::Scene::_Impl&) const final {}
private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
        ar& _index;
        ar& _plane[0] & _plane[1] & _plane[2] & _plane[3];
    }

    unsigned int _index;
    Vector4f _plane;
};
SCENE_OPERATION_SERIALIZERS(UpdateClipPlane)

class Scene::_Impl::ClearClipPlanes : public SceneOperation
{
public:
    ClearClipPlanes() {}
    void operator()(SubScene& subscene) const final
    {
        subscene.clearClipPlanes();
    }

    void operator()(rtneuron::Scene::_Impl&) const final {}
private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
    }
};
SCENE_OPERATION_SERIALIZERS(ClearClipPlanes)

class Scene::_Impl::StyleAttributeUpdate : public SceneOperation
{
public:
    StyleAttributeUpdate() {}
    StyleAttributeUpdate(const AttributeMap& attributes)
        : _attributes(attributes)
    {
    }

    virtual void operator()(SubScene& subscene) const
    {
        const SceneStyle& current = subscene.getStyle();
        AttributeMap desired(current.getAttributes());
        desired.merge(_attributes);

        if (desired != current.getAttributes())
        {
            SceneStylePtr style(new SceneStyle(desired));
            style->setColorMapAtlas(subscene.getScene()._colorMaps);
            subscene.applyStyle(style);
        }
    }

    virtual void operator()(rtneuron::Scene::_Impl& scene) const
    {
        AttributeMap desired(scene._sceneStyle->getAttributes());
        desired.merge(_attributes);

        /* Check if there's no scene style or needs updating. */
        if (desired != scene._sceneStyle->getAttributes())
        {
            SceneStylePtr style(new SceneStyle(desired));
            style->setColorMapAtlas(scene._colorMaps);
            scene._sceneStyle = style;
            for (const auto& i : scene._geometryObjects)
                i.second->applyStyle(style);
            for (const auto& i : scene._modelObjects)
                i.second->applyStyle(style);
        }
    }

private:
    AttributeMap _attributes;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
        ar& _attributes;
    }
};
SCENE_OPERATION_SERIALIZERS(StyleAttributeUpdate)

// Scene::_Impl::Object --------------------------------------------------------

unsigned int _nextObjectID()
{
    static unsigned int id = 0;
    return id++;
}

Scene::_Impl::BaseObject::BaseObject(_Impl* parent)
    : _parent(parent)
    , _id(_nextObjectID())
{
}

Scene::_Impl::BaseObject::BaseObject(_Impl* parent, unsigned int id)
    : _parent(parent)
    , _id(id)
{
}

void Scene::_Impl::BaseObject::invalidate()
{
    lunchbox::ScopedWrite mutex(_parentLock);
    _parent = 0;
    invalidateImplementation();
}

// Scene::_Impl ---------------------------------------------------------------

/*
  Constructors/destructor
*/

Scene::_Impl::_Impl(const AttributeMap& attributes, rtneuron::Scene* parent)
    : Configurable(attributes)
    , _parentPointerMonitor(0)
    , _parent(parent)
    , _config(0)
    , _masterSceneOperations(new SceneOperations)
    , _sceneOperations(new SceneOperations)
    , _subscenesNeedReset(false)
    , _extraModels(new osg::Group())
    , _sceneAttributes(new CircuitSceneAttributes(attributes))
    , _sceneStyle(new SceneStyle(attributes))
    , _colorMaps(new ColorMapAtlas)
    , _version(0)
{
    /* Creating the circuit if the attribute has been assigned. */
    const std::string circuitURI = attributes("circuit", std::string());
    if (!circuitURI.empty())
        setCircuit(CircuitPtr(new brain::Circuit(brain::URI(circuitURI))));

    /* We don't want this attribute to appear later on. */
    getAttributes().unset("circuit");

    getAttributes().set("auto_update", (bool)attributes("auto_update", true));

    _sceneStyle->setColorMapAtlas(_colorMaps);

    static unsigned int s_nextID = 0;
    static std::mutex s_lock;
    lunchbox::ScopedWrite mutex(s_lock);
    _id = s_nextID++;
    LBASSERT(s_nextID != 0); // Checking an unprobable overflow.
}

Scene::_Impl::~_Impl()
{
    lunchbox::ScopedWrite mutex(_lock); /* Mutex really needed? */
    if (_config)
    {
        _unmapDistributedObjects();
        if (_masterSceneOperations->isAttached())
            _config->deregisterObject(_masterSceneOperations);
        _config = 0;
    }
    for (const auto& n : _neuronObjects)
        n.second->invalidate();

    delete _masterSceneOperations;
    delete _sceneOperations;
}

/*
  Member functions
*/

Scene::ObjectPtr Scene::_Impl::addNeurons(const GIDSet& gids,
                                          const AttributeMap& attributes)
{
    if (!_circuit)
        throw std::runtime_error(
            "Cannot add neurons to a scene without an assigned circuit");

    const bool autoUpdate = getAttributes()("auto_update");
    NeuronObjectPtr handler;
    {
        lunchbox::ScopedWrite mutex(_lock);
        handler.reset(
            new NeuronObject(Neurons(gids, _circuit), attributes, this));
        _neuronObjects.insert(std::make_pair(handler->getID(), handler));
        ++_version;
    }
    _neurons += handler->getNeurons();

    if (autoUpdate)
        dirty(true);
    return handler;
}

Scene::ObjectPtr Scene::_Impl::addEfferentSynapses(
    const brain::Synapses& synapses, const AttributeMap& attributes)
{
    if (!_circuit)
        throw std::runtime_error(
            "Cannot add synapses to a scene without an assigned circuit");

    const bool autoUpdate = getAttributes()("auto_update");
    SynapseObjectPtr handler;
    {
        lunchbox::ScopedWrite mutex(_lock);
        handler.reset(new SynapseObject(synapses, SynapseObject::EFFERENT,
                                        attributes, this));
        _synapseObjects.insert(std::make_pair(handler->getID(), handler));
        ++_version;
    }
    if (autoUpdate)
        dirty(true);
    return handler;
}

Scene::ObjectPtr Scene::_Impl::addAfferentSynapses(
    const brain::Synapses& synapses, const AttributeMap& attributes)
{
    if (!_circuit)
        throw std::runtime_error(
            "Cannot add synapses to a scene without an assigned circuit");

    const bool autoUpdate = getAttributes()("auto_update");
    SynapseObjectPtr handler;
    {
        lunchbox::ScopedWrite mutex(_lock);
        handler.reset(new SynapseObject(synapses, SynapseObject::AFFERENT,
                                        attributes, this));
        _synapseObjects.insert(std::make_pair(handler->getID(), handler));
        ++_version;
    }
    if (autoUpdate)
        dirty(true);
    return handler;
}

Scene::ObjectPtr Scene::_Impl::addModel(const char* filename,
                                        const Matrix4f& transform,
                                        const AttributeMap& attributes)
{
    const bool autoUpdate = getAttributes()("auto_update");
    osg::Matrix osgTransform;
    /* OSG and vmml use transposed notations */
    for (unsigned int i = 0; i < 4; ++i)
        for (unsigned int j = 0; j < 4; ++j)
            osgTransform(j, i) = transform(j, i);

    ModelObjectPtr handler(
        new ModelObject(filename, osgTransform, attributes, this));

    {
        lunchbox::ScopedWrite mutex(_lock);
        ModelObjects::value_type pair(handler->getID(), handler);
        _modelObjects.insert(pair);
        _toAdd.insert(pair);
        ++_version;
    }
    if (autoUpdate)
        dirty(true);
    return handler;
}

Scene::ObjectPtr Scene::_Impl::addModel(const char* filename,
                                        const char* transform,
                                        const AttributeMap& attributes)
{
    const bool autoUpdate = getAttributes()("auto_update");
    ModelObjectPtr handler(
        new ModelObject(filename, parseTranslateRotateScaleString(transform),
                        attributes, this));

    {
        lunchbox::ScopedWrite mutex(_lock);
        ModelObjects::value_type pair(handler->getID(), handler);
        _modelObjects.insert(pair);
        _toAdd.insert(pair);
        ++_version;
    }
    if (autoUpdate)
        dirty(true);
    return handler;
}

Scene::ObjectPtr Scene::_Impl::addGeometry(
    const osg::ref_ptr<osg::Array>& vertices,
    const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
    const osg::ref_ptr<osg::Vec4Array>& colors,
    const osg::ref_ptr<osg::Vec3Array>& normals, const AttributeMap& attributes)
{
    const bool autoUpdate = getAttributes()("auto_update");
    GeometryObjectPtr handler(new GeometryObject(vertices, primitive, colors,
                                                 normals, attributes, this));
    {
        lunchbox::ScopedWrite mutex(_lock);
        auto pair = std::make_pair(handler->getID(), handler);
        _geometryObjects.insert(pair);
        _toAdd.insert(pair);
        ++_version;
    }
    if (autoUpdate)
        dirty(true);
    return handler;
}

void Scene::_Impl::update()
{
    {
        lunchbox::ScopedWrite mutex(_lock);

        if (!_styleAttributeUpdates.empty())
        {
            auto operation =
                std::make_shared<StyleAttributeUpdate>(_styleAttributeUpdates);
            {
                lunchbox::ScopedWrite opMutex(_operationsLock);
                _masterSceneOperations->push(operation);
            }
            _styleAttributeUpdates.clear();
        }
        ++_version;
    }
    dirty(true);
}

std::vector<Scene::ObjectPtr> Scene::_Impl::getObjects()
{
    std::vector<rtneuron::Scene::ObjectPtr> objects;
    for (const auto& i : _neuronObjects)
        objects.push_back(i.second);
    for (const auto& i : _synapseObjects)
        objects.push_back(i.second);
    for (const auto& i : _modelObjects)
        objects.push_back(i.second);
    for (const auto& i : _geometryObjects)
        objects.push_back(i.second);
    return objects;
}

CircuitPtr Scene::_Impl::getCircuit() const
{
    return _circuit;
}

void Scene::_Impl::setCircuit(const CircuitPtr& circuit)
{
    if (!_neuronObjects.empty() || !_synapseObjects.empty())
        throw std::runtime_error(
            "Cannot change the circuit of a scene "
            "with neurons or synapses added");
    _circuit = circuit;
    /* Finding out the mesh path */
    brion::BlueConfig config(circuit->getSource().getPath());
    brion::Strings runs = config.getSectionNames(brion::CONFIGSECTION_RUN);
    _sceneAttributes->circuitMeshPath =
        config.get(brion::CONFIGSECTION_RUN, runs[0], "MeshPath");
}

template <typename T>
Scene::_Impl::BaseObjectPtr _findObject(const std::map<unsigned int, T>& map,
                                        const unsigned int id)
{
    const typename std::map<unsigned int, T>::const_iterator i = map.find(id);
    if (i == map.end())
        return Scene::_Impl::BaseObjectPtr();
    return i->second;
}

Scene::_Impl::BaseObjectPtr Scene::_Impl::getObjectByID(
    const unsigned int id) const
{
    BaseObjectPtr object;
    if ((object = _findObject(_neuronObjects, id)))
        return object;
    else if ((object = _findObject(_synapseObjects, id)))
        return object;
    else if ((object = _findObject(_geometryObjects, id)))
        return object;
    else if ((object = _findObject(_modelObjects, id)))
        return object;
    else if ((object = _findObject(_clientObjects, id)))
        return object;
    return BaseObjectPtr();
}

template <typename T>
struct Scene::_Impl::Remover
{
    Remover(T& map)
        : _map(map)
    {
    }

    bool operator()(const unsigned int id)
    {
        typename T::iterator i = _map.find(id);
        if (i == _map.end())
            return false;
        i->second->cleanup();
        i->second->invalidate();
        _map.erase(i);
        return true;
    }

private:
    T& _map;
};

void Scene::_Impl::remove(const rtneuron::Scene::ObjectPtr& object)
{
    lunchbox::ScopedWrite mutex(_lock);

    const BaseObject* sceneObject =
        dynamic_cast<const BaseObject*>(object.get());
    assert(sceneObject);

    if (_removeObject(sceneObject->getID()) && getAttributes()("auto_update"))
    {
        mutex.unlock();
        dirty(false);
    }
}

void Scene::_Impl::removeObject(const unsigned int id)
{
    lunchbox::ScopedWrite mutex(_lock);

    if (_removeObject(id) && getAttributes()("auto_update"))
        dirty(false);
}

void Scene::_Impl::insertObject(const BaseObjectPtr& handler)
{
    _clientObjects[handler->getID()] = handler;
}

void Scene::_Impl::clear()
{
    lunchbox::ScopedWrite mutex(_lock);
    _neuronObjects.clear();
    _synapseObjects.clear();
    _geometryObjects.clear();
    _modelObjects.clear();
    _clientObjects.clear();
    _userClipPlanes.clear();

    _neurons.clear();
    _extraModels->removeChildren(0, _extraModels->getNumChildren());
    _toAdd.clear();

    _somasBound.init();
    _synapsesBound.init();

    _compartmentReport.reset();
    _spikeReport.reset();

    if (_config)
    {
        lunchbox::ScopedWrite opMutex(_operationsLock);
        /* Throwing away any pending operations. */
        _masterSceneOperations->clear();
        /* And clear out subscenes. */
        _masterSceneOperations->push(std::make_shared<Clear>());
        _masterSceneOperations->push(std::make_shared<ClearClipPlanes>());
    }

    const bool autoUpdate = getAttributes()("auto_update");
    if (autoUpdate)
    {
        mutex.unlock();
        dirty(false);
    }
}

void Scene::_Impl::highlight(const GIDSet& target, const bool on)
{
    {
        lunchbox::ScopedWrite mutex(_lock);
        uint32_ts cells;
        const auto subset = target & _neurons;
        for (const auto gid : subset)
        {
            cells.push_back(gid);
            if (on)
                _highlightedNeurons.insert(gid);
            else
                _highlightedNeurons.erase(gid);
        }
        lunchbox::ScopedWrite opMutex(_operationsLock);
        _masterSceneOperations->push(
            SceneOperationPtr(new HighlightCells(cells, on)));
    }
    dirty(false);
}

void Scene::_Impl::pick(const Rayf& ray) const
{
    float nearest = std::numeric_limits<float>::max();
    enum Hit
    {
        NONE,
        NEURON,
        SYNAPSE
    } hit = NONE;

    uint32_t gid = uint32_t();
    uint16_t section = uint16_t();
    const auto& selectable = _maskedNeurons.empty()
                                 ? _neurons
                                 : _neurons - (_neurons & _maskedNeurons);
    for (const auto& neuron : selectable)
    {
        const float distance = ray.test(_getSomaSphere(*neuron));
        if (distance > 0 && distance < nearest)
        {
            nearest = distance;
            hit = NEURON;
            gid = neuron->getGID();
            /* Assume section 0 is always the soma. Is this something that
               could change? */
            section = 0;
        }
    }

    brain::Synapses::const_iterator synapse;
    for (const auto& obj : _synapseObjects)
    {
        const auto& handler = *obj.second;
        const auto& synapses = handler.getSynapses();
        const auto& centers = handler.getLocations();
        /* The radius parameter can come from Python so we have to convert
           to double. */
        const double radius = handler.getAttributes()("radius");

        auto s = synapses.begin();
        for (size_t i = 0; i != synapses.size(); ++i, ++s)
        {
            const float distance = ray.test(Vector4f(centers[i], radius));
            if (distance > 0 && distance < nearest)
            {
                nearest = distance;
                hit = SYNAPSE;
                synapse = s;
            }
        }
    }
    switch (hit)
    {
    case NEURON:
        _parent->cellSelected(gid, section, 0);
        break;
    case SYNAPSE:
        _parent->synapseSelected(*synapse);
        break;
    default:;
    }
}

void Scene::_Impl::pick(const Planes& planes) const
{
    std::vector<Vector4f> somas;
    somas.reserve(_neurons.size() - _maskedNeurons.size());

    const auto& selectable = _maskedNeurons.empty()
                                 ? _neurons
                                 : _neurons - (_neurons & _maskedNeurons);
    for (const auto& neuron : selectable)
        somas.push_back(_getSomaSphere(*neuron));

    std::vector<char> selected(somas.size(), true);
#pragma omp parallel for
    for (size_t i = 0; i < somas.size(); ++i)
    {
        for (const auto& plane : planes)
        {
            const Vector3f center((const float*)&somas[i][0]);
            const float radius = somas[i][3];
            if (plane.dot(Vector4f(center, 1)) > radius)
            {
                selected[i] = false;
                break;
            }
        }
    }

    size_t i = 0;
    GIDSet target;
    for (const auto& neuron : selectable)
    {
        if (selected[i])
            target.insert(neuron->getGID());
        ++i;
    }

    if (!target.empty())
        _parent->cellSetSelected(target);
}

void Scene::_Impl::setNeuronSelectionMask(const GIDSet& target)
{
    _maskedNeurons = target;
}

const GIDSet& Scene::_Impl::getNeuronSelectionMask() const
{
    return _maskedNeurons;
}

const GIDSet& Scene::_Impl::getHighlightedNeurons() const
{
    return _highlightedNeurons;
}

const osg::BoundingSphere& Scene::_Impl::getSomasBoundingSphere() const
{
    lunchbox::ScopedWrite mutex(_lock);
    if (!_somasBound.valid() && !_neuronObjects.empty())
        _calculateSomasBoundingSphere();
    return _somasBound;
}

const osg::BoundingSphere& Scene::_Impl::getSynapsesBoundingSphere() const
{
    lunchbox::ScopedWrite mutex(_lock);
    if (!_synapsesBound.valid() && !_synapseObjects.empty())
        _calculateSynapsesBoundingSphere();
    return _synapsesBound;
}

osg::BoundingSphere Scene::_Impl::getExtraModelsBoundingSphere() const
{
    /* Extra models is only updated in nodeSync */
    osg::BoundingSphere sphere;
    for (const auto& m : _modelObjects)
        sphere.expandBy(m.second->getNode()->getBound());
    for (const auto& m : _geometryObjects)
        sphere.expandBy(m.second->getNode()->getBound());
    return sphere;
}

void Scene::_Impl::setClipPlane(const unsigned int index, const Vector4f& plane)
{
    if (index >= 8)
        std::runtime_error("Invalid plane index.");

    /* Checking if the plane has changed. */
    if (_userClipPlanes.size() <= index)
        _userClipPlanes.resize(index + 1);
    if (_userClipPlanes[index].first == plane)
        return;

    /* Now the plane is valid. */
    _userClipPlanes[index].second = true;
    _userClipPlanes[index].first = plane;

    /* Submitting the scene operation and dirtying the scene. */
    SceneOperationPtr operation(new UpdateClipPlane(index, plane));
    {
        lunchbox::ScopedWrite mutex(_operationsLock);
        _masterSceneOperations->push(operation);
    }
    dirty(true);
}

const Vector4f& Scene::_Impl::getClipPlane(unsigned int index) const
{
    if (_userClipPlanes.size() <= index || !_userClipPlanes[index].second)
        throw std::runtime_error("Unexistent clipping plane");
    return _userClipPlanes[index].first;
}

void Scene::_Impl::clearClipPlanes()
{
    SceneOperationPtr operation(new ClearClipPlanes());
    {
        lunchbox::ScopedWrite mutex(_operationsLock);
        _masterSceneOperations->push(operation);
    }
    _userClipPlanes.clear();
    dirty(true);
}

void Scene::_Impl::setSimulation(const CompartmentReportPtr& report)
{
    {
        lunchbox::ScopedWrite mutex(_lock);

        for (const auto& i : _subScenes)
            i.second.subScene->setSimulation(report);

        _compartmentReport = report;
    }
    getAttributes().set("show_soma_spikes", false);
    dirty(false);
}

void Scene::_Impl::setSimulation(const SpikeReportReaderPtr& report)
{
    {
        lunchbox::ScopedWrite mutex(_lock);

        if (report)
            _spikeReport.reset(new SpikeReport(report));
        else
            _spikeReport.reset();

        for (const auto& i : _subScenes)
            i.second.subScene->setSimulation(_spikeReport);

        /* If spike visualization modes have changed the scene style will
           be reapplied (i.e. state sets and shaders will be rebuilt. */
        const bool showSpikes = (bool)report;
        const bool showSomaSpikes = showSpikes && !_compartmentReport;
        blockAttributeMapSignals();
        getAttributes().set("show_spikes", showSpikes);
        getAttributes().set("show_soma_spikes", showSomaSpikes);
        unblockAttributeMapSignals();

        _styleAttributeUpdates.set("show_spikes", showSpikes);
        _styleAttributeUpdates.set("show_soma_spikes", showSomaSpikes);
        SceneOperationPtr operation(
            new StyleAttributeUpdate(_styleAttributeUpdates));
        {
            lunchbox::ScopedWrite opMutex(_operationsLock);
            _masterSceneOperations->push(operation);
        }
        _styleAttributeUpdates.clear();
    }
    dirty(false);
}

void Scene::_Impl::mapSimulation(const uint32_t frameNumber,
                                 const float milliseconds)
{
    lunchbox::ScopedWrite mutex(_lock);

    if (!_compartmentReport && !_spikeReport)
        return;

    for (const auto& i : _subScenes)
        i.second.subScene->mapSimulation(frameNumber, milliseconds);
}

const CompartmentReportPtr& Scene::_Impl::getCompartmentReport() const
{
    return _compartmentReport;
}

const SpikeReportPtr& Scene::_Impl::getSpikeReport() const
{
    return _spikeReport;
}

unsigned int Scene::_Impl::prepareSimulation(const uint32_t frameNumber,
                                             const float milliseconds)
{
    lunchbox::ScopedWrite mutex(_lock);

    if (!_compartmentReport && !_spikeReport)
        return 0;

    size_t activeSubscenes = 0;
    for (const auto& i : _subScenes)
        if (i.second.subScene->prepareSimulation(frameNumber, milliseconds))
            ++activeSubscenes;

    return activeSubscenes;
}

void Scene::_Impl::pushOperation(const SceneOperationPtr& operation)
{
    lunchbox::ScopedWrite mutex(_operationsLock);
    _masterSceneOperations->push(operation);
}

void Scene::_Impl::registerDistributedObjects(osgEq::Config* config,
                                              InitData& initData)
{
    lunchbox::ScopedWrite mutex(_lock); /* Mutex really needed? */

    _config = config;
    config->registerObject(_masterSceneOperations);
    initData.sceneUUIDs[getID()] = _masterSceneOperations->getID();
}

void Scene::_Impl::deregisterDistributedObjects(InitData& initData)
{
    lunchbox::ScopedWrite mutex(_lock); /* Mutex really needed? */

    LBASSERT(_config);
    _config->deregisterObject(_masterSceneOperations);
    initData.sceneUUIDs.erase(getID());
}

void Scene::_Impl::mapDistributedObjects(osgEq::Config* config)
{
    lunchbox::ScopedWrite mutex(_lock); /* Mutex really needed? */

    const InitData& initData =
        static_cast<const InitData&>(config->getInitData());
    InitData::SceneDistributableUUIDs::const_iterator uuid =
        initData.sceneUUIDs.find(getID());
    if (uuid == initData.sceneUUIDs.end())
        throw std::runtime_error("Scene ID not found");
    config->mapObject(_sceneOperations, uuid->second);

    /* Indeed it's not possible to map all distributed objects yet
       because the subscenes have not been created yet.
       This function will just store the pointer to the config and
       delay object mapping to getOrCreateSubSceneNode. */
    LBASSERT(_masterSceneOperations->isAttached() ? _config == config
                                                  : _config == 0);
    _config = config;
}

void Scene::_Impl::unmapDistributedObjects()
{
    lunchbox::ScopedWrite mutex(_lock); /* Mutex really needed? */
    _unmapDistributedObjects();
}

osg::ref_ptr<osg::Node> Scene::_Impl::getOrCreateSubSceneNode(
    const eq::fabric::Range& range)
{
    lunchbox::ScopedRead mutex(_lock);

    SubSceneData& data = _subScenes[range];
    SubScenePtr& subScene = data.subScene;

    if (subScene.get() == 0)
    {
        subScene = _createSubScene();
        data.range = range;
    }

    return subScene->getNode();
}

unsigned int Scene::_Impl::computeCompositingPosition(
    const osg::Matrix& modelView, const eq::fabric::Range& range) const
{
    SubScenePtr subScene;
    {
        lunchbox::ScopedRead mutex(_lock);
        SubSceneMap::const_iterator i = _subScenes.find(range);
        if (i == _subScenes.end())
            return 0;
        subScene = i->second.subScene;
    }
    unsigned int nodeCount = round(1 / (range.end - range.start));
    return subScene->compositingPosition(modelView, nodeCount);
}

void Scene::_Impl::updateCameraData(const eq::fabric::Range& range,
                                    osg::Camera* camera)
{
    lunchbox::ScopedRead mutex(_lock);
    CameraData* cameraData = CameraData::getOrCreateCameraData(camera);
    SubSceneMap::const_iterator i = _subScenes.find(range);
    if (i != _subScenes.end())
        cameraData->setCircuitScene(i->second.subScene.get());
}

void Scene::_Impl::nodeSync(const uint32_t /*frameNumber*/,
                            const osgEq::FrameData& frameData)
{
    osgEq::FrameData::ObjectUUIDCommitMap::const_iterator commitID =
        frameData.commits.find(_sceneOperations->getID());
    _pendingOperationUpdates.push_back(commitID->second);

    {
        lunchbox::ScopedWrite mutex(_lock);
        for (const auto& m : _toAdd)
        {
            m.second->applyStyle(_sceneStyle);
            _extraModels->addChild(m.second->getNode());
        }
        _toAdd.clear();

        for (const auto& node : _toRemove)
            _extraModels->removeChild(node.get());
        _toRemove.clear();
    }

    /* Do not sync scene operations unless there's a subscene. The reason
       for this is that subscenes are created with 1 frame of delay after
       a scene is applied to a view. This delay causes an unexpected loss
       of scene operations by subscenes. */
    SceneOperations preOperations;
    SceneOperations postOperations;
    bool processPending = !_subScenes.empty();

    if (processPending)
    {
        for (const auto& id : _pendingOperationUpdates)
        {
            _sceneOperations->sync(id);
            _sceneOperations->extract(preOperations,
                                      SceneOperation::Order::preUpdate);
            _sceneOperations->extract(postOperations,
                                      SceneOperation::Order::postUpdate);
        }
    }
    preOperations.apply(*this);

    {
        lunchbox::ScopedWrite mutex(_lock);
        for (auto& i : _subScenes)
            _syncSubScene(i.second, _pendingOperationUpdates);
        _subscenesNeedReset = false;
    }

    postOperations.apply(*this);

    if (processPending)
        _pendingOperationUpdates.clear();
}

void Scene::_Impl::channelSync(const eq::fabric::Range& range,
                               const uint32_t frameNumber,
                               const osgEq::FrameData&)
{
    SubScenePtr subScene;
    {
        lunchbox::ScopedRead mutex(_lock);
        const auto s = _subScenes.find(range);
        if (s == _subScenes.end())
            return;
        subScene = s->second.subScene;
    }
    LBLOG(LOG_SCENE_UPDATES) << "channelSync: frame " << frameNumber
                             << ", subsene " << subScene.get() << std::endl;
    subScene->channelSync();
}

osg::BoundingBox Scene::_Impl::getSceneBoundingBox(
    const eq::fabric::Range& range) const
{
    lunchbox::ScopedRead mutex(_lock);

    SubSceneMap::const_iterator s = _subScenes.find(range);
    if (s != _subScenes.end())
        return s->second.subScene->getBoundingBox();
    return osg::BoundingBox();
}

osgEq::Scene::DBCompositingTechnique Scene::_Impl::getDBCompositingTechnique()
    const
{
    switch (_sceneAttributes->partitionType)
    {
    case DataBasePartitioning::NONE:
    /* no break */
    case DataBasePartitioning::SPATIAL:
        return ORDER_DEPENDENT;
        break;
    case DataBasePartitioning::ROUND_ROBIN:
        /* In this case the compositing technique depends on whether alpha
           blending is enabled or not. */
        if (_sceneStyle->getRenderBinManager().getAlphaBlendedRenderBin())
            return ORDER_INDEPENDENT_MULTIFRAGMENT;
        return ORDER_INDEPENDENT_SINGLE_FRAGMENT;
        break;
    default:
        LBUNREACHABLE;
        abort(); // LBUNREACHABLE should be enough, but it isn't
    }
}

osgTransparency::BaseRenderBin* Scene::_Impl::getAlphaBlendedRenderBin()
{
    return _sceneStyle->getRenderBinManager().getAlphaBlendedRenderBin();
}

void Scene::_Impl::addColorMapToAtlas(const ColorMapPtr& colorMap)
{
    _colorMaps->addColorMap(colorMap);
}

void Scene::_Impl::onAttributeChangedImpl(const AttributeMap& map,
                                          const std::string& name)
{
    lunchbox::ScopedWrite mutex(_lock);

    const bool autoUpdate = map("auto_update");

    /* Checking for attributes that change the scene styles */
    if (_isSceneStyleAttribute(name))
    {
        /* Copying the attribute value to _sceneStyleAttributeUpdates. */
        try
        {
            const AttributeMapPtr options = map(name);
            /* Attributes which are attribute maps themselves are deep copied
               to avoid problems with dirty signals. */
            AttributeMapPtr copy;
            if (options)
                copy.reset(new AttributeMap(*(options)));
            _styleAttributeUpdates.set(name, options);
        }
        catch (...)
        {
            _styleAttributeUpdates.set(name, map(name));
        }
    }
    else if (name.substr(0, 15) == "alpha_blending.")
    {
        const size_t i = name.find_last_of('.');
        const std::string prefix = name.substr(0, i);
        const std::string suffix = name.substr(i + 1);
        _styleAttributeUpdates(prefix).set(suffix, map(name));
    }

    if (!_styleAttributeUpdates.empty() && autoUpdate)
    {
        SceneOperationPtr operation(
            new StyleAttributeUpdate(_styleAttributeUpdates));
        {
            lunchbox::ScopedWrite opMutex(_operationsLock);
            _masterSceneOperations->push(operation);
        }
        _styleAttributeUpdates.clear();
        mutex.unlock();
        dirty(false);
    }
}

void Scene::_Impl::onAttributeChangingImpl(
    const AttributeMap& map, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    if (name == "alpha_blending")
    {
        const AttributeMapPtr attributes = parameters;
        RenderBinManager::validateAttributes(*attributes);
        return;
    }
    else if (name.substr(0, 15) == "alpha_blending.")
    {
        const AttributeMapPtr attributes = map("alpha_blending");
        AttributeMap copy = *attributes;
        const size_t i = name.find_last_of('.');
        if (i == name.find_first_of('.'))
        {
            const std::string suffix = name.substr(i + 1);
            copy.set(suffix, parameters);
        }
        else
        {
            const std::string prefix = name.substr(15, i);
            const std::string suffix = name.substr(i + 1);
            copy(prefix).set(suffix, parameters);
        }
        RenderBinManager::validateAttributes(copy);
        return;
    }
    if (name == "em_shading" || name == "auto_update" ||
        name == "inflatable_neurons" || name == "show_spikes" ||
        name == "show_soma_spikes")
    {
        (void)(bool) parameters;
        return;
    }
    throw std::runtime_error("Unknown or inmmutable attribute: " + name);
}

SubScenePtr Scene::_Impl::_createSubScene()
{
    SubScenePtr subScene(new SubScene(this));

    subScene->getNode()->addChild(_extraModels);
    SceneStylePtr style(new SceneStyle(_sceneStyle->getAttributes()));
    style->setColorMapAtlas(_colorMaps);
    subScene->applyStyle(style);

    subScene->setSimulation(_compartmentReport);
    subScene->setSimulation(_spikeReport);
    subScene->getMapper().simulationUpdated.connect(
        boost::bind(&_Impl::_onSimulationUpdated, this, _1));

    LBASSERT(_config);
    subScene->mapDistributedObjects(_config);

    return subScene;
}

bool Scene::_Impl::_removeObject(const unsigned int id)
{
    const size_t version = _version;

    if (Remover<NeuronObjects>(_neuronObjects)(id) ||
        Remover<SynapseObjects>(_synapseObjects)(id))
    {
        _subscenesNeedReset = true;
        ++_version;
        /* The scene will be recreated, so all existing subset handlers are
           going to be overriden, we need to notify them. */
        for (auto object : _neuronObjects)
            object.second->dirtySubsets();
    }
    /* For extra models there's no need to rebuild the subscenes because they
       are in a separate node and they are neither partitioned nor affect the
       partitioning of neurons/synapses. */
    else if (Remover<GeometryObjects>(_geometryObjects)(id))
        ;
    else if (Remover<ModelObjects>(_modelObjects)(id))
        ;
    else
        /* For these objects the scene doesn't need a real update */
        Remover<ClientOnlyObjects>(_clientObjects).operator()(id);

    return version != _version;
}

void Scene::_Impl::_clearSubScenes()
{
    if (!_config)
        return;

    for (const auto& i : _subScenes)
        i.second.subScene->unmapDistributedObjects(_config);

    _subScenes.clear();
}

void Scene::_Impl::_updateSubScene(SubSceneData& data)
{
    if (data.version == _version)
        return;

    LBLOG(LOG_SCENE_UPDATES) << "_updateSubScene " << data.subScene.get() << ' '
                             << data.range << std::endl;
    data.version = _version;

    if (_neuronObjects.empty() && _synapseObjects.empty())
    {
        assert(data.objects.empty());
        return;
    }

    if (data.range != eq::fabric::Range::ALL)
    {
        /* This can only work properly for empty subscenes */
        assert(data.objects.empty());
        const Neurons subset =
            _computeTargetSubsets(*data.subScene, data.range);
        _addNeuronsToSubScene(data, &subset);
    }
    else
        _addNeuronsToSubScene(data, nullptr);

    /* Synapses are not load balanced in DB modes at the moment. */
    _addSynapsesToSubScene(data);
}

void Scene::_Impl::_syncSubScene(SubSceneData& data,
                                 const std::vector<co::uint128_t>& pendingOps)
{
    if (_subSceneNeedsResetting(data))
    {
        SubScenePtr& subScene = data.subScene;
        LBASSERT(_config);
        subScene->unmapDistributedObjects(_config);
        /* Resetting before creating the new SubScene so the scene ID
           can be resused. Otherwise the arrays that depend on the
           scene ID will grow unnecessarily. */
        subScene.reset();
        data.objects.clear();
        subScene = _createSubScene();
    }
    _updateSubScene(data);

    data.subScene->applyOperations(pendingOps);
}

bool Scene::_Impl::_subSceneNeedsResetting(const SubSceneData& data)
{
    /* Subscenes still pending the first update are never reset */
    if (data.version == 0)
        return false;

    /* If the version is different and the subscene is not full range the
       scene will be completely recreated for load balancing. */
    if (data.version != _version && data.range != eq::fabric::Range::ALL)
        return true;

    return _subscenesNeedReset;
}

Neurons Scene::_Impl::_computeTargetSubsets(SubScene& subScene,
                                            const eq::fabric::Range& range)
{
    if (_neuronObjects.empty())
        return Neurons();

    /* Collapsing all neuron containers into a single one so the subset
       of neurons to display is computed over the whole scene set. */
    auto neurons = _neuronObjects.begin()->second->getNeurons();
    using namespace AttributeMapHelpers;
    auto mode = getEnum(_neuronObjects.begin()->second->getAttributes(), "mode",
                        RepresentationMode::WHOLE_NEURON);

    for (const auto& target : _neuronObjects)
    {
        const auto& neuronObject = target.second;
        neurons += neuronObject->getNeurons();
        /* This is a workaround the computeSubtarget function not
           accepting different representation modes. */
        if (RepresentationMode::SOMA !=
            getEnum(neuronObject->getAttributes(), "mode",
                    RepresentationMode::WHOLE_NEURON))
            mode = RepresentationMode::WHOLE_NEURON;
    }

    if (range == eq::fabric::Range::ALL)
        return neurons;

    const size_t originalSize = neurons.size();
    subScene.computeSubtarget(neurons, mode, range.start, range.end);
    LBINFO << "The subtarget for the range " << range << " contains "
           << neurons.size() << " neurons ("
           << (double(neurons.size()) / originalSize * 100) << "%)"
           << std::endl;
    return neurons;
}

void Scene::_Impl::_addNeuronsToSubScene(SubSceneData& data,
                                         const Neurons* subset)
{
    std::list<NeuronObjectPtr> toAdd;
    for (const auto& i : _neuronObjects)
    {
        if (data.objects.find(i.first) == data.objects.end())
        {
            data.objects.insert(i.first);
            toAdd.push_back(i.second);
        }
    }

    while (!toAdd.empty())
    {
        /* Collapsing targets whose display mode is the same as the head
           of the pending queue.
           During the process, the neuron containers are intersected with
           the computed subset in order to filter out those neurons that
           are not part of the scene in this node. */
        auto target = toAdd.begin();
        const auto& attributes = (*target)->getAttributes();
        auto neurons = subset ? *subset & (*target)->getNeurons()
                              : (*target)->getNeurons();
        toAdd.erase(target++);
        /** \bug Commenting the loop below and loading a large number of
            small targets (as is with -n 1 -n 2 ... -n 100) causes OpenGL
            errors (race conditions?). */
        for (; target != toAdd.end();)
        {
            if ((*target)->getAttributes() == attributes)
            {
                if (subset)
                    neurons += *subset & (*target)->getNeurons();
                else
                    neurons += (*target)->getNeurons();
                toAdd.erase(target++);
            }
            else
            {
                ++target;
            }
        }

        /* In some corner cases the target can become empty */
        if (neurons.empty())
            continue;

        data.subScene->addNeurons(neurons, attributes);
    }
}

void Scene::_Impl::_addSynapsesToSubScene(SubSceneData& data)
{
    for (const auto& i : _synapseObjects)
    {
        if (data.objects.find(i.first) != data.objects.end())
            continue;

        data.objects.insert(i.first);
        SynapseObject& target = *i.second;
        data.subScene->addSynapses(target);
    }
}

void Scene::_Impl::_unmapDistributedObjects()
{
    LBASSERT(_config);

    for (const auto& i : _subScenes)
    {
        SubScene& subScene = *i.second.subScene;
        subScene.unmapDistributedObjects(_config);
    }
    _subScenes.clear();
    _config->unmapObject(_sceneOperations);

    /* Clearing _config pointer only if this scene doesn't contain the
       master copy (this one also needs deregistering. */
    if (!_masterSceneOperations->isAttached())
        _config = 0;
}

void Scene::_Impl::commit(osgEq::FrameData& frameData)
{
    {
        lunchbox::ScopedWrite mutex(_operationsLock);
        if (_masterSceneOperations->isDirty())
        {
            frameData.commits[_masterSceneOperations->getID()] =
                _masterSceneOperations->commit();
        }
    }
}

void Scene::_Impl::_onSimulationUpdated(const float timestamp)
{
    ++_parentPointerMonitor;
    _parent->simulationUpdated(timestamp);
    --_parentPointerMonitor;
}

void Scene::_Impl::dirty(bool recomputeHome)
{
    ++_parentPointerMonitor;
    _parent->dirty(recomputeHome);
    --_parentPointerMonitor;
}

void Scene::_Impl::_invalidateParent()
{
    /* Despite boost::signals2 is thread safe, this lock is needed to make
       the following sequence atomic:
       if (_parent)
           _parent->someSignal();
       Otherwise there's always a chance of someSignal being invoked on an
       invalid pointer.
    */
    _parentPointerMonitor.waitEQ(0);
    _parent = 0;
}

void Scene::_Impl::_calculateSomasBoundingSphere() const
{
    std::vector<osg::Vec3> positions;
    for (const auto& target : _neuronObjects)
    {
        const Neurons& neurons = target.second->getNeurons();
        positions.reserve(positions.size() + neurons.size());
        for (const auto& neuron : neurons)
            positions.push_back(neuron->getPosition());
    }

    _somasBound = _calculateBoundingSphere(positions);
}

void Scene::_Impl::_calculateSynapsesBoundingSphere() const
{
    std::vector<osg::Vec3> positions;
    for (const auto& target : _synapseObjects)
    {
        const Vector3fs& locations = target.second->getLocations();
        for (const auto& v : locations)
            positions.push_back(vec_to_vec(v));
    }

    _synapsesBound = _calculateBoundingSphere(positions);
}
}
}

BOOST_CLASS_EXPORT(bbp::rtneuron::Scene::_Impl::Clear)
BOOST_CLASS_EXPORT(bbp::rtneuron::Scene::_Impl::ClearClipPlanes)
BOOST_CLASS_EXPORT(bbp::rtneuron::Scene::_Impl::HighlightCells)
BOOST_CLASS_EXPORT(bbp::rtneuron::Scene::_Impl::StyleAttributeUpdate)
BOOST_CLASS_EXPORT(bbp::rtneuron::Scene::_Impl::UpdateClipPlane)
