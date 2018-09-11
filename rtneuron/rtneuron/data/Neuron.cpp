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

#include "Neuron.h"

#include "NeuronMesh.h"

#include "../AttributeMap.h"
#include "config/Globals.h"
#include "config/constants.h"
#include "data/CircuitCache.h"
#include "data/SimulationDataMapper.h"
#include "data/loaders.h"
#include "render/Skeleton.h"
#include "scene/CircuitScene.h"
#include "scene/CircuitSceneAttributes.h"
#include "scene/NeuronModel.h"
#include "scene/NeuronModelClipping.h"
#include "scene/PerFrameAttribute.h"
#include "util/shapes.h"
#include "util/vec_to_vec.h"

#include <brain/circuit.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

#include <OpenThreads/ScopedLock>
#include <osg/Geode>
#include <osg/PolygonMode>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>

#include <osg/Geometry>

#include <sstream>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper classes and functions
*/
namespace
{
using MorphologyCache = ObjectCache<brain::neuron::Morphology, std::string>;
MorphologyCache _morphologyCache;

using MeshCache = ObjectCache<NeuronMesh, std::string>;
MeshCache _meshCache;

using MorphologySkeletonCache =
    ObjectCache<osg::Geode, const brain::neuron::Morphology*, osg::ref_ptr>;
MorphologySkeletonCache _morphologySkeletonCache;
ObjectCache<osg::StateSet, void, osg::ref_ptr> _morphologySkeletonStateSet;

size_t _maxControlGroupsPerGroup = 25000;

inline osg::Vec3 _center(const brain::Vector4f& sample)
{
    return osg::Vec3(sample[0], sample[1], sample[2]);
}
inline float _radius(const brain::Vector4f& sample)
{
    return sample[3] * 0.5;
}
inline osg::Quat _toQuaternion(const brain::Quaternionf& v)
{
    return osg::Quat(v.x(), v.y(), v.z(), v.w());
}

osg::ref_ptr<osg::StateSet> _createFlatShadingState()
{
    osg::StateSet* stateSet = new osg::StateSet;
    std::vector<std::string> shaders;
    shaders.push_back("shading/default_color.frag");
    shaders.push_back("shading/phong_mesh.frag");
    shaders.push_back("shading/phong.frag");
    shaders.push_back("geom/flat_mesh.vert");
    shaders.push_back("geom/flat_mesh.geom");
#ifdef OSG_GL3_AVAILABLE
    shaders.push_back("geom/compute_clip_distances.geom");
#endif
    shaders.push_back("main.vert");
    shaders.push_back("main.frag");
    osg::Program* program = Loaders::loadProgram(shaders);
    program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    stateSet->setAttributeAndModes(program);

    return stateSet;
}

void _addSkeletonSegments(const brain::Vector4fs& samples,
                          const brain::neuron::SectionType type,
                          osg::Geode& geode)
{
    using SectionType = brain::neuron::SectionType;
    for (size_t i = 0; i != samples.size() - 1; ++i)
    {
        const osg::Vec3 p = _center(samples[i]);
        const float r1 = _radius(samples[i]);
        const osg::Vec3 q = _center(samples[i + 1]);
        const float r2 = _radius(samples[i + 1]);

        const osg::Vec3 center = (p + q) * 0.5;
        const float radius =
            type == SectionType::soma ? 0.25 : std::max(r1, r2);
        const osg::Vec3 axis = p - q;

        osg::Vec4 color;
        if (type == SectionType::soma)
            color = osg::Vec4(0.2, 1.0, 0.2, 1.0);
        else if (type == SectionType::axon)
            color = osg::Vec4(0.2, 0.6, 1, 1);
        else
            color = osg::Vec4(1, 0.1, 0, 1);

        geode.addDrawable(capsuleDrawable(center, radius, axis, color, 0.2));
    }
}

osg::ref_ptr<osg::Geode> _createSimpleRepresentation(
    const brain::neuron::Morphology& morphology)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();

    using S = brain::neuron::SectionType;
    _addSkeletonSegments(morphology.getSoma().getProfilePoints(), S::soma,
                         *geode);
    const auto sections =
        morphology.getSections({S::dendrite, S::apicalDendrite, S::axon});
    for (const auto& section : sections)
    {
        const auto samples = section.getSamples();
        _addSkeletonSegments(samples, section.getType(), *geode);
    }
    geode->setStateSet(
        _morphologySkeletonStateSet.getOrCreate(_createFlatShadingState));
    return geode;
}

osg::Geode* _getOrCreateSimpleRepresentation(
    const brain::neuron::Morphology& morphology)
{
    return _morphologySkeletonCache.getOrCreate(&morphology,
                                                _createSimpleRepresentation,
                                                morphology);
}

NeuronMeshPtr _loadOrCreateMesh(const std::string& name,
                                const std::string& prefix,
                                const brain::neuron::MorphologyPtr& morphology)
{
    try
    {
        return NeuronMesh::load(name, prefix);
    }
    catch (...)
    {
    }

#if RTNEURON_USE_NEUMESH
    if (morphology)
        return std::make_shared<NeuronMesh>(*morphology);
#else
    (void)morphology;
#endif

    return NeuronMeshPtr();
}

class RepresentationClipping : public NeuronModelClipping
{
public:
    RepresentationClipping(const Neuron& neuron, const RepresentationMode mode,
                           const RepresentationMode current)
        : _neuron(neuron)
        , _mode(mode)
        , _current(current)
    {
        assert(mode != current);
    }

    void operator()(Skeleton& skeleton) const final
    {
        if (_mode == RepresentationMode::NO_AXON)
        {
            uint16_ts axon;
            for (const auto& id : _neuron.getMorphology()->getSectionIDs(
                     {brain::neuron::SectionType::axon}))
                axon.push_back(id);
            skeleton.softUnclipAll();
            skeleton.softClip(axon);
            skeleton.protectClipMasks();
        }
        else if (_current == RepresentationMode::NO_AXON)
        {
            /* Popping the axon masks is only needed if NO_AXON is using
               soft clipping. This is always the case except when NO_AXON
               is used at construction with unique morphologies, and then
               representation mode changes do not call this code because the
               transitions that would trigger it are not allowed. */

            {
                skeleton.popClipMasks();
                skeleton.softUnclipAll();
            }
        }
    }

    bool isNoOperation()
    {
        return _mode != RepresentationMode::NO_AXON &&
               _current != RepresentationMode::NO_AXON;
    }

    SomaClippingEnum getSomaClipping() const final
    {
        return SomaClipping::NO_OPERATION;
    }

private:
    const Neuron& _neuron;
    const RepresentationMode _mode;
    const RepresentationMode _current;
};

/**
   Class for perfoming visibility updates on SphericalSomaModel models that
   do not depend on LODNeuronModel.
   This is the case when the restrictedMode of the neuron is SOMA.
*/
class SomaModelCulling : public NeuronModelClipping
{
public:
    SomaModelCulling(const bool visible)
        : _visible(visible)
    {
    }

    void operator()(Skeleton&) const final {}
    SomaClippingEnum getSomaClipping() const final
    {
        return _visible ? SomaClipping::UNCLIP : SomaClipping::CLIP;
    }

private:
    bool _visible;
};
}

struct Neuron::RenderData
{
    RenderData(const Neuron& neuron, const RepresentationMode mode)
        : _bufferIndex(0)
        , _restrictedMode(mode)
        , _displayMode(mode)
        , _simulationDataPending(false)
        , _extraObjects(new osg::PositionAttitudeTransform())
    {
        _extraObjects->setPosition(neuron.getPosition());
        _extraObjects->setAttitude(neuron.getOrientation());
    }

    void updateMode(const RepresentationMode mode, Neuron& neuron)
    {
        _displayMode.setValue(mode);

        /* Dealing with the special case of SEGMENT_SKELETON */
        if (mode == RepresentationMode::SEGMENT_SKELETON)
        {
            if (_extraObjects->getNumChildren() == 0)
            {
                _extraObjects->addChild(
                    _getOrCreateSimpleRepresentation(*neuron.getMorphology()));
            }
            _extraObjects->getChild(0)->setNodeMask(0xFFFFFFFF);
            if (!_extraObjects->getParents().empty())
            {
                /* Enabling cull/draw at the node containing the extra objects.
                   The group won't be set back to invisible for simplicity. */
                _extraObjects->getParent(0)->setNodeMask(0xFFFFFFFF);
            }
        }
        else
        {
            if (_extraObjects->getNumChildren() != 0)
                _extraObjects->getChild(0)->setNodeMask(~CULL_AND_DRAW_MASK);
        }
    }

    NeuronModels _models;

    size_t _bufferIndex;
    RepresentationMode _restrictedMode;
    PerFrameAttribute<RepresentationMode> _displayMode;
    bool _simulationDataPending;

    /* Stores operations to can only be applied to neuron models after
       instantiation. */
    typedef std::vector<ModelOperation> PendingOperations;
    PendingOperations _pendingOperations;

    osg::ref_ptr<osg::PositionAttitudeTransform> _extraObjects;
};

/*
  Constructors/destructor
*/
Neuron::Neuron(const uint32_t gid, const CircuitCachePtr& circuit)
    : _gid(gid)
    , _circuit(circuit)
    , _position(vec_to_vec(_circuit->circuit->getPositions({_gid})[0]))
    , _orientation(_toQuaternion(_circuit->circuit->getRotations({_gid})[0]))
{
    try
    {
        _mtype = _circuit->circuit->getMorphologyTypes({_gid})[0];
        _etype = _circuit->circuit->getElectrophysiologyTypes({_gid})[0];
    }
    catch (...)
    {
        /* mtype and etype information is not available. Leaving members
           to their default values. */
    }
}

Neuron::Neuron(const Neuron& other, const CircuitSceneAttributes& attributes,
               RepresentationMode mode)
    : _gid(other._gid)
    , _circuit(other._circuit)
    , _morphology(other._morphology)
    , _mesh(other._mesh)
    , _position(other._position)
    , _orientation(other._orientation)
    , _mtype(other._mtype)
    , _etype(other._etype)
    , _rd(new RenderData(other, mode))
{
    if (mode == RepresentationMode::SOMA ||
        mode == RepresentationMode::NO_DISPLAY)
        return;

    if (mode == RepresentationMode::SEGMENT_SKELETON)
        _rd->updateMode(mode, *this);

    getMorphology();
    if (mode != RepresentationMode::SOMA && !_mesh &&
        attributes.areMeshesRequired())
    {
        getMesh(attributes);
    }
}

Neuron::Neuron()
{
}

Neuron::~Neuron()
{
}

/*
  Member functions
*/
const osg::Vec3& Neuron::getPosition() const
{
    return _position;
}

const osg::Quat& Neuron::getOrientation() const
{
    return _orientation;
}

std::string Neuron::getMorphologyLabel() const
{
    return _circuit->circuit->getMorphologyNames({_gid})[0];
}

std::string Neuron::getMorphologyType() const
{
    if (_mtype >= _circuit->mtypeNames.size())
        return "unknown";
    return _circuit->mtypeNames[_mtype];
}

std::string Neuron::getElectrophysiologyType() const
{
    if (_etype >= _circuit->etypeNames.size())
        return "unknown";
    return _circuit->etypeNames[_etype];
}

brain::neuron::MorphologyPtr _getMorphology(brain::Circuit& circuit,
                                            const uint32_t gid)
{
    return circuit.loadMorphologies(
        {gid}, brain::Circuit::Coordinates::local)[0];
}

brain::neuron::MorphologyPtr Neuron::getMorphology() const
{
    auto& circuit = *_circuit->circuit;
    if (!_morphology)
    {
        auto name = circuit.getMorphologyNames({_gid})[0];
        try
        {
            if (circuit.getAttribute<char>("recenter", {_gid})[0])
                name += "c";
        }
        catch (...)
        {}
        _morphology = _morphologyCache.getOrCreate(
            name, _getMorphology, circuit, _gid);
    }
    return _morphology;
}

NeuronMeshPtr Neuron::getMesh(const CircuitSceneAttributes& attributes) const
{
    if (!_mesh)
    {
        const auto prefix = attributes.getMeshPath();
        const auto& name = getMorphologyLabel();
        auto morphology = attributes.generateMeshes
                              ? getMorphology()
                              : brain::neuron::MorphologyPtr();
        _mesh = _meshCache.getOrCreate(name, _loadOrCreateMesh, name, prefix,
                                       morphology);
    }
    return _mesh;
}

float Neuron::getSomaRenderRadius() const
{
    if (_morphology)
        return _morphology->getSoma().getMeanRadius() *
               SOMA_MEAN_TO_RENDER_RADIUS_RATIO;
    return Globals::getDefaultSomaRadius(getMorphologyType());
}

void Neuron::clearCaches()
{
    _morphologyCache.clear();
    _meshCache.clear();
    _morphologySkeletonCache.clear();
    _morphologySkeletonStateSet.clear();
    NeuronModel::clearCaches();
}

size_t Neuron::getSimulationBufferIndex() const
{
    return _rd->_bufferIndex;
}
void Neuron::setSimulationBufferIndex(const size_t index)
{
    _rd->_bufferIndex = index;
}

void Neuron::setupSimulationOffsetsAndDelays(const SimulationDataMapper& mapper,
                                             const bool reuseCached)
{
    for (const auto& model : _rd->_models)
        model->setupSimulationOffsetsAndDelays(mapper, reuseCached);
}

void Neuron::addToScene(CircuitScene& scene)
{
    assert(_rd->_models.empty());
    _rd->_models = NeuronModel::createNeuronModels(this, scene);

    if (_rd->_restrictedMode == RepresentationMode::NO_DISPLAY)
        return;

    assert(!_rd->_models.empty());

    const bool unmask =
        _rd->_restrictedMode == RepresentationMode::SEGMENT_SKELETON;
    scene.addNode(_rd->_extraObjects, _gid / _maxControlGroupsPerGroup, unmask);

    for (const auto& model : _rd->_models)
    {
        model->addToScene(scene);
        model->applyStyle(scene.getStyle());
    }

    /* Pending operations are applied after addition to scene because some
       models require this to make the operations effective (e.g. coloring
       of spherical somas. */
    for (const auto& operation : _rd->_pendingOperations)
        for (const auto& model : _rd->_models)
            operation(*model);
    _rd->_pendingOperations.clear();
}

void Neuron::applyStyle(SceneStyle& style)
{
    if (_rd->_models.empty())
        /* It doesn't make sense to store this operation as pending if the
           models haven't been instanced because the scene style is applied
           after the instantiation occurs in addToScene */
        return;

    for (const auto& model : _rd->_models)
        model->applyStyle(style);
}

void Neuron::setMaximumVisibleBranchOrder(const unsigned int order)
{
    ModelOperation operation =
        boost::bind(&NeuronModel::setMaximumVisibleBranchOrder, _1, order);

    if (_rd->_models.empty() &&
        _rd->_restrictedMode != RepresentationMode::NO_DISPLAY)
    {
        _rd->_pendingOperations.push_back(operation);
        return;
    }

    applyPerModelOperation(operation);
}

void Neuron::applyColoring(const NeuronColoring& coloring)
{
    if (coloring.getScheme() == ColorScheme::RANDOM)
    {
        /* Ensuring that all submodels use the same color. */
        NeuronColoring resolved = coloring;
        resolved.resolveRandomColor();
        applyPerModelOperation(
            boost::bind(&NeuronModel::setColoring, _1, resolved));
    }
    else
    {
        applyPerModelOperation(
            boost::bind(&NeuronModel::setColoring, _1, boost::cref(coloring)));
    }
}

void Neuron::highlight(const bool on)
{
    applyPerModelOperation(boost::bind(&NeuronModel::highlight, _1, on));
}

void Neuron::applyPerModelOperation(const ModelOperation& operation)
{
    if (_rd->_models.empty() &&
        _rd->_restrictedMode != RepresentationMode::NO_DISPLAY)
    {
        _rd->_pendingOperations.push_back(operation);
        return;
    }

    for (const auto& model : _rd->_models)
        operation(*model);
}

bool Neuron::getWorldCoordsExtents(double& xMin, double& xMax, double& yMin,
                                   double& yMax, double& zMin,
                                   double& zMax) const
{
    /* Local coordinate frame bounding box */
    osg::BoundingBox local;
    if (_rd->_restrictedMode == RepresentationMode::SOMA ||
        _rd->_restrictedMode == RepresentationMode::NO_DISPLAY)
    {
        const float radius =
            _morphology ? _morphology->getSoma().getMaxRadius()
                        : Globals::getDefaultSomaRadius(getMorphologyType());
        /* In soma mode, the soma is assumed to be centered at (0, 0, 0), which
           should be the case anyway. */
        local.set(-radius, -radius, -radius, radius, radius, radius);
    }
    else if (_rd->_models.empty())
    {
        if (!_morphology)
            return false;
        using S = brain::neuron::SectionType;
        for (const auto& section : _morphology->getSections(
                 {S::axon, S::dendrite, S::apicalDendrite}))
        {
            for (const auto& s : section.getSamples())
            {
                local.expandBy(
                    osg::BoundingSphere(osg::Vec3(s[0], s[1], s[2]), s[3]));
            }
        }
    }

    for (const auto& model : _rd->_models)
        local.expandBy(model->getInitialBound());

    /* World bounding box computation */
    osg::BoundingBox world;
    for (int i = 0; i < 8; i++)
        world.expandBy(_orientation * local.corner(i) + _position);
    xMin = world.xMin();
    xMax = world.xMax();
    yMin = world.yMin();
    yMax = world.yMax();
    zMin = world.zMin();
    zMax = world.zMax();

    return true;
}

RepresentationMode Neuron::getRestrictedRepresentationMode() const
{
    return _rd->_restrictedMode;
}

RepresentationMode Neuron::getRepresentationMode() const
{
    return _rd->_displayMode.getValue();
}

RepresentationMode Neuron::getRepresentationMode(
    const unsigned long frameNumber) const
{
    RepresentationMode mode;
    /* If there is no value for the given frameNumber then we return
       the last value stored. */
    if (!_rd->_displayMode.getValue(frameNumber, mode))
        mode = _rd->_displayMode.getValue();
    return mode;
}

void Neuron::setRepresentationMode(const RepresentationMode mode)
{
    const RepresentationMode current = _rd->_displayMode.getValue();
    if (mode == current)
        return;

    if ((_rd->_restrictedMode == RepresentationMode::SOMA &&
         mode != RepresentationMode::SOMA &&
         mode != RepresentationMode::NO_DISPLAY) ||
        (_rd->_restrictedMode == RepresentationMode::NO_AXON &&
         mode == RepresentationMode::WHOLE_NEURON))
    {
        LBWARN << lexical_cast<std::string>(mode)
               << " representation not available for neuron: " << _gid
               << std::endl;
        return;
    }

    /* Updating the clipping masks depending on the representation mode
       only makes sense if all modes are available. */
    if (_rd->_restrictedMode == RepresentationMode::SEGMENT_SKELETON ||
        _rd->_restrictedMode == RepresentationMode::WHOLE_NEURON)
    {
        RepresentationClipping clipping(*this, mode, current);
        /* Don't do nothing unless needed, otherwise the softClip throws
           when culling is disabled, even despite it's no clipping is needed
           for some transitions (e.g. whole neuron to segment skeleton). */
        if (!clipping.isNoOperation())
            applyPerModelOperation(
                boost::bind(&NeuronModel::softClip, _1, clipping));
    }

    if (_rd->_restrictedMode == RepresentationMode::SOMA &&
        (mode == RepresentationMode::NO_DISPLAY ||
         mode == RepresentationMode::SOMA))
    {
        /* Most transitions rely on the LODNeuronModel::Drawable::CullCallback
           to toggle models in/visible. This one is a special because there's
           no LODNeuronModel, so we have to toggle the visibility from here. */
        SomaModelCulling clipping(mode == RepresentationMode::SOMA);
        applyPerModelOperation(
            boost::bind(&NeuronModel::softClip, _1, clipping));
    }

    _rd->updateMode(mode, *this);
}

void Neuron::upgradeRepresentationMode(const RepresentationMode mode,
                                       CircuitScene& scene)
{
    /* Ensuring somas are made visible if the restricted mode is SOMA.
       Otherwise setRepresentationMode won't make them visible if the current
       representation mode is NO_DISPLAY. */
    if (_rd->_restrictedMode == RepresentationMode::SOMA)
    {
        SomaModelCulling clipping(true);
        applyPerModelOperation(
            boost::bind(&NeuronModel::softClip, _1, clipping));
    }

    NeuronModel::upgradeModels(this, scene, mode).swap(_rd->_models);
    _rd->_restrictedMode = mode;

    for (const auto& model : _rd->_models)
    {
        model->addToScene(scene);
        model->applyStyle(scene.getStyle());
    }

    setRepresentationMode(mode);
}

const NeuronModels& Neuron::getModels() const
{
    return _rd->_models;
}
void Neuron::clearCircuitData(const CircuitScene& scene)
{
    for (const auto& model : _rd->_models)
        model->clearCircuitData(scene);
}

osg::Node* Neuron::getExtraObjectNode()
{
    return _rd->_extraObjects;
}
}
}
}
