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

#include "DetailedNeuronModel.h"

#include "../ColorMap.h"
#include "config/constants.h"
#include "data/Neuron.h"
#include "data/SimulationDataMapper.h"
#include "scene/CircuitScene.h"
#include "scene/NeuronModelClipping.h"
#include "scene/models/SkeletonModel.h"
#include "util/vec_to_vec.h"
#ifdef OSG_GL3_AVAILABLE
#include "scene/ModelViewInverseUpdate.h"
#endif

#include <brain/neuron/morphology.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/NodeCallback>
#include <osg/PositionAttitudeTransform>
#include <osgUtil/CullVisitor>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
   Helper functions
*/
namespace
{
/* Some models require the skeleton to be preprocessed before clipping when
   using hard clipping. This is needed because otherwise some capsules might
   be clipped but still referenced by a primitive element (triangle or line.
   This function determines whether preprocessing is needed or not.
*/
bool _skeletonNeedsPreprocessing(const NeuronLOD lod,
                                 const CircuitSceneAttributes& attributes)
{
    return (lod == NeuronLOD::MEMBRANE_MESH ||
            lod == NeuronLOD::LOW_DETAIL_CYLINDERS ||
            /* If meshes are loaded, detailed cylinders will be connected
               to the detailed soma, so the segments for the first order
               sections shouldn't lie outside the original capsules. */
            (lod == NeuronLOD::HIGH_DETAIL_CYLINDERS && !attributes.useMeshes));
}
}

/*
  Helper classes
*/
class ModelCullCallback : public osg::NodeCallback
{
public:
    ModelCullCallback(NeuronModel* model)
        : _model(model)
    {
    }

    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
    {
        osgUtil::CullVisitor* visitor = static_cast<osgUtil::CullVisitor*>(nv);
        const Neuron* neuron = _model->getNeuron();
        if (!neuron)
            return;

        const unsigned long frameNumber =
            visitor->getFrameStamp()->getFrameNumber();
        if (neuron->getRepresentationMode(frameNumber) ==
            RepresentationMode::WHOLE_NEURON)
        {
            traverse(node, nv);
        }
    }

private:
    NeuronModel* _model;
};

/*
  Constructors/destructor
*/
DetailedNeuronModel::DetailedNeuronModel(NeuronLOD lod,
                                         const model::NeuronParts parts,
                                         const model::ConstructionData& data)
    : NeuronModel(&data.neuron)
    , _lod(lod)
    , _simulationDataPending(false)
    , _hasOffsetArray(false)
    , _isSubModel(false)
{
    const CircuitScene& scene = data.scene;
    const model::Cache& cache = scene.getModelCache();
    const CircuitSceneAttributes& sceneAttr = data.scene.getAttributes();

    if (sceneAttr.assumeUniqueMorphologies)
    {
        /* With unique morphologies the models are assumed to not be
           reusable. This is to allow DB decompositions perform hard clipping
           on the models. */
        _model = cache.createModel(lod, parts, data);
        /* The full skeleton is instantiated now. We need it to be
           instantiated before clipping because all submodels that use it
           need to enlarge the capsules as needed. Otherwise, the final
           post processing of the primitives may fail (because a primitive
           that maps to a capsule that doesn't fully contain it and the
           capsule is clipped but not the primitive). */
        _skeleton = cache.getSkeleton(data.neuron, parts);
        if (_skeletonNeedsPreprocessing(lod, sceneAttr))
            _model->postProcess(*_skeleton);
        /* The drawable cannot be instantiated until we have the clipping
           planes from the scene */
    }
    else
    {
        /* Searching in the model database the morphology and type of model
           for it. Regardless of the display mode used, we need the full
           models if morphologies are shared, this is what getModel does. */
        _model = cache.getModel(lod, data);

        /* Creating the drawable node. */
        _skeleton = cache.getSkeleton(data.neuron, model::NEURON_PARTS_FULL);
        if (parts == model::NEURON_PARTS_SOMA_DENDRITES)
        {
            /* Ensures that the clipping is done only once for all models
               sharing the same skeleton */
            if (_skeleton.unique())
            {
                auto axon = _neuron->getMorphology()->getSectionIDs(
                    {brain::neuron::SectionType::axon});
                _skeleton->softClip(uint16_ts(axon.begin(), axon.end()));
                /* These masks are protected, so NeuronClipping cannot
                   unclip axon sections. */
                _skeleton->protectClipMasks();
            }
        }
        _drawable = _model->instantiate(_skeleton, sceneAttr);
        /* The state set of the drawable will be created when a rendering
           style is applied. */
    }
}

DetailedNeuronModel::~DetailedNeuronModel()
{
}

/*
  Member functions
*/

bool DetailedNeuronModel::isSomaOnlyModel() const
{
    return _lod == NeuronLOD::DETAILED_SOMA;
}

void DetailedNeuronModel::setMaximumVisibleBranchOrder(unsigned int order)
{
    /* This operation requires the drawable to be instantiated */
    assert(_drawable);
    _skeleton->setMaximumVisibleBranchOrder(order);
}

void DetailedNeuronModel::setupSimulationOffsetsAndDelays(
    const SimulationDataMapper& mapper, const bool reuseCached)
{
    if (_hasOffsetArray && reuseCached)
        return;

    /* Creating tne array of relative indices of this model and neuron. */
    osg::ref_ptr<osg::ShortArray> offsetsAndDelays(
        new osg::ShortArray(_model->_length));
    /* Not reusing the same VBO than for the rest of the geometry because
       different instances of the same morphology can have different number
       of compartments. */
    offsetsAndDelays->setVertexBufferObject(new osg::VertexBufferObject);

    /* Different instances of the same morphology may have different
       discretizations. Given that circuits will tend to have more unique
       morphologies, no optimization to reduce the memory footprint of
       repeated morphology/discretization pairs will be implemented */
    uint64_t absolute;
    mapper.getSimulationOffsetsAndDelays(*_neuron, &offsetsAndDelays->front(),
                                         absolute, _model->_sections->data(),
                                         _model->_positions->data(),
                                         _model->_length);

    if (absolute == LB_UNDEFINED_UINT64 && !mapper.getSpikeReport())
        /* This cell is not present in the simulation report. */
        return;

    assert(_drawable);
    /** \bug This code has race conditions with the rendering.
        It works but probably triggers undefined behaviour.
        https://bbpteam.epfl.ch/project/issues/browse/BBPRTN-36 */
    _drawable->setVertexAttribArray(BUFFER_OFFSETS_AND_DELAYS_GLATTRIB,
                                    offsetsAndDelays);
    _drawable->setVertexAttribBinding(BUFFER_OFFSETS_AND_DELAYS_GLATTRIB,
                                      osg::Geometry::BIND_PER_VERTEX);
    _hasOffsetArray = true;

    if (!_isSubModel)
        setSimulationUniforms(absolute);
}

void DetailedNeuronModel::addToScene(CircuitScene& scene)
{
    _instantiateDrawable(scene);

    if (_isDrawableEmpty())
        return;

    const auto position = _neuron->getPosition();
    const auto orientation = _neuron->getOrientation();

    /* Create a position attitude transform for the model (store it
       inside the neuron directly?) and add it to the scene */
    auto* neuron = new osg::PositionAttitudeTransform();
    /* When this function is called this model is not part of a LOD. That
       means that it needs its own cull callback to decide whether the neuron
       is visible or not in the current display mode. */
    /** \todo Storing the plain pointer is a bit dangerous because the
        lifetime of the node is not bound to the NeuronModel in any way. */
    neuron->addCullCallback(new ModelCullCallback(this));

#ifdef OSG_GL3_AVAILABLE
    /* Adding a osg_ModelViewMatrixInverse uniform to this state */
    auto* state = neuron->getOrCreateStateSet();
    auto* callback = new ModelViewInverseUpdate();
    state->addUniform(callback->getUniform());
    neuron->setCullCallback(callback);
#endif

    auto* geode = new osg::Geode;
    geode->addDrawable(_drawable.get());
    geode->setCullingActive(false); /* There is a cull callback in the drawable
                                       objects */
    neuron->addChild(geode);
    neuron->setPosition(position);
    neuron->setAttitude(orientation);

    /* Adding the model to the proper scene node */
    scene.addNeuronModel(neuron, _lod);
}

void DetailedNeuronModel::setupAsSubModel(CircuitScene& scene)
{
    _instantiateDrawable(scene);
}

void DetailedNeuronModel::applyStyle(const SceneStyle& style)
{
    osg::StateSet* stateSet = _model->getModelStateSet(_isSubModel, style);
    if (stateSet)
    {
        assert(_drawable);
        _drawable->setStateSet(stateSet);
    }
}

void DetailedNeuronModel::softClip(const NeuronModelClipping& operation)
{
    if (_lod == NeuronLOD::DETAILED_SOMA)
    {
        /* This is a special case as it doesn't use the skeleton for clipping */
        static const osg::ref_ptr<osg::Drawable::DrawCallback> noDraw =
            new osg::Drawable::DrawCallback;
        switch (operation.getSomaClipping())
        {
        case SomaClipping::CLIP:
            /* Making the drawable invisible */
            assert(_drawable->getDrawCallback() == 0 ||
                   _drawable->getDrawCallback() == noDraw);
            _drawable->setDrawCallback(noDraw);
            break;
        case SomaClipping::UNCLIP:
            /* Making the drawable visible */
            if (!_drawable->getDrawCallback())
                return;
            assert(_drawable->getDrawCallback() == noDraw);
            _drawable->setDrawCallback(0);
            break;
        default:;
        }
        return;
    }

    operation(*_skeleton);
}

osg::BoundingBox DetailedNeuronModel::getInitialBound() const
{
    return _model->getOrCreateBound(*getNeuron());
}

void DetailedNeuronModel::setColoring(const NeuronColoring& coloring)
{
    /* This operation requires the drawable to be instantiated */
    assert(_drawable);
    if (!_drawable)
        return;

    typedef osg::ref_ptr<osg::Vec4Array> Vec4ArrayPtr;
    typedef osg::ref_ptr<osg::Array> ArrayPtr;

    switch (coloring.getScheme())
    {
    case ColorScheme::SOLID:
    {
        Vec4ArrayPtr array =
            static_cast<osg::Vec4Array*>(_drawable->getColorArray());
        /* Reusing the array if posssible */
        if (!array || array->getNumElements() != 1)
        {
            array = osg::static_pointer_cast<osg::Vec4Array>(
                _model->getOrCreateColorArray(coloring,
                                              *_neuron->getMorphology()));
            _drawable->setColorArray(array);
        }
        (*array)[0] = coloring.getPrimaryColor();
        array->dirty();
        _drawable->setColorBinding(osg::Geometry::BIND_OVERALL);
        break;
    }
    case ColorScheme::BY_BRANCH_TYPE:
    {
        const auto& morphology = *_neuron->getMorphology();
        ArrayPtr colors = _model->getOrCreateColorArray(coloring, morphology);
        _drawable->setColorArray(colors);
        _drawable->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        break;
    }
    case ColorScheme::BY_WIDTH:
    case ColorScheme::BY_DISTANCE_TO_SOMA:
    {
        const auto& morphology = *_neuron->getMorphology();
        ArrayPtr values = _model->getOrCreateColorArray(coloring, morphology);
        size_t arrayIndex = STATIC_COLOR_MAP_VALUE_ATTRIB_NUM;
        _drawable->setVertexAttribArray(arrayIndex, values);
        _drawable->setVertexAttribBinding(arrayIndex,
                                          osg::Geometry::BIND_PER_VERTEX);
        break;
    }
    default:
        /* RANDOM must be treated by the Neuron object, otherwise the colors
           are different for each model */
        LBUNREACHABLE;
    }

    if (!_isSubModel &&
        (coloring.requiresStateSet() || _drawable->getStateSet()))
    {
        osg::StateSet* stateSet = _drawable->getOrCreateStateSet();
        coloring.applyColorMapState(stateSet);
    }
}

void DetailedNeuronModel::highlight(const bool on)
{
    if (_isSubModel)
        /* The state is handled by the LODNeuronModel in this case. */
        return;

    static osg::ref_ptr<osg::Uniform> s_highlight =
        new osg::Uniform("highlight", true);

    assert(_drawable);
    if (on)
    {
        osg::StateSet* stateSet = _drawable->getOrCreateStateSet();
        stateSet->addUniform(s_highlight);
    }
    else if (_drawable->getStateSet())
    {
        osg::StateSet* stateSet = _drawable->getStateSet();
        stateSet->removeUniform(s_highlight);
    }
}

void DetailedNeuronModel::setRenderOrderHint(int order)
{
    if (_isSubModel)
        return; /* Actually, this method shouldn't have been called */

    /* This operation requires the drawable to be instantiated */
    assert(_drawable);
    _drawable->getOrCreateStateSet()->setBinNumber(order);
}

osg::Drawable* DetailedNeuronModel::getDrawable()
{
    return _drawable.get();
}

Skeleton* DetailedNeuronModel::getSkeleton()
{
    return _skeleton.get();
}

void DetailedNeuronModel::clip(const Planes& planes, const Clipping type)
{
    assert(type == CLIP_SOFT || !_drawable);

    for (const auto& plane : planes)
    {
        if (type == CLIP_HARD)
            _skeleton->hardClip(plane);
        else
            _skeleton->softClip(plane);
    }
    if (_skeleton->getClipMasks() != 0)
    {
        /* Clipping from spatial partitions is protected. Furthermore, these
           masks will never be popped from the mask stack. */
        _skeleton->protectClipMasks();
    }
    if (type == CLIP_HARD)
        _model->clip(planes);
}

void DetailedNeuronModel::_instantiateDrawable(const CircuitScene& scene)
{
    const CircuitSceneAttributes& attributes = scene.getAttributes();

    if (attributes.assumeUniqueMorphologies && _drawable)
        throw std::runtime_error("Models cannot be reinstantiated");

    if (scene.isClippingStatic())
        applySceneClip(scene);

    if (!_drawable)
    {
        assert(attributes.assumeUniqueMorphologies);
        _drawable = _model->instantiate(_skeleton, attributes);

        if (_isDrawableEmpty())
        {
            if (!scene.getClipPlaneList().empty() &&
                _lod != NeuronLOD::DETAILED_SOMA)
            {
                LBWARN << "Instantiated an empty neuron model after clipping: "
                          "GID "
                       << _neuron->getGID() << ", morphology "
                       << _neuron->getMorphologyLabel() << std::endl;
            }
            return;
        }
    }
}

bool DetailedNeuronModel::_isDrawableEmpty() const
{
    if (!_drawable)
        return true;
    for (unsigned int i = 0; i != _drawable->getNumPrimitiveSets(); ++i)
    {
        if (_drawable->getPrimitiveSet(i)->getNumIndices() != 0)
            return false;
    }
    return true;
}
}
}
}
