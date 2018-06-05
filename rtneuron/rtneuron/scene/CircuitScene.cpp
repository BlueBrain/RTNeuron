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

#include "CircuitScene.h"

#include "config/Globals.h"
#include "config/constants.h"
#include "data/Neuron.h"
#include "render/SceneStyle.h"
#include "render/Skeleton.h"

#include <lunchbox/intervalSet.h>

#include <OpenThreads/ScopedLock>

#include <osg/Geode>
#include <osg/Version>
#include <osgUtil/CullVisitor>

#ifdef RTNEURON_USE_OPENMP
#include <omp.h>
#endif

#include <chrono>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
class IDSet
{
public:
    IDSet() { _indices.insert(0, std::numeric_limits<unsigned int>::max()); }
    size_t getNextID()
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        size_t id = *_indices.begin();
        _indices.erase(id);
        return id;
    }

    void returnID(size_t id)
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        _indices.insert(id);
    }

protected:
    OpenThreads::Mutex _mutex;
    typedef lunchbox::IntervalSet<unsigned int> Indices;
    Indices _indices;
};

static IDSet s_freeIdentifiers;
}

/*
  Helper functions and classes
*/

void _applyStyle(osg::Node& node, const NeuronLOD lod, SceneStyle& style,
                 const AttributeMap& extra = AttributeMap())
{
    const NeuronLOD lods[] = {NeuronLOD::MEMBRANE_MESH,
                              NeuronLOD::DETAILED_SOMA,
                              NeuronLOD::HIGH_DETAIL_CYLINDERS,
                              NeuronLOD::LOW_DETAIL_CYLINDERS,
                              NeuronLOD::TUBELETS,
                              NeuronLOD::SPHERICAL_SOMA};
    SceneStyle::StateType styles[] = {SceneStyle::NEURON_MESH,
                                      SceneStyle::NEURON_MESH,
                                      SceneStyle::NEURON_PSEUDO_CYLINDERS,
                                      SceneStyle::NEURON_PSEUDO_CYLINDERS,
                                      SceneStyle::NEURON_TUBELETS,
                                      SceneStyle::NEURON_SPHERICAL_SOMA};
    for (size_t i = 0; i != size_t(NeuronLOD::NUM_NEURON_LODS); ++i)
    {
        if (lod == lods[i])
        {
            node.setStateSet(style.getStateSet(styles[i], extra));
            return;
        }
    }
}

class CircuitScene::CompileBuffersCallback : public osg::NodeCallback
{
public:
    CompileBuffersCallback(CircuitScene* scene LB_UNUSED)
#if USE_CUDA
        : _scene(scene)
#endif
    {
    }

    ~CompileBuffersCallback() {}
    void operator()(osg::Node* node, osg::NodeVisitor* nv)
    {
        osgUtil::CullVisitor* culler = dynamic_cast<osgUtil::CullVisitor*>(nv);
        osg::RenderInfo& renderInfo = culler->getRenderInfo();

        osg::Camera* camera = culler->getCurrentCamera();
        /* No camera is current in renderInfo when this callback is invoked
           and the compilation needs it to indentify the CUDA device. */
        renderInfo.pushCamera(camera);

#if USE_CUDA
        if (_scene->_attributes->useCUDACulling &&
            _scene->_attributes->preloadSkeletons)
        {
            Skeleton::MasterCullCallback* callback =
                dynamic_cast<Skeleton::MasterCullCallback*>(
                    _nestedCallback.get());
            assert(callback);
            /** \todo Only do this if the circuit has changed */

            callback->compileSkeletons(renderInfo, node);
        }
#endif
        if (_nestedCallback)
        {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
            (*dynamic_cast<osg::NodeCallback*>(_nestedCallback.get()))(node,
                                                                       nv);
#else
            (*_nestedCallback)(node, nv);
#endif
        }
        else
            traverse(node, nv);

        renderInfo.popCamera();
    }

private:
#if USE_CUDA
    CircuitScene* _scene;
#endif
};

void _nestCullCallback(osg::Node& node,
                       const osg::ref_ptr<osg::NodeCallback>& callback)
{
    callback->setNestedCallback(node.getCullCallback());
    node.setCullCallback(callback);
}

/*
  Constructor
*/

CircuitScene::CircuitScene(const CircuitSceneAttributesPtr& attributes)
    : _identifier(s_freeIdentifiers.getNextID())
    , _circuit(new osg::ClipNode())
    , _somas(SceneStyle::NEURON_SPHERICAL_SOMA)
    , _extraObjects(new osg::Group())
    , _staticClipping(false)
    , _modelCache(*this)
    , _attributes(attributes)
    , _style(new SceneStyle(AttributeMap())) // The style is applied externally
{
    _circuit->setNodeMask(~EVENT_MASK);
#if USE_CUDA
    if (_attributes->useCUDACulling)
        _nestCullCallback(*_circuit, Skeleton::createMasterCullCallback());
#endif
    _nestCullCallback(*_circuit, new CompileBuffersCallback(this));
}

/*
   Destructor
*/

CircuitScene::~CircuitScene()
{
    for (const NeuronPtr& neuron : _neurons)
        neuron->clearCircuitData(*this);

    s_freeIdentifiers.returnID(_identifier);
}

/*
  Member functions
*/

osg::Group* CircuitScene::getNeuronModelNode()
{
    return _circuit;
}

std::vector<osg::Node*> CircuitScene::getNodes()
{
    std::vector<osg::Node*> nodes;
    nodes.reserve(3);
    nodes.push_back(_circuit);
    nodes.push_back(_synapses.getNode());
    nodes.push_back(_extraObjects);

    return nodes;
}

void CircuitScene::addNeurons(const Neurons& neurons)
{
    if (neurons.empty())
        return;

    if (_somas.getNode()->getParents().empty())
        getNeuronModelNode()->addChild(_somas.getNode());

    const size_t neuronCount = neurons.size();

    std::stringstream progressMessage;
    progressMessage << "Adding " << neurons.size() << " neurons to the scene";
    size_t progress = 0;

    const auto start = std::chrono::high_resolution_clock::now();

    reportProgress(progressMessage.str(), progress, neuronCount);

/* Dynamic scheduling is needed because the varied neuron sizes are
   not distributed uniformly along the gid space. */
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < neuronCount; ++i)
    {
        NeuronPtr neuron = neurons[i];
        neuron->addToScene(*this);

        _updateCircuitBoundingBox(*neuron);

#pragma omp atomic
        ++progress;

#ifdef RTNEURON_USE_OPENMP
        if (omp_get_thread_num() == 0 && progress < neuronCount)
#endif
            reportProgress(progressMessage.str(), progress, neuronCount);
    }

    reportProgress(progressMessage.str(), progress, neuronCount);

    if (Globals::profileDataLoading())
    {
        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start)
                .count();
        std::cout << "Adding neurons to the scene time: " << elapsed
                  << " seconds" << std::endl;
    }

    _neurons += neurons;
}

void CircuitScene::addNode(osg::Node* node, const uint32_t groupHint,
                           const bool unmaskGroup)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if (groupHint == std::numeric_limits<uint32_t>::max())
    {
        _extraObjects->addChild(node);
        return;
    }
    osg::ref_ptr<osg::Group>& group = _extraObjectGroups[groupHint];
    if (!group)
    {
        group = new osg::Group();
        /* Making the auxiliary object subgroup invisible by default. */
        group->setNodeMask(~CULL_AND_DRAW_MASK);
        _extraObjects->addChild(group);
    }
    if (unmaskGroup)
        /* Making the subgroup visible when requested (e.g. a neuron has
           been added with skeleton representation mode) */
        group->setNodeMask(0xFFFFFFFF);

    group->addChild(node);
}

void CircuitScene::addNeuronModel(osg::Node* node)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    getNeuronModelNode()->addChild(node);
}

void CircuitScene::addNeuronModel(osg::Node* node, const NeuronLOD lod)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _getOrCreateSceneTypeGroup(lod)->addChild(node);
}

osg::Geode* CircuitScene::getOrCreateSphericalSomaGeode(
    const GeodeConstructor& functor)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    osg::Group* group = _getOrCreateSceneTypeGroup(NeuronLOD::SPHERICAL_SOMA);
    if (group->getNumChildren())
        return group->getChild(0)->asGeode();

    osg::ref_ptr<osg::Geode> geode = functor();
    group->addChild(geode);
    return geode;
}

SphereSet& CircuitScene::getSomas()
{
    return _somas;
}

SphereSet& CircuitScene::getSynapses()
{
    return _synapses;
}

void CircuitScene::applyStyle(const SceneStylePtr& style)
{
    _style = style;

    /* Applying style to different models groups */
    for (NeuronModelMapIter i = _neuronModels.begin(); i != _neuronModels.end();
         ++i)
    {
        if (i->second)
            _applyStyle(*i->second, i->first, *_style);
    }

    /* Applying style to individual objects */
    for (const NeuronPtr& neuron : _neurons)
    {
        neuron->applyStyle(*_style);
    }

    _somas.applyStyle(style);
    _synapses.applyStyle(style);
}

void CircuitScene::clear()
{
    _circuit->removeChild(0, _circuit->getNumChildren());
    _neuronModels.clear();

    //! \bug When we have some kind of delay between the frame end and
    //  the rendering (e.g time multiplexing) this sentence will make
    //  the rendering of the pending frame fail because neuron objects
    //  are still being used. Can shared pointers solve this issue?
    _neurons.clear();
    _somas.clear();
    _synapses.clear();
    _circuitBoundingBox = osg::BoundingBox();
}

osg::Group* CircuitScene::_getOrCreateSceneTypeGroup(const NeuronLOD lod)
{
    NeuronLOD finalLod;
    if (lod == NeuronLOD::LOW_DETAIL_CYLINDERS)
        finalLod = NeuronLOD::HIGH_DETAIL_CYLINDERS;
    else if (lod == NeuronLOD::DETAILED_SOMA)
        finalLod = NeuronLOD::MEMBRANE_MESH;
    else
        finalLod = lod;

    osg::Group*& group = _neuronModels[finalLod];

    if (!group)
    {
        group = new osg::Group();
        /* For the moment all the groups refer to neuron models */
        getNeuronModelNode()->addChild(group);
        AttributeMap extra;
        /* Extra attribute with the winding type for neuron meshes */
        if (finalLod == NeuronLOD::MEMBRANE_MESH &&
            getAttributes().primitiveOptions.strips &&
            !getAttributes().primitiveOptions.unrollStrips)
        {
            extra.set("clockwise_winding", true);
        }
        _applyStyle(*group, finalLod, *_style, extra);
    }
    return group;
}

void CircuitScene::_updateCircuitBoundingBox(const Neuron& neuron)
{
    double bounds[6];
    neuron.getWorldCoordsExtents(bounds[0], bounds[1],  // x and X
                                 bounds[2], bounds[3],  // y and Y
                                 bounds[4], bounds[5]); // z and Z

    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _circuitBoundingBox.expandBy(osg::BoundingBox(bounds[0], bounds[2],
                                                  bounds[4], bounds[1],
                                                  bounds[3], bounds[5]));
}
}
}
}

#define DEBUG_VIEW_FRUSTUM_CULLING
#ifdef DEBUG_VIEW_FRUSTUM_CULLING

#include <osg/Notify>
#include <osgUtil/RenderLeaf>
#include <osgUtil/StateGraph>

using namespace osg;
using namespace osgUtil;

#if !defined(OSG_GLES1_AVAILABLE) && !defined(OSG_GLES2_AVAILABLE) && \
    !defined(OSG_GL3_AVAILABLE)
void RenderLeaf::render(osg::RenderInfo& renderInfo, RenderLeaf* previous)
{
    osg::State& state = *renderInfo.getState();

    // don't draw this leaf if the abort rendering flag has been set.
    if (state.getAbortRendering())
    {
        // cout << "early abort"<<endl;
        return;
    }

    if (previous)
    {
        // apply matrices if required.
        state.applyProjectionMatrix(_projection.get());
        state.applyModelViewMatrix(_modelview.get());

        // apply state if required.
        StateGraph* prev_rg = previous->_parent;
        StateGraph* prev_rg_parent = prev_rg->_parent;
        StateGraph* rg = _parent;
        if (prev_rg_parent != rg->_parent)
        {
            StateGraph::moveStateGraph(state, prev_rg_parent, rg->_parent);

            // send state changes and matrix changes to OpenGL.
            state.apply(rg->getStateSet());
        }
        else if (rg != prev_rg)
        {
            // send state changes and matrix changes to OpenGL.
            state.apply(rg->getStateSet());
        }

        // draw the drawable
        _drawable->draw(renderInfo);
    }
    else
    {
        const double f = 1 + (getenv("SCREEN_OVERSIZE")
                                  ? strtod(getenv("SCREEN_OVERSIZE"), 0)
                                  : 0);
        if (f != 1)
        {
            double left, right, bottom, top, near, far;
            _projection->getFrustum(left, right, bottom, top, near, far);
            _projection->makeFrustum(left * f, right * f, bottom * f, top * f,
                                     near, far);

            glClearColor(0.1, 0.2, 0.3, 1.0);
            glClear(GL_COLOR_BUFFER_BIT);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glBegin(GL_QUADS);
            glColor4f(0.2, 0.3, 0.4, 1.0);
            double p = -1 / f, q = 1 / f;
            glVertex3f(p, p, 0);
            glVertex3f(p, q, 0);
            glVertex3f(q, q, 0);
            glVertex3f(q, p, 0);
            glEnd();
            glMatrixMode(GL_MODELVIEW);
        }
        // apply matrices if required.
        state.applyProjectionMatrix(_projection.get());
        state.applyModelViewMatrix(_modelview.get());

        // apply state if required.
        StateGraph::moveStateGraph(state, NULL, _parent->_parent);

        state.apply(_parent->getStateSet());

        // draw the drawable
        _drawable->draw(renderInfo);
    }

    if (_dynamic)
    {
        state.decrementDynamicObjectCount();
    }
}
#endif // OSG_VERSION_MAJOR == 2 && OSG_VERSION_MINOR <= 8

#endif
