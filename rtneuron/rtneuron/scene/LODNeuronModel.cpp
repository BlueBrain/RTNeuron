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

#include "LODNeuronModel.h"

#include "../ColorMap.h"
#include "data/Neuron.h"
#include "data/SimulationDataMapper.h"
#include "render/LODNeuronModelDrawable.h"
#include "render/Skeleton.h"
#include "scene/CircuitScene.h"
#include "scene/DetailedNeuronModel.h"
#include "scene/ModelViewInverseUpdate.h"
#include "scene/NeuronModelClipping.h"
#include "scene/SphericalSomaModel.h"
#include "util/vec_to_vec.h"

#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

#include <lunchbox/debug.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Constructors
*/
LODNeuronModel::LODNeuronModel(const Neuron* neuron)
    : NeuronModel(neuron)
{
    Drawable* drawable = new Drawable();
    drawable->_model = this;
    _drawable = drawable;
}

/*
  Destructor
*/
LODNeuronModel::~LODNeuronModel()
{
    _drawable->_model = 0;
}

/*
  Member functions
*/
void LODNeuronModel::setMaximumVisibleBranchOrder(unsigned int order)
{
    for (auto& submodel : _models)
        submodel.model->setMaximumVisibleBranchOrder(order);
}

void LODNeuronModel::setupSimulationOffsetsAndDelays(
    const SimulationDataMapper& mapper, const bool reuseCached)
{
    /* Setting up/updating offsets and delays on all submodels */
    for (auto& submodel : _models)
        submodel.model->setupSimulationOffsetsAndDelays(mapper, reuseCached);

    setSimulationUniforms(mapper);
}

void LODNeuronModel::addToScene(CircuitScene& scene)
{
    const auto position = _neuron->getPosition();
    const auto orientation = _neuron->getOrientation();

    /* Instantiating the submodel drawables when needed and adding them to
       the LOD drawable. */
    std::set<Skeleton*> skeletons;
    for (auto& submodel : _models)
    {
        submodel.model->setupAsSubModel(scene);

        auto bb = _drawable->getInitialBound();
        bb.expandBy(submodel.model->getInitialBound());
        _drawable->setInitialBound(bb);
    }

    /* Create a position attitude transform for the model (store it
       inside the neuron directly?) and add it to the scene */
    auto* neuron = new osg::PositionAttitudeTransform();
    auto* geode = new osg::Geode;
    geode->addDrawable(getDrawable());
    neuron->setPosition(position);
    neuron->setAttitude(orientation);
    neuron->addChild(geode);

#ifdef OSG_GL3_AVAILABLE
    /* Adding a osg_ModelViewMatrixInverse uniform to the drawable. */
    ModelViewInverseUpdate* callback = new ModelViewInverseUpdate();
    neuron->getOrCreateStateSet()->addUniform(callback->getUniform());
    neuron->setCullCallback(callback);
#endif

    scene.addNeuronModel(neuron);
}

void LODNeuronModel::applyStyle(const SceneStyle& style)
{
    for (auto& submodel : _models)
        submodel.model->applyStyle(style);
}

osg::BoundingBox LODNeuronModel::getInitialBound() const
{
    return _drawable->getInitialBound();
}

void LODNeuronModel::setColoring(const NeuronColoring& coloring)
{
    for (auto& submodel : _models)
        submodel.model->setColoring(coloring);

    /* Coloring state set attributes is not applied at the individual
       models, but from the LOD drawable. */
    if (coloring.requiresStateSet() || _drawable->getStateSet())
    {
        osg::StateSet* stateSet = _drawable->getOrCreateStateSet();
        coloring.applyColorMapState(stateSet);
    }
}

void LODNeuronModel::highlight(const bool on)
{
    static osg::ref_ptr<osg::Uniform> s_highlight =
        new osg::Uniform("highlighted", true);

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

    for (auto& submodel : _models)
        submodel.model->highlight(on);
}

void LODNeuronModel::setRenderOrderHint(int order)
{
    getDrawable()->getOrCreateStateSet()->setBinNumber(order);
    /* Not propagated to submodels because:
       - Their state sets are shared between model types, the unique state
         set for the morphology instance is at this model.
       - It's not really needed because sub model state sets are pushed on
         the LOD state set in the state set stack. */
}

osg::Drawable* LODNeuronModel::getDrawable()
{
    return _drawable;
}

void LODNeuronModel::softClip(const NeuronModelClipping& operation)
{
    _doOncePerSkeleton(
        boost::bind(&NeuronModel::softClip, _1, boost::cref(operation)));

    /* Revisiting the models to deal with soma only models (those models
       don't rely on any skeleton for clipping). */
    for (auto& submodel : _models)
    {
        if (submodel.model->isSomaOnlyModel())
            submodel.model->softClip(operation);
    }
}

void LODNeuronModel::addSubModel(NeuronModel* model, float min, float max)
{
    _models.push_back(SubModel(model, min, max));
    osg::BoundingBox bbox = _drawable->getInitialBound();
    bbox.expandBy(model->getInitialBound());
    _drawable->setInitialBound(bbox);
    auto* skeletonModel = dynamic_cast<DetailedNeuronModel*>(model);
    if (skeletonModel != 0)
        skeletonModel->_isSubModel = true;
}

void LODNeuronModel::accept(CollectSkeletonsVisitor& visitor LB_UNUSED) const
{
#ifdef USE_CUDA
    for (const auto& submodel : _models)
    {
        const osg::Drawable* drawable = submodel.model->getDrawable();
        if (!drawable)
            continue;
        const auto* callback = dynamic_cast<const Skeleton::CullCallback*>(
            drawable->getCullCallback());
        if (callback)
            callback->accept(visitor);
    }
#endif
}

void LODNeuronModel::setupAsSubModel(CircuitScene&)
{
    /* This function mustn't be called for this class */
    LBDONTCALL;
}

void LODNeuronModel::clip(const Planes& planes, const Clipping type)
{
    if (type == CLIP_HARD)
    {
        for (auto& submodel : _models)
            submodel.model->clip(planes, type);
    }
    else
    {
        _doOncePerSkeleton(boost::bind(&NeuronModel::clip, _1, planes, type));
    }
}

template <typename T>
void LODNeuronModel::_doOncePerSkeleton(const T& functor)
{
    std::set<Skeleton*> skeletons;
    for (auto& submodel : _models)
    {
        DetailedNeuronModel* model =
            dynamic_cast<DetailedNeuronModel*>(submodel.model.get());
        if (model && !model->isSomaOnlyModel())
        {
            /* Clipping skeletons only once. */
            Skeleton* skeleton = model->getSkeleton();
            if (skeletons.insert(skeleton).second)
                functor(model);
        }
    }
}
}
}
}
