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

#include "Cache.h"
#include "ConstructionData.h"
#include "CylinderBasedModel.h"
#include "MeshModel.h"
#include "NeuronSkeleton.h"
#include "TubeletBasedModel.h"

#include "util/ObjectCache.h"

#include "scene/CircuitScene.h"

#include <brain/neuron/morphology.h>

#include <condition_variable>
#include <mutex>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
namespace
{
ModelPtr _createModel(const NeuronLOD lod, const NeuronParts parts,
                      const ConstructionData& data)
{
    /* Removing the axon at the model level only makes sense when unique
       morphologies can be assumed. */
    assert(parts == model::NEURON_PARTS_FULL ||
           data.sceneAttr.assumeUniqueMorphologies);

    ModelPtr model;
    switch (lod)
    {
    case NeuronLOD::MEMBRANE_MESH:
        model.reset(new model::MeshModel(parts, data));
        break;
    case NeuronLOD::DETAILED_SOMA:
    {
        model.reset(new model::MeshModel(model::NEURON_PARTS_SOMA, data));
        break;
    }
    case NeuronLOD::TUBELETS:
        model.reset(new model::TubeletBasedModel(parts, data));
        break;
    case NeuronLOD::HIGH_DETAIL_CYLINDERS:
        model.reset(new model::CylinderBasedModel(parts, data, 0, true));
        break;
    case NeuronLOD::LOW_DETAIL_CYLINDERS:
        model.reset(new model::CylinderBasedModel(parts, data, 4, false));
        break;
    default:
        std::cerr << "RTNeuron: invalid lod for this neuron "
                     "representation model class"
                  << std::endl;
        abort();
    }
    model->_lod = lod;
    return model;
}

using SkeletonDatabase =
    ObjectCache<NeuronSkeleton, const brain::neuron::Morphology*>;
SkeletonDatabase s_skeletonDatabase;

using ModelKey = std::pair<const brain::neuron::Morphology*, NeuronLOD>;
using ModelDatabase = ObjectCache<SkeletonModel, ModelKey>;
ModelDatabase s_modelDatabase;
}

class Cache::Impl
{
public:
    Impl(const CircuitScene& scene_)
        : scene(scene_)
    {
    }

    const CircuitScene& scene;

    SkeletonPtr getSkeleton(const Neuron& neuron, const NeuronParts parts)
    {
        /* Since neurons cannot be added twice to the scene we don't need
           to consider the axon for indexing the cache. Since the cache uses
           weak pointers a new skeleton will be created if the neuron is
           removed and added back to the scene with different options. */
        SkeletonWeakPtr& cachedSkeleton = _getSkeleton(neuron.getGID());
        SkeletonPtr skeleton = cachedSkeleton.lock();
        if (skeleton)
            return skeleton;

        auto morphology = neuron.getMorphology();
        assert(morphology);
        if (scene.getAttributes().assumeUniqueMorphologies)
            /* Always creating a new skeleton for the cell */
            skeleton.reset(new model::NeuronSkeleton(*morphology, parts));
        else
        {
            SkeletonPtr sharedSkeleton =
                s_skeletonDatabase.getOrCreate(morphology.get(), *morphology);
            skeleton.reset(new model::NeuronSkeleton(*sharedSkeleton));
        }
        cachedSkeleton = skeleton;
        return skeleton;
    }

private:
    std::mutex _mutex;
    typedef std::weak_ptr<model::NeuronSkeleton> SkeletonWeakPtr;
    typedef std::map<boost::uint32_t, SkeletonWeakPtr> SkeletonInstanceMap;
    SkeletonInstanceMap _skeletonInstances;

    SkeletonWeakPtr& _getSkeleton(uint32_t gid)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _skeletonInstances[gid];
    }
};

Cache::Cache(const CircuitScene& scene)
    : _impl(new Impl(scene))
{
}

Cache::~Cache()
{
    delete _impl;
}

ModelPtr Cache::getModel(const NeuronLOD lod,
                         const ConstructionData& data) const
{
    assert(&data.scene == &_impl->scene);
    const auto key = ModelKey(&data.morphology, lod);
    return s_modelDatabase.getOrCreate(key, _createModel, lod,
                                       NEURON_PARTS_FULL, data);
}

ModelPtr Cache::createModel(const NeuronLOD lod, const NeuronParts parts,
                            const ConstructionData& data) const
{
    assert(&data.scene == &_impl->scene);
    return _createModel(lod, parts, data);
}

SkeletonPtr Cache::getSkeleton(const Neuron& neuron,
                               const NeuronParts parts) const
{
    return _impl->getSkeleton(neuron, parts);
}

void Cache::clear()
{
    s_modelDatabase.clear();
    s_skeletonDatabase.clear();
}
}
}
}
}
