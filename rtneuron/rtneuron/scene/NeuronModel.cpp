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

#include "NeuronModel.h"

#include "../AttributeMap.h"
#include "data/Neuron.h"
#include "data/SimulationDataMapper.h"
#include "render/LODNeuronModelDrawable.h"
#include "scene/CircuitScene.h"
#include "scene/DetailedNeuronModel.h"
#include "scene/LODNeuronModel.h"
#include "scene/SphericalSomaModel.h"
#include "scene/models/Cache.h"
#include "util/attributeMapHelpers.h"
#include "util/vec_to_vec.h"

#include <osg/BoundingBox>
#include <osg/BoundingSphere>

#include <boost/lexical_cast.hpp>

#include <fstream>
#include <sstream>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper functions
*/
class NeuronModel::Helpers
{
public:
    /* The functions here need to be declared inside NeuronModel because
       they access protected/private members of classes that have NeuronModel
       as a friend. */

    static NeuronModels _processLODSpecification(
        const AttributeMap& lods, const model::NeuronParts parts,
        const model::ConstructionData& data, NeuronModel* somaModel)
    {
        LODNeuronModel* model = new LODNeuronModel(&data.neuron);
        NeuronModels models{model};

        for (const auto& lod : lods)
        {
            NeuronLOD lodName;
            try
            {
                lodName = lexical_cast<NeuronLOD>(lod.first);
            }
            catch (const std::runtime_error&)
            {
                LBWARN << "Warning: Ignoring unknown level of detail model: "
                       << lod.first << std::endl;
                continue;
            }

            double start = 0, end = 0;
            try
            {
                start = lod.second(0);
                end = lod.second(1);
            }
            catch (...)
            {
                LBWARN << "Invalid range specification for model: " << lod.first
                       << std::endl;
                continue;
            }
            if (start > end)
            {
                LBWARN << "Invalid range specification for model: " << lod.first
                       << std::endl;
                continue;
            }

            NeuronModelPtr subModel;
            switch (lodName)
            {
            case NeuronLOD::SPHERICAL_SOMA:
                subModel = somaModel ? somaModel
                                     : new SphericalSomaModel(&data.neuron);
                break;
            case NeuronLOD::MEMBRANE_MESH:
            case NeuronLOD::DETAILED_SOMA:
                if (!data.mesh)
                {
                    LBWARN << "Cannot create mesh based model for neuron "
                           << data.neuron.getGID()
                           << ", ignoring this level of detail" << std::endl;
                    continue;
                }
            /* no break */
            default:
            {
                auto& neuron = data.neuron;
                if (!neuron.getMorphology())
                {
                    LBWARN << "Morphology " << neuron.getMorphologyLabel()
                           << " of neuron " << neuron.getGID() << " not found."
                           << std::endl;
                    continue;
                }
                subModel = new DetailedNeuronModel(lodName, parts, data);
            }
            }
            model->addSubModel(subModel, start, end);
        }

        return models;
    }

    static NeuronModels _createDetailedModels(Neuron* neuron,
                                              const CircuitScene& scene,
                                              const RepresentationMode mode,
                                              NeuronModel* somaModel)
    {
        const model::NeuronParts parts =
            mode == RepresentationMode::NO_AXON
                ? model::NEURON_PARTS_SOMA_DENDRITES
                : model::NEURON_PARTS_FULL;
        const auto& sceneAttr = scene.getAttributes();

        auto lod = sceneAttr.neuronLODs;
        if (lod)
        {
            const model::ConstructionData data(*neuron, scene);
            return Helpers::_processLODSpecification(*lod, parts, data,
                                                     somaModel);
        }

        auto* model = new LODNeuronModel(neuron);
        NeuronModels models{model};
        NeuronModelPtr soma(somaModel ? somaModel
                                      : new SphericalSomaModel(neuron));
        const model::ConstructionData data(*neuron, scene);

        if (!neuron->getMorphology())
        {
            LBWARN << "Morphology " << neuron->getMorphologyLabel()
                   << " of neuron " << neuron->getGID() << " not found."
                   << std::endl;
            model->addSubModel(soma, 0, 1);
            return models;
        }

        if (sceneAttr.areMeshesRequired() && data.mesh)
        {
            /* The soma model is only added to support display mode
               switching. */
            model->addSubModel(soma, 0, 0);
            DetailedNeuronModelPtr mesh(
                new DetailedNeuronModel(NeuronLOD::MEMBRANE_MESH, parts, data));
            model->addSubModel(mesh, 0, 1);
        }
        else
        {
            model->addSubModel(soma, 0, 1);
            DetailedNeuronModelPtr cylinders(
                new DetailedNeuronModel(NeuronLOD::HIGH_DETAIL_CYLINDERS, parts,
                                        data));
            model->addSubModel(cylinders, 0, 1);
        }
        return models;
    }

    static NeuronModels _createModels(Neuron* neuron,
                                      const RepresentationMode mode,
                                      const CircuitScene& scene)
    {
        if (mode == RepresentationMode::NO_DISPLAY)
            return NeuronModels();

        if (mode == RepresentationMode::SOMA ||
            (mode == RepresentationMode::SEGMENT_SKELETON &&
             !neuron->getMesh(scene.getAttributes())))
        {
            return NeuronModels{new SphericalSomaModel(neuron)};
        }

        return _createDetailedModels(neuron, scene, mode, 0);
    }
};

/*
  Constructors/destructor
*/
NeuronModel::NeuronModel(const Neuron* neuron)
    : _neuron(neuron)
{
}

NeuronModel::~NeuronModel()
{
}

/*
  Member functions
*/

void NeuronModel::clearCaches()
{
    model::Cache::clear();
}

NeuronModels NeuronModel::createNeuronModels(Neuron* neuron,
                                             const CircuitScene& scene)
{
    using namespace AttributeMapHelpers;
    const RepresentationMode mode = neuron->getRestrictedRepresentationMode();
    return Helpers::_createModels(neuron, mode, scene);
}

NeuronModels NeuronModel::upgradeModels(Neuron* neuron,
                                        const CircuitScene& scene,
                                        const RepresentationMode mode)
{
    const RepresentationMode currentMode =
        neuron->getRestrictedRepresentationMode();

    switch (currentMode)
    {
    case RepresentationMode::NO_DISPLAY:
        assert(neuron->getModels().empty());
        return Helpers::_createModels(neuron, mode, scene);

    case RepresentationMode::NO_AXON:
        if (mode != RepresentationMode::SOMA &&
            mode != RepresentationMode::NO_DISPLAY)
        {
            throw std::runtime_error(
                "Promotion of NO_AXON representation mode to " +
                lexical_cast<std::string>(mode) + " is not possible");
        }
        return neuron->getModels(); /* Nothing to do */

    case RepresentationMode::SOMA:
    {
        assert(neuron->getModels().size() == 1);
        NeuronModelPtr originalSoma = neuron->getModels()[0];
        assert(dynamic_cast<SphericalSomaModel*>(originalSoma.get()));
        NeuronModels models =
            Helpers::_createDetailedModels(neuron, scene, mode,
                                           originalSoma.get());
        return models;
    }

    case RepresentationMode::WHOLE_NEURON:
    case RepresentationMode::SEGMENT_SKELETON:
    default: /* to avoid the warning for NUM_REPRESENTATION_MODES */
        return neuron->getModels(); /* Nothing to do */
    }
}

void NeuronModel::applySceneClip(const CircuitScene& scene)
{
    const auto position = _neuron->getPosition();
    const auto orientation = _neuron->getOrientation();

    std::vector<osg::Vec4d> planes;
    for (osg::ClipNode::ClipPlaneList::const_iterator plane =
             scene.getClipPlaneList().begin();
         plane != scene.getClipPlaneList().end(); ++plane)
    {
        /* Transforming the plane to the skeleton reference system */
        osg::Matrix t =
            osg::Matrix::inverse(osg::Matrix::translate(-position) *
                                 osg::Matrix::rotate(orientation.inverse()));
        /* This is equivalent to p * t' */
        planes.push_back(t * (*plane)->getClipPlane());
    }

    clip(planes, scene.getAttributes().assumeUniqueMorphologies ? CLIP_HARD
                                                                : CLIP_SOFT);
}

void NeuronModel::setSimulationUniforms(const SimulationDataMapper& mapper)
{
    uint64_t offset = mapper.findAbsoluteOffset(*_neuron);
    setSimulationUniforms(offset);
}

void NeuronModel::setSimulationUniforms(const uint64_t offset)
{
    osg::Drawable* drawable = getDrawable();
    assert(drawable);

    /* Creating the uniforms that are common for all submodels, namely
       the cell index and the compartmental data buffer offset. */
    if (offset != LB_UNDEFINED_UINT64)
    {
        osg::Uniform* uniform =
            new osg::Uniform(osg::Uniform::INT,
                             "compartmentBufferObjectOffset");
        uniform->set((int)offset);
        drawable->getOrCreateStateSet()->addUniform(uniform);
    }
    /* And the index to access spike time data */
    osg::Uniform* uniform = new osg::Uniform(osg::Uniform::INT, "cellIndex");
    uniform->set((int)_neuron->getSimulationBufferIndex());
    drawable->getOrCreateStateSet()->addUniform(uniform);
}
}
}
}
