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

#include "SphericalSomaModel.h"

#include "config/Globals.h"
#include "config/constants.h"
#include "data/Neuron.h"
#include "data/SimulationDataMapper.h"
#include "scene/CircuitScene.h"
#include "scene/NeuronModelClipping.h"
#include "util/vec_to_vec.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
void _updateColorMapIndex(SphereSet& spheres, const SphereSet::SubSetID id,
                          const unsigned int attribute, const size_t index)
{
    if (index == std::numeric_limits<size_t>::max())
        spheres.updateAttribute(id, attribute, (char)0);
    else
        spheres.updateAttribute(id, attribute, (char)(index + 1));
}

/*
  Helper classes
*/

struct SimulationBuffers : public osg::Referenced
{
    osg::ref_ptr<osg::FloatArray> simulationBuffers[2];
    unsigned char backBuffer;
    size_t pending;
};
}

SphericalSomaModel::CircuitData::CircuitData()
    : _id(0)
{
}

SphericalSomaModel::CircuitData::~CircuitData()
{
}

/*
  Constructor
*/
SphericalSomaModel::SphericalSomaModel(const Neuron* neuron)
    : NeuronModel(neuron)
    , _radius(neuron->getSomaRenderRadius())
    , _clipped(false)
{
}

/*
  Member functions
*/
void SphericalSomaModel::setupSimulationOffsetsAndDelays(
    const SimulationDataMapper& mapper, const bool /* reuseCached */)
{
    /* For spherical somas the simulation offsets are absolute */
    uint64_t offset = mapper.getSomaSimulationOffset(*_neuron);
    float offsetF = offset;
    if (offset == LB_UNDEFINED_UINT64)
    {
        if (mapper.getSpikeReport())
        {
            /* No compartmental data source available. Assuming we are
               going to render spikes only.
               The buffer index must be converted to a signed integer
               before subtraction. */
            offsetF = _neuron->getSimulationBufferIndex();
        }
        else
        {
            /* This cell is not present in the simulation report. */
            /** \todo Tag it for rendering with some color meaning that its
                values are undefined. The uniform with the absolute offset
                may need to be removed if previously set.
                Related ticket:
                https://bbpteam.epfl.ch/project/issues/browse/BBPRTN-217
            */
        }
    }

    for (CircuitData& data : _circuitData)
    {
        if (!data._id)
            continue;
        data._spheres->updateAttribute(data._id, CELL_INDEX_GLATTRIB, offsetF);
    }
}

void SphericalSomaModel::addToScene(CircuitScene& scene)
{
    /* Checking if the neuron has already been added. No mutex needed here. */
    CircuitData& data = _circuitData[scene.getID()];
    if (data._id)
        return;

    const osg::Vec3& position = _neuron->getPosition();

    /* Securing concurrent access to the same sphere set from
       CircuitScene::addNeuronList */
    SphereSet& spheres = scene.getSomas();
    /* We don't care about the actual color given here because
       Neuron::addToScene is going to set the color after adding the models. */
    data._id = spheres.addSphere(position, _radius, osg::Vec4(1, 1, 1, 1));
    /* The lifetime of this object is bound to the scene, so using a raw
       pointer is now a problem. We want to avoid smart pointers for
       perfomance and memory usage reasons. */
    data._spheres = &spheres;

    /* Initializing vertex attributes.
       All arrays are preset now because resizing from here
       is safe whereas doing that from other threads is not (as in
       setupSimulationOffsetsAndDelays from the simulation mapping thread
       for example). */
    data._spheres->updateAttribute(data._id, HIGHLIGHTED_SOMA_ATTRIB_NUM,
                                   (char)false);

    /* OSG < 3.2 uses glVertexAttribPointer instead of glVertexAttribIPointer
       for all vertex attributes. That means that double and integer arrays
       are converted into float and must be used as floats inside GLSL.
       @sa BBPRTN-379. */
    data._spheres->updateAttribute(data._id, CELL_INDEX_GLATTRIB, 0.0f);

    /* It's desirable to use a single index for both simulation colormaps to
       save space, but then it's impossible to do a correct implementation of
       changes of the Scene attribute show_soma_spikes.
       Having two attributes we only need to recompile the shader (because the
       code path depends on macros not uniforms). Having one would require
       to re-apply the coloring scheme to all spherical soma models to update
       the color map index. However, coloring schemes from temporary scene
       handlers (i.e. returned by NeuronObject::query) may have been lost,
       making it impossible to update the models consistently. */
    data._spheres->updateAttribute(data._id,
                                   COMPARTMENT_COLOR_MAP_INDEX_ATTRIB_NUM,
                                   (char)0);
    data._spheres->updateAttribute(data._id, SPIKE_COLOR_MAP_INDEX_ATTRIB_NUM,
                                   (char)0);
}

void SphericalSomaModel::setupAsSubModel(CircuitScene& scene)
{
    addToScene(scene);
}

void SphericalSomaModel::applyStyle(const SceneStyle&)
{
    /* In principle there's no need to do anything here because the state set
       for this models will be updated at the scene level. */
}

void SphericalSomaModel::softClip(const NeuronModelClipping& operation)
{
    switch (operation.getSomaClipping())
    {
    case SomaClipping::CLIP:
    {
        _clipped = true;
        /* For non submodel spheres we need to update the radius array right
           away to make the sphere invisible. */
        for (CircuitData& data : _circuitData)
        {
            if (!data._id)
                continue;
            data._spheres->updateRadius(data._id, -_radius);
        }
        break;
    }
    case SomaClipping::UNCLIP:
    {
        _clipped = false;
        /* For non submodel spheres we need to update the radius array right
           away to make the sphere visible. */
        for (CircuitData& data : _circuitData)
        {
            if (!data._id)
                continue;
            data._spheres->updateRadius(data._id, _radius);
        }
        break;
    }
    default:
        /* Nothing to do here, leaving it as it is */
        ;
    }
}

osg::BoundingBox SphericalSomaModel::getInitialBound() const
{
    osg::BoundingBox bb;
    bb.expandBy(osg::BoundingSphere(osg::Vec3(), _radius));
    return bb;
}

void SphericalSomaModel::setColoring(const NeuronColoring& coloring)
{
    bool allEmpty = true;
    for (CircuitData& data : _circuitData)
        allEmpty &= data._id == 0;

    /* Can't assign colors to somas before they are added to the scene.
       Neuron::addToScene will assign colors at the moment the neuron
       is added to the scene. */
    if (allEmpty)
        return;

    const osg::Vec4 color = coloring.getSomaBaseColor(*_neuron);

    for (CircuitData& data : _circuitData)
    {
        if (!data._id)
            continue;

        data._spheres->updateColor(data._id, color);

        _updateColorMapIndex(*data._spheres, data._id,
                             COMPARTMENT_COLOR_MAP_INDEX_ATTRIB_NUM,
                             coloring.getCompartmentColorMapIndex());
        _updateColorMapIndex(*data._spheres, data._id,
                             SPIKE_COLOR_MAP_INDEX_ATTRIB_NUM,
                             coloring.getSpikeColorMapIndex());
    }
}

void SphericalSomaModel::highlight(const bool on)
{
    for (CircuitData& data : _circuitData)
    {
        if (!data._id)
            continue;

        data._spheres->updateAttribute(data._id, HIGHLIGHTED_SOMA_ATTRIB_NUM,
                                       (char)on);
    }
}

void SphericalSomaModel::setVisibility(const uint32_t circuitSceneID,
                                       const bool visible)
{
    CircuitData& data = _circuitData[circuitSceneID];
    if (!data._id)
        return;

    data._spheres->updateRadius(data._id,
                                visible && !_clipped ? _radius : -_radius);
}

void SphericalSomaModel::clearCircuitData(const CircuitScene& scene)
{
    _circuitData[scene.getID()] = CircuitData();
}
}
}
}
