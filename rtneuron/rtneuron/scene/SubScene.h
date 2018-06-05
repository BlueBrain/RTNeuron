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
//////////////////////////////////////////////////////////////////////

#ifndef RTNEURON_SCENE_SUBSCENE_H
#define RTNEURON_SCENE_SUBSCENE_H

#include "SceneOperations.h"
#include "SynapseObject.h"
#include "coreTypes.h"

#include "../SceneImpl.h"
#include "scene/CircuitScene.h"
#include "scene/SphereSet.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class SimulationRenderBuffer;
class SimulationDataMapper;

class SubScene : public CircuitScene
{
public:
    /*--- Public constructors/destructor */

    SubScene(Scene::_Impl* parent);

    ~SubScene();

    /*--- Public member functions ---*/

    osg::ref_ptr<osg::Group> getNode() { return _node; }
    /**
       \brief Removes from the given container those cells not to be rendered
       in the start, end range given.

       Depending on the decomposition mode used, this object will store
       additional information that will be used to later add or render the
       objects into the scene.
    */
    void computeSubtarget(Neurons& neurons, RepresentationMode mode,
                          float start, float end);

    void computeRoundRobinSubtarget(Neurons& neurons, RepresentationMode mode,
                                    float start, float end);

    void computeSpatialPartitionSubtarget(Neurons& neurons,
                                          RepresentationMode mode, float start,
                                          float end);

    unsigned int compositingPosition(const osg::Matrix& modelView,
                                     unsigned int nodeCount) const;

    void addNeurons(const Neurons& neurons, const AttributeMap& attributes);

    void addSynapses(const SynapseObject& synapses);

    core::SphereSet::SubSetID findSynapseSubset(
        const unsigned int objectID) const;

    void highlightCells(const GIDSet& cells, bool highlight);

    void setSimulation(const CompartmentReportPtr& report);

    void setSimulation(const SpikeReportPtr& report);

    core::SimulationDataMapper& getMapper() const { return *_simulationMapper; }
    /**
       Updates the offsets/delays of the neuron models from the given neuron.

       This is done depending on the availabitlity of compartment and spike
       reports.
    */
    void updateSimulationMapping(Neuron& neuron);

    /**
       Returns true is simulation update has been triggered in this subscene
       and the mapper needs to be taken into account for synching simulation
       display.
    */
    bool prepareSimulation(const uint32_t frameNumber,
                           const float milliseconds);

    void mapSimulation(const uint32_t frameNumber, const float milliseconds);

    /**
       Apply scene operations to this sub scene.

       To by done from node sync in SceneImpl
       @param commits List of SceneOperations commits.
    */
    void applyOperations(const std::vector<co::uint128_t>& commits);

    void channelSync();

    void mapDistributedObjects(osgEq::Config* config);

    void unmapDistributedObjects(osgEq::Config* config);

    void onSimulationUpdated(float timestamp);

    void setClipPlane(unsigned int index, const osg::Vec4f& plane);

    void clearClipPlanes();

    Scene::_Impl& getScene() { return *_parent; }
    const Scene::_Impl& getScene() const { return *_parent; }
    /* Made public to be callable from NeuronObject mode upgrades. */
    void reportProgress(const std::string& message, size_t count,
                        size_t total) final
    {
        ++_parent->_parentPointerMonitor;
        if (_parent->_parent)
            _parent->_parent->progress(message, count, total);
        --_parent->_parentPointerMonitor;
    }

private:
    /*--- Private member attributes ---*/
    SceneOperations _sceneOperations;

    Scene::_Impl* _parent;

    mutable std::mutex _mutex;
    osg::ref_ptr<osg::ClipNode> _node;

    std::vector<osg::Plane> _splittingPlanes;

    /* Simulation attributes */
    SimulationRenderBufferPtr _simulationBuffer;
    SimulationDataMapperPtr _simulationMapper;
    float _timestampReady;
    bool _doSwapSimBuffers;
    bool _needsMappingUpdate;

    typedef std::map<unsigned int, core::SphereSet::SubSetID> SynapseSphereSets;
    SynapseSphereSets _synapseSphereSets;

    /*--- Private member functions ---*/
    void _applyClipBox(const osg::BoundingBoxd& box);
};
}
}
}
#endif
