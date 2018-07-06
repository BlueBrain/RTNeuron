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

#ifndef RTNEURON_NEURON_H
#define RTNEURON_NEURON_H

#include "coreTypes.h"
#include "types.h"

#include "render/NeuronColoring.h"

#include <brain/neuron/types.h>

#include <OpenThreads/Mutex>
#include <osg/Quat>
#include <osg/Referenced>
#include <osg/Vec4>
#include <osg/ref_ptr>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

namespace osg
{
class Node;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! Contains the data and geometrical models of a neuron from a Scene.
class Neuron
{
public:
    /*--- Public declarations ---*/

    friend class Neurons;

    typedef boost::function<void(NeuronModel&)> ModelOperation;

    /*--- Public constructors/destructor ---*/

    /**
       Create a neuron from a circuit with a given gid.

       The representation mode of the neuron is initially set to NO_DISPLAY.

       @param gid The gid of the neuron
       @param circuit The circuit from this this neuron comes from
    */
    Neuron(uint32_t gid, const CircuitCachePtr& circuit);

    /**
       Creates a neuron object that is actually renderable.

       To be created only at the SubScene level and not at the SceneImpl.
    */
    Neuron(const Neuron& other, const CircuitSceneAttributes& attributes,
           RepresentationMode mode);

    ~Neuron();

    /*--- Public member functions ---*/

    /* Attribute accessors */

    uint32_t getGID() const { return _gid; }
    const osg::Vec3& getPosition() const;

    const osg::Quat& getOrientation() const;

    std::string getMorphologyLabel() const;

    std::string getMorphologyType() const;

    std::string getElectrophysiologyType() const;

    /** Get the morphology of this neuron, load if necessary. */
    brain::neuron::MorphologyPtr getMorphology() const;

    /* RTNeuron internal */

    const CircuitCachePtr& getCircuit() const { return _circuit; }

    /** Load mesh and/or morphology according to the neuron representation mode
        and circuit scene attributes given.
        The given mode is taken as the mode that will be used when addToScene
        is called.
    */
    void init(const CircuitSceneAttributes& attributes,
              RepresentationMode mode);

    /** Get the mesh of this neuron, load and/or generate if necessary.
        Mesh generation is only available if compied with NeuMesh support.
        @param attributes The scene attributes. Used  to find out the mesh path
        and decide whether the mesh needs to be generated if missing.
        @return A pointer to the mesh or a null pointer if the mesh cannot be
        loaded (e.g. it doesn't exist).
    */
    NeuronMeshPtr getMesh(const CircuitSceneAttributes& attributes) const;

    /** Return the soma radius to be used for rendering.
        If the morphology of the neuron is already loaded, return the mean
        soma radius from the morphology multiplied by
        SOMA_MEAN_TO_RENDER_RADIUS_RATIO, otherwise, use the morphology type
        to return the estimated average radius for that type. */
    float getSomaRenderRadius() const;

    static void clearCaches();

    size_t getSimulationBufferIndex() const;
    void setSimulationBufferIndex(const size_t index);
    /**
       Prepare all internal models to use a global texture with simulation
       data for simulation update and display.

       @param mapper The simulation data mapper with the mapping
       @param reuseCached True when the mapping and simulation reports of the
                          mapper haven't changed. In this case the neuron
                          model is requested to compute the offsets for models
                          that don't have them yet (this happens during
                          representation mode upgrades), other models should
                          avoid recomputing anything.
    */
    void setupSimulationOffsetsAndDelays(const SimulationDataMapper& mapper,
                                         bool resuseCached);

    /**
       Adds the models, synapses and GUI elements of the neuron to the
       given scene.

       The scene style is also applied to the neuron and operations that
       were postponed are applied after instantiating and adding the models
       to the scene.
     */
    void addToScene(CircuitScene& scene);

    void applyStyle(SceneStyle& style);

    void setMaximumVisibleBranchOrder(unsigned int order);

    /**
       Sets the coloring mode to be used for rendering this neuron.

       If the models have not been instantiated yet, the coloring is
       copied to be applied when addToScene is called.
       @param coloring The coloring scheme.
     */
    void applyColoring(const NeuronColoring& coloring);

    void highlight(bool on);

    /**
       Apply a per model operation.

       If the models have not been instantiated yet, the model operation is
       copied and postponed until addToScene is called.
       @param A copy constructible function object of type ModelOperation
     */
    void applyPerModelOperation(const ModelOperation& operation);

    /**
       Query the bounding box in world coordinates.

       Returns true if the bounding box could be determined, and false
       otherwise (e.g. no geometrical data available).
       The bounding box coordinates are computed translating and rotating
       the local bounding box. No submodel clipping is considered in the
       bounding box computation.
     */
    bool getWorldCoordsExtents(double& xMin, double& xMax, double& yMin,
                               double& yMax, double& zMin, double& zMax) const;

    /**
       Return the restricted representation mode.

       The restricted mode is the representation mode passed at construction
       or upgrade. The restricted mode constrains which representation mode
       are available without an upgrade.

       The available modes for each restricted mode are:
       - SOMA: NO_DISPLAY or SOMA
       - NO_AXON: NO_DISPLAY, SOMA or NO_AXON
       - WHOLE_NEURON and SEGMENT_SKELETON: all modes

       @see upgradeRepresentationMode()
    */
    RepresentationMode getRestrictedRepresentationMode() const;

    /**
       Return the current representation mode.
    */
    RepresentationMode getRepresentationMode() const;

    RepresentationMode getRepresentationMode(unsigned long frameNumber) const;

    void setRepresentationMode(RepresentationMode mode);

    /**
       Upgrades the current restricted mode to a new one if possible.
       Throws if the upgrade is not possible.

       The only allowed transition at the moment is from SOMA to NO_AXON or
       WHOLE_NEURON. Notice specially that NO_AXON cannot be upgrades to
       WHOLE_NEURON.
    */
    void upgradeRepresentationMode(RepresentationMode mode,
                                   CircuitScene& scene);

    const NeuronModels& getModels() const;

    /**
       Request neuron models to clear all their data which is
       specific to the given CircuitScene.

       This is called when the given CircuitScene is about to be destroyed.
       In particular, SphericalSomaModels remove their links to the CircuitData.
    */
    void clearCircuitData(const CircuitScene& scene);

    osg::Node* getExtraObjectNode();

private:
    /*-- Private member variables ---*/
    Neuron();

    uint32_t _gid;

    CircuitCachePtr _circuit;
    mutable brain::neuron::MorphologyPtr _morphology;
    mutable NeuronMeshPtr _mesh;

    osg::Vec3 _position;
    osg::Quat _orientation;
    uint32_t _mtype;
    uint32_t _etype;

    struct RenderData;
    std::unique_ptr<RenderData> _rd;
};
}
}
}
#endif
