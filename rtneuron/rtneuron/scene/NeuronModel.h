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

#ifndef RTNEURON_NEURONMODEL_H
#define RTNEURON_NEURONMODEL_H

#include "../AttributeMap.h"
#include "../coreTypes.h"
#include "../types.h"

#include <osg/BoundingBox>
#include <osg/Referenced>
#include <osg/Vec4d>
#include <osg/ref_ptr>

#include <vector>

#include <cassert>

/** \cond DOXYGEN_IGNORE_COND */
namespace osg
{
class Group;
class Drawable;
}
/** \endcond */

/* The namespace is closed to workaround an issue with ccpcheck 1.63 */
namespace bbp
{
class Sections;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
class ConstructionData;
}

/**
   \brief Geometrical model of a neuron.

   A NeuronModel is an object that handles a particular geometrical
   representation of a neuron. This abstract class provides functions that
   deal with the inclusion of the model into a Scene and the update of the
   simulation data among others.

   A Neuron can be represented in a scene by a single NeuronModel (a detailed
   mesh) or a combination of models (e.g. mesh soma + tubelets).
 */
class NeuronModel : public osg::Referenced
{
public:
    /*--- Public declarations ---*/

    friend class LODNeuronModel;

    /*--- Public member functions ---*/

    /**
       Clear all cached data.

       Necessary to clean up OSG objects before reusing them in a different
       OpenGL contexts that have the same internal OSG id.
    */
    static void clearCaches();

    /**
       See Neuron::Neuron for the description of each attribute map.
    */
    static NeuronModels createNeuronModels(Neuron* neuron,
                                           const CircuitScene& scene);

    /**
       Creates a new list of neuron models which is an upgraded version
       of the current restricted mode of a neuron to a new restricted mode.

       The current implementation only supports upgrades from SOMA to
       NO_AXON or WHOLE_NEURON

       After upgrade the models need to be added to the scene as if returned
       by createNeuronModels.

       @see Neuron::getRestrictedRepresentationMode and
            Neuron::upgradeRepresentationMode
    */
    static NeuronModels upgradeModels(Neuron* neuron, const CircuitScene& scene,
                                      RepresentationMode mode);

    virtual bool isSomaOnlyModel() const = 0;

    virtual void setMaximumVisibleBranchOrder(unsigned int order) = 0;

    /**
       Setups GLSL input variables (e.g. vertex attributes) needed by
       shaders to access spike and compartment simulation buffers.

       Not to be used if simulation is applied as vertex attributes.
     */
    virtual void setupSimulationOffsetsAndDelays(
        const SimulationDataMapper& mapper, bool reuseCached) = 0;

    virtual void addToScene(CircuitScene& scene) = 0;

    virtual void applyStyle(const SceneStyle& style) = 0;

    /**
       Apply a user given soft un/clip.

       This clipping cannot override the scene clipping used for sort-last
       decomposition with spatial partitioning.
    */
    virtual void softClip(const NeuronModelClipping& operation) = 0;

    /**
       Returns the bounding box of this model.

       For models that support clipping, the bounding box returned does
       not consider it.
    */
    virtual osg::BoundingBox getInitialBound() const = 0;

    virtual void setColoring(const NeuronColoring& coloring) = 0;

    virtual void highlight(bool on) = 0;

    virtual void setRenderOrderHint(int /* order */) {}
    const Neuron* getNeuron() const { return _neuron; }
    virtual osg::Drawable* getDrawable() = 0;

    virtual void clearCircuitData(const CircuitScene&) {}
protected:
    /*--- Protected declarations ---*/

    typedef std::vector<osg::Vec4d> Planes;
    enum Clipping
    {
        CLIP_SOFT,
        CLIP_HARD
    };
    typedef std::pair<float, float> LODRange;

    /*--- Protected member attributes ---*/

    const Neuron* _neuron;

    /*--- Protected constructors/destructor ---*/

    NeuronModel(const Neuron* neuron);

    virtual ~NeuronModel();

    /*--- Protected member functions ---*/

    /**
       Called from addSubModel so this model can instantiate its sub model
       drawable (if any for the scene attributes).

       This method can perform any other function needed before adding the
       drawble to the LODNeuronModel.
    */

    virtual void setupAsSubModel(CircuitScene& scene) = 0;

    /**
       Gets the clipping planes from the scene add calls the proper clip type
       based on the scene options.
     */
    void applySceneClip(const CircuitScene& scene);

    /**
        Applies a clip plane to the model to be used inside a subcircuit scene.

        When supported, this function will reduce the internal geometry with
        the given planes. Derived classes can assume that at rendering time
        the planes will also be OpenGL clip planes.
        The visible part of the scene is in the intersection of the positive
        semispace (i.e positive distance to the plane) defined by each plane.
        The clip plane is in model local coordinates.

        @param planes
        @param type Whether the models should be soft clipped (they are shared)
                    or the geometry outside the clipped region can be removed.
    */
    virtual void clip(const Planes& planes, const Clipping type) = 0;

    /**
       Creates the GLSL uniforms required for simulation display for this
       model.

       Takes the drawable returned by getDrawable and adds to it a state set
       with the uniforms variables with the cell index and the absolute
       simulation offset.

       In order for this to work, the state set must be unique for the
       drawable object. This is the case when no LODs are used because
       the common state set is assigned to the scene group and the unique
       state set for the drawable is created from here.
       When LODs are used, these uniforms go in the LOD drawable which
       also has a unique state set per instance.
     */
    void setSimulationUniforms(const SimulationDataMapper& mapper);

    /**
       An overload of the function above in which the absolute offset is
       already provided instead of looked upon in the mapper.
     */
    void setSimulationUniforms(const uint64_t offset);

private:
    class Helpers;
};
}
}
}
#endif
