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

#ifndef RTNEURON_CIRCUITSCENE_H
#define RTNEURON_CIRCUITSCENE_H

#include "coreTypes.h"

#include "CircuitSceneAttributes.h"
#include "SphereSet.h"

#include "data/Neuron.h"
#include "data/Neurons.h"
#include "scene/models/Cache.h"

#include <osg/ClipNode>

#include <OpenThreads/Mutex>

namespace osg
{
class RenderInfo;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   \brief Class that holds the scenegraph elements of a circuit

   A one-to-one mapping between osgEq::Channel and CircuitScene objects, and
   oeqEq::Channel and eq::Pipe objects is expected, otherwise more GPU/CPU
   memory than needed will be used.

   This class is not thread-safe.
*/
class CircuitScene
{
public:
    /*--- Public constructors/destructor ---*/

    /**
       Creates a CircuitScene with a given configuration.
       @param attributes Static attributes that define the characteristics
       of the scene (culling, lod config, ...). These attributes may affect
       the properties of the models for this scene and how they must be
       created. Because of that, models created for one scene may not be used
       for a different one.
     */
    CircuitScene(const CircuitSceneAttributesPtr& attributes);

    virtual ~CircuitScene();

    /*--- Public member functions ---*/

    /**
       Returns the identifier of this scene.

       Identifiers are assigned sequentially starting from 0 as
       CircuitScenes are created. The purpose of an identifier is to be
       used to index scene specific data of a NeuronModel object.
     */
    size_t getID() const { return _identifier; }
    /** Return the node with the geometry of all neuron models except
        spherical somas */
    osg::Group* getNeuronModelNode();

    std::vector<osg::Node*> getNodes();

    const osg::ClipNode::ClipPlaneList& getClipPlaneList() const
    {
        return _circuit->getClipPlaneList();
    }

    /** Return true if this scene has static clipping applied.

        There are two kinds of scene clippings, the static one is that one
        that won't change once applied and is the case of spatial partitions.
        Dynamic clipping refers to user given clipping, which is incompatible
        with spatial partitions. */
    bool isClippingStatic() const { return _staticClipping; }
    /**
       Returns the axis aligned bounding box of the circuit
    */
    const osg::BoundingBox& getBoundingBox() const
    {
        return _circuitBoundingBox;
    }

    void addNeurons(const Neurons& neurons);

    SphereSet& getSynapses();

    SphereSet& getSomas();

    void applyStyle(const SceneStylePtr& style);

    const SceneStyle& getStyle() const { return *_style; }
    /** Return the cache of model data specific for this circuit scene. */
    const model::Cache& getModelCache() const { return _modelCache; }
    void clear();

    /**
       Returns the static options parsed from the attribute map passed to
       the constructor.
    */
    const CircuitSceneAttributes& getAttributes() const { return *_attributes; }
    const Neurons& getNeurons() const { return _neurons; }
    Neurons& getNeurons() { return _neurons; }
    /**
       Adds a neuron model of a specific type to the scene.

       The scene style is already applied to parent node where the node will
       be added.
     */
    void addNeuronModel(osg::Node* node, NeuronLOD lod);

    /**
       Returns the geode that contains the geometry object to render spherical
       soma models.

       If needed, will create the geode and the geometry using the functor
       given.
     */
    typedef boost::function<osg::ref_ptr<osg::Geode>()> GeodeConstructor;
    osg::Geode* getOrCreateSphericalSomaGeode(const GeodeConstructor& functor);

    /**
       Adds a neuron model with no specific type to the scene.

       The scene style must be handled by the node added. This is the function
       to use for LOD models.
    */
    void addNeuronModel(osg::Node* node);

    /**
       Add a node to the group of auxiliary objects.

       @param node Node to add
       @param columnHint Id to chose a subgroup where the node should be placed
       @param unmaskGroup By default, the auxiliary groups are masked for
              cull and draw. Unmasking is carried out by external objects that
              want to make the subgroup visible. This approach tries to solve
              the performance problems caused by the placeholder nodes used for
              text labels and morphologies with large circuits. This is
              mainly a workaround because labels will eventually be removed
              from the circuit scene (and core rendering) and moved to the
              overlay and probably the node devoted to skeletal
              representations can be removed.
    */
    void addNode(osg::Node* node,
                 uint32_t groupHint = std::numeric_limits<uint32_t>::max(),
                 bool unmaskGroup = false);

protected:
    /*--- Protected member attributes ---*/

    size_t _identifier;

    Neurons _neurons;

    /* Scene elements */
    osg::ref_ptr<osg::ClipNode> _circuit;

    SphereSet _synapses;
    SphereSet _somas;

    osg::ref_ptr<osg::Group> _extraObjects;
    std::unordered_map<uint32_t, osg::ref_ptr<osg::Group>> _extraObjectGroups;

    osg::BoundingBox _circuitBoundingBox;
    bool _staticClipping;

    typedef std::map<NeuronLOD, osg::Group*> NeuronModelMap;
    typedef std::map<NeuronLOD, osg::Group*>::const_iterator NeuronModelMapIter;
    NeuronModelMap _neuronModels;

    model::Cache _modelCache;

    /* Initialized from the attribute map passed at construction */
    CircuitSceneAttributesPtr _attributes;

    SceneStylePtr _style;

    /*--- Protected member functions ---*/

    virtual void reportProgress(const std::string& message, size_t count,
                                size_t total) = 0;

private:
    /*--- Private declarations ---*/
    class CompileBuffersCallback;

    /*--- Private member attributes ---*/
    OpenThreads::Mutex _mutex;

    /*--- Private member functions ---*/

    osg::Group* _getOrCreateSceneTypeGroup(NeuronLOD type);

    void _updateCircuitBoundingBox(const Neuron& neuron);
};
}
}
}
#endif
