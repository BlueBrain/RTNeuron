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

#ifndef RTNEURON_API_SCENE_IMPL_H
#define RTNEURON_API_SCENE_IMPL_H

#include "InitData.h"
#include "Scene.h"
#include "coreTypes.h"
#include "detail/Configurable.h"
#include "types.h"

#include "data/Neurons.h"
#include "scene/CircuitSceneAttributes.h"
#include "viewer/osgEq/Scene.h"

#include <osg/BoundingSphere>

#include <eq/fabric/range.h>

#include <lunchbox/monitor.h>
#include <mutex>

namespace osg
{
class Node;
class Group;
class DrawElemetsUInt;
class Array;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Config;
}

class InitData;

using Planes = std::vector<Vector4f>;
using vmml::Rayf;
/*
  Scene::_Impl
*/
class Scene::_Impl : public detail::Configurable, public osgEq::Scene
{
public:
    /*--- Public declarations ---*/

    friend class rtneuron::Scene;
    friend class core::ModelObject;
    friend class core::NeuronObject;
    friend class core::SceneOperations;
    friend class core::SubScene;
    friend class core::SynapseObject;

    /* Made public because boost::serialization needs to access the types
       for registration */
    class Clear;
    class ClearClipPlanes;
    class HighlightCells;
    class StyleAttributeUpdate;
    class UpdateClipPlane;

    /**
       Base abstract class for all master and client only object handlers.
     */
    class BaseObject
    {
    public:
        /*--- Public constructors ---*/
        BaseObject(_Impl* parent);

        BaseObject(_Impl* parent, unsigned int id);

        virtual ~BaseObject() {}
        /*--- Public member functions ---*/
        unsigned int getID() const { return _id; }
        void invalidate();

        /**
           Called from Scene::_Impl::remove to clean the master scene from
           leftovers besides this object itself.
        */
        virtual void cleanup() = 0;

    protected:
        /*--- Protected member attributes ---*/
        mutable std::mutex _parentLock;
        _Impl* _parent;

        /*--- Protected member functions ---*/
        /** Called from invalidate */
        virtual void invalidateImplementation() {}
    private:
        /*--- Private member attributes ---*/
        const unsigned int _id;
    };
    using BaseObjectPtr = std::shared_ptr<BaseObject>;

    /*-- Public constructor/destructors ---*/

    /**
       Creates a Scene with a given configuration attributes.

       @param attributes See also CircuitScene::CircuitScene
    */
    _Impl(const AttributeMap& attributes, rtneuron::Scene* parent);

    ~_Impl();

    /*--- Public member functions ---*/

    static void setDefaultAfferentSynapseColor(const osg::Vec4& color);

    static void setDefaultEfferentSynapseColor(const osg::Vec4& color);

    //! @sa Scene::addNeurons
    rtneuron::Scene::ObjectPtr addNeurons(
        const GIDSet& gids, const AttributeMap& attributes = AttributeMap());

    //! @sa Scene::addEfferentSynapses
    rtneuron::Scene::ObjectPtr addEfferentSynapses(
        const brain::Synapses& synapses,
        const AttributeMap& attributes = AttributeMap());

    //! @sa Scene::addAfferentSynapses
    rtneuron::Scene::ObjectPtr addAfferentSynapses(
        const brain::Synapses& synapses,
        const AttributeMap& attributes = AttributeMap());

    //! @sa Scene::addModel
    rtneuron::Scene::ObjectPtr addModel(
        const char* filename, const Matrix4f& transform,
        const AttributeMap& attributes = AttributeMap());

    //! @sa Scene::addModel
    rtneuron::Scene::ObjectPtr addModel(
        const char* filename, const char* transform = "",
        const AttributeMap& attributes = AttributeMap());

    //! @sa Scene::addGeometry
    rtneuron::Scene::ObjectPtr addGeometry(
        const osg::ref_ptr<osg::Array>& vertices,
        const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
        const osg::ref_ptr<osg::Vec4Array>& colors,
        const osg::ref_ptr<osg::Vec3Array>& normals,
        const AttributeMap& attributes);

    //! @sa Scene::update
    void update();

    std::vector<rtneuron::Scene::ObjectPtr> getObjects();

    CircuitPtr getCircuit() const;

    void setCircuit(const CircuitPtr& circuit);

    /** Return and object handler by its id.

        This method will also search subset object handlers.
    */
    BaseObjectPtr getObjectByID(unsigned int id) const;

    /** Remove an object from the scene */
    void remove(const rtneuron::Scene::ObjectPtr& object);

    /** Remove an object by id.

        Intended to be used only from scene operations.

        Non thread-safe.
    */
    void removeObject(unsigned int id);

    /** Add an auxiliary object handler.

        This method is currently used to store the auxiliary handlers for
        NeuronObject subsets.
        Intended to be used only from scene operations.

        Non thread-safe.
    */
    void insertObject(const BaseObjectPtr& handler);

    void clear();

    void highlight(const GIDSet& target, bool on);

    void pick(const Rayf& ray) const;

    void pick(const Planes& planes) const;

    void setNeuronSelectionMask(const GIDSet& target);

    const GIDSet& getNeuronSelectionMask() const;

    const GIDSet& getHighlightedNeurons() const;

    const osg::BoundingSphere& getSomasBoundingSphere() const;

    const osg::BoundingSphere& getSynapsesBoundingSphere() const;

    osg::BoundingSphere getExtraModelsBoundingSphere() const;

    const core::CircuitSceneAttributes& getSceneAttributes() const
    {
        return *_sceneAttributes;
    }

    void setClipPlane(unsigned int index, const Vector4f& plane);

    const Vector4f& getClipPlane(unsigned int index) const;

    void clearClipPlanes();

    void setSimulation(const CompartmentReportPtr& report);

    void setSimulation(const SpikeReportReaderPtr& report);

    void mapSimulation(const uint32_t frameNumber, const float millisecond);

    const CompartmentReportPtr& getCompartmentReport() const;

    const core::SpikeReportPtr& getSpikeReport() const;

    /**
       Returns the number of subscenes that need to be updated.

       This is the numbers that onSimulationUpdated will be called at the
       parent scene.
    */
    unsigned int prepareSimulation(const uint32_t frameNumber,
                                   const float millisecond);

    /**
       Push an operation to the master SceneOperations queue.

       The operation queue is commited by commit() and synched by rendering
       clients in nodeSync().
     */
    void pushOperation(const core::SceneOperationPtr& operation);

    /**
       Invokes the dirty signal of the parent Scene.
     */
    void dirty(bool recomputeHome);

    /* Distribution methods */

    /**
       Register distributed objects used in scene modification.

       Stores the config pointer internally, the caller is responsible of
       calling deregisterDistributedObjects before the osgEq::Config is
       destroyed.
       Objects are deregistered at scene destruction.
    */
    void registerDistributedObjects(osgEq::Config* config, InitData& initData);

    void deregisterDistributedObjects(InitData& initData);

    /**
       Map distributed objects used in scene modification.

       Stores the config pointer internally, the caller is responsible of
       calling unmapDistributedObjects before the osgEq::Config is destroyed.
       Objects are unmapped at scene destruction.
    */
    void mapDistributedObjects(osgEq::Config* config);

    void unmapDistributedObjects();

    /**
       Commits scene operations and stores the commit IDs of the
       distributed objects in the frame data.

       To be called only at the application node before starting a new frame.
    */
    void commit(osgEq::FrameData& frameData);

    /* Methods inherited from osgEq::Scene */

    unsigned int getID() const { return _id; }
    /**
       Create a new SubScene object, store it internally and return its
       root node.

       The scenegraph is initially empty. Next call to updatePendingSubScenes
       will create the content (this is done during Node::frameStart).
    */
    osg::ref_ptr<osg::Node> getOrCreateSubSceneNode(
        const eq::fabric::Range& range) final;

    unsigned int computeCompositingPosition(
        const osg::Matrix& modelView,
        const eq::fabric::Range& range) const final;

    void updateCameraData(const eq::fabric::Range& range,
                          osg::Camera* camera) final;

    /**
       Update the scene to the context to render at a given frame.

       Creates the CircuitScene object for each new sub scene if necessary.
       This method may process each sub scene in parallel.
       Runs in mutual exclusion with _Impl::addNeurons,
       _Impl::addEfferentSynapses, _Impl::addAfferentSynapses,
       _Impl::addModel and _Impl::remove
    */
    void nodeSync(const uint32_t frameNumber,
                  const osgEq::FrameData& frameData) final;

    /**
       At the moment this function swaps the simulation buffers if new
       simulation data is ready.
    */
    void channelSync(const eq::fabric::Range& range, const uint32_t frameNumber,
                     const osgEq::FrameData& frameData) final;

    /**
        This is used to compute the ROI.
        It uses the knowledge about the circuit and the spatial partition
        to provide a tigher bound than the bounding sphere of the
        scenegraph root.
    */
    osg::BoundingBox getSceneBoundingBox(
        const eq::fabric::Range& range) const final;

    /**
       Returns the right compositing technique for sort-last configurations
       to be used for this scene.

       The value returned depends on the scene partition and the alpha-blending
       algorithm used.
    */
    DBCompositingTechnique getDBCompositingTechnique() const final;

    osgTransparency::BaseRenderBin* getAlphaBlendedRenderBin() final;

    /**
       Adds a color map to the color map atlas.

       If the colormap is modified the atlas texture will be updated
       automatically.
       The atlas keeps track of the colormaps using weak pointers. In order
       to remove the colormap from the atlas it is enough to destroy the
       object. The atlas texture won't be updated immediately, but its
       slot will be freed.

       The color map is assigned an altas index.
    */
    void addColorMapToAtlas(const ColorMapPtr& colormap);

protected:
    /*--- Protected member functions ---*/

    void onAttributeChangingImpl(
        const AttributeMap& map, const std::string& name,
        const AttributeMap::AttributeProxy& parameters) final;

    void onAttributeChangedImpl(const AttributeMap& map,
                                const std::string& name) final;

private:
    /*--- Private declarations ---*/

    using IDSet = std::set<unsigned int>;

    struct SubSceneData
    {
        SubSceneData()
            : version(0)
        {
        }

        core::SubScenePtr subScene;
        /** The DB range of the subscene. */
        eq::fabric::Range range;
        /* Identifiers of the objects present in the subscene. */
        IDSet objects;
        /* This version number refers to static elements not to simulation
           data. */
        size_t version;
    };

    using SubSceneMap = std::map<eq::fabric::Range, SubSceneData>;

    using NeuronObjects = std::map<unsigned int, core::NeuronObjectPtr>;
    using SynapseObjects = std::map<unsigned int, core::SynapseObjectPtr>;
    using ModelObjects = std::map<unsigned int, core::ModelObjectPtr>;
    using GeometryObjects = std::map<unsigned int, core::GeometryObjectPtr>;

    using ClientOnlyObjects = std::map<unsigned int, BaseObjectPtr>;

    template <typename T>
    struct Remover;

    /*--- Private member attributes ---*/
    unsigned int _id;

    mutable std::mutex _lock;
    lunchbox::Monitor<int> _parentPointerMonitor;
    rtneuron::Scene* _parent; /* Used to emit the signals */

    osgEq::Config* _config;

    SubSceneMap _subScenes;

    /* This is the master copy of the object handling the distributable
       scene operations. */
    core::SceneOperations* _masterSceneOperations;
    core::SceneOperations* _sceneOperations;
    std::vector<co::uint128_t> _pendingOperationUpdates;

    /* There is a corner case in which we want to be able to push operations
       while the GIL needs to stay locked, which is when a NeuronObject::SubSet
       is destroyed from Python. This lock prevents potential deadlocks by
       reducing the scope of the critical regions (compared to using _lock). */
    mutable std::mutex _operationsLock;

    CircuitPtr _circuit;

    NeuronObjects _neuronObjects;
    SynapseObjects _synapseObjects;
    GeometryObjects _geometryObjects;
    ModelObjects _modelObjects;

    /* Becomes true when an object is removed until nodeSync is called. */
    bool _subscenesNeedReset;

    /* Container for auxiliary objects used only to carry out certain
       operations (like subset handler updates). */
    ClientOnlyObjects _clientObjects;

    core::Neurons _neurons;

    GIDSet _highlightedNeurons;
    GIDSet _maskedNeurons;

    osg::ref_ptr<osg::Group> _extraModels;
    ModelObjects _toAdd;
    using Nodes = std::vector<osg::ref_ptr<osg::Node>>;
    Nodes _toRemove;
    mutable osg::BoundingSphere _somasBound;
    mutable osg::BoundingSphere _synapsesBound;

    std::vector<std::pair<Vector4f, bool>> _userClipPlanes;

    core::SpikeReportPtr _spikeReport;
    CompartmentReportPtr _compartmentReport;

    /* Initialized from the attribute map passed at construction.
       This attributes will be common to all subscenes contained. */
    core::CircuitSceneAttributesPtr _sceneAttributes;

    AttributeMap _styleAttributeUpdates;
    /* Used for objects outside the subscenes. */
    core::SceneStylePtr _sceneStyle;
    /* ColorMap atlas used for all states from SceneStyle */
    core::ColorMapAtlasPtr _colorMaps;

    size_t _version;

    /*--- Private member functions ---*/

    /** Try to remove an object.

        @return True if the object was found and removed.
    */
    bool _removeObject(unsigned int id);

    void _clearSubScenes();

    void _syncSubScene(SubSceneData& data,
                       const std::vector<co::uint128_t>& pendingOps);

    bool _subSceneNeedsResetting(const SubSceneData& data);

    core::SubScenePtr _createSubScene();

    void _updateSubScene(SubSceneData& data);

    core::Neurons _computeTargetSubsets(core::SubScene& subScene,
                                        const eq::fabric::Range& range);

    void _addNeuronsToSubScene(SubSceneData& data, const core::Neurons* subset);

    void _addSynapsesToSubScene(SubSceneData& data);

    /** Unsynched version of unmapDistributedObjects */
    void _unmapDistributedObjects();

    /** Called internally as many times as the number of subscenes contained
        internally. */
    void _onSimulationUpdated(float timestamp);

    /**
       Sets the parent pointer to 0.
       Thread safe with regard to the pieces of code that use the parent
       pointer to emit signals.
    */
    void _invalidateParent();

    void _calculateSomasBoundingSphere() const;
    void _calculateSynapsesBoundingSphere() const;
};
}
}
#endif
