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

#ifndef RTNEURON_API_SCENE_H
#define RTNEURON_API_SCENE_H

#include "AttributeMap.h"
#include "View.h"
#include "types.h"

// This headers leak an OSG dependency that should be avoided if possible.
#include <osg/Array>
#include <osg/PrimitiveSet>

#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signals2/signal.hpp>

#include <memory>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Scene;
class Config;
}

// clanf-format spoils the indentation of comments before #ifdef
// clang-format off

/**
   The scene to be rendered by one or more views.

   \anchor Scene_details
   A scene contains the circuit elements to be displayed, additional mesh
   models and is associated with simulation data (this may be moved to
   the View class in the future).

   The attributes that can be passed to RTNeuron::createScene are the
   following:
   - *accurate_headlight* (bool):
     Apply shading assuming directional light rays from the camera position
     or parallel to the projection plane.
   - *alpha_blending* (AttributeMap):
     If provided, transparent rendering will be enabled in this scene.
     The attributes to configure the alpha-blending algorithm are:
     - *mode* (string):
       depth_peeling, multilayer_depth_peeling or fragment_linked_list if
       compiled with OpenGL 3 support.
     - *max_passes* (string):
       Maximum number of rendering passes for multipass algorithms.
     - *cutoff_samples* (string):
       In multipass algorithms, the number of samples returned by the
       occlusion query at which the frame can be considered finished.
     - *slices* (int) [only for multi-layer depth peeling]:
       Number of slices to use in the per-pixel depth partition of the scene.
     If the input attribute map is empty, transparent rendering will be
     disabled.
   - *circuit* (string):
     URI of the circuit to use for this scene.
   - *mesh_path* (string):
     Path where neuron meshes are located for the given circuit.
     This path will be try to be inferred for circuits described by
     Circuit/BlueConfig.
   - *connect_first_order_branches* (bool):
     Translate the start point of first order branches to connect them to
     the soma (detailed or spherical depending on the case).
   - *em_shading* (bool):
     Choose between regular phong or fake electron microscopy shading.
   - *inflatable_neurons* (bool):
     If true the neuron models can be inflated by displacing the membrame
     surface in the normal direction. The inflation factor is specified
     as a view attribute called *inflation_factor*.
   - *load_morphologyes* (bool):
     Whether load morphologies for calculating soma radii or not.
   - *lod* (AttributeMap): Level of detail options for different types of
     objects
     - *neurons* (AttributeMap):
       Each attribute is a pair of floats between [0, 1] indicating the
       relative range in which a particular level of detail is used.
       Attribute names refer to levels of detail and they can be: *mesh*,
   *high_detail_cylinders*, *low_detail_cylinders*, *tubelets*, *detailed_soma*,
   *spherical_soma*
   - *mesh_based_partition* (bool):
     Use the meshes for load balancing spatial partitions. Otherwise only
     the morpholgies are used. This options requires use_meshes to be also
     true.
   - *partitioning* (DataBasePartitioning):
     The type of decomposition to use for DB (sort-last) partitions.
   - *preload_skeletons* (bool):
     Preload all the capsule skeletons used for view frustum culling into the
     GPU instead of doing the first type they are visible.
   - *unique_morphologies* (bool)
     If true, enables optimizations in spatial partitions that are only
     possible assuming that morphologies are unique.
   - *use_cuda* (bool):
     Enable CUDA based view frustums culling.
   - *use_meshes* (bool):
     Whether triangular meshes should be used for neurons or not.


   Scenes must be created using RTNeuron::createScene before the Equalizer
   configuration is started. At creation time scenes are assigned an
   internal ID. In multi-procress configurations (be it in the same machine
   or not), the scenes to used must be created in the same order to ensure
   the consistency of the frames.

   At this moment, scene changes are not propagated from the application
   process to the rendering clients.
*/
class Scene
{
public:
    /*--- Public declarations ---*/
    friend class RTNeuron; /* To call _invalidate */
    friend class View;

#define DECLARE_SIGNAL(prefix, ...)                    \
    typedef void prefix##SignalSignature(__VA_ARGS__); \
    typedef boost::signals2::signal<prefix##SignalSignature> prefix##Signal;

    DECLARE_SIGNAL(CellSelected, uint32_t, uint16_t, uint16_t);
    DECLARE_SIGNAL(CellSetSelected, const GIDSet&);
    DECLARE_SIGNAL(SynapseSelected, const brain::Synapse&)

    DECLARE_SIGNAL(Progress, const std::string&, size_t, size_t);

    /* The parameter is true if the home position needs to be recomputed
       and false otherwise */
    DECLARE_SIGNAL(Dirty, bool);

    DECLARE_SIGNAL(SimulationUpdated, float);

#undef DECLARE_SIGNAL

    class Object;

    /**
       Base class to encapsulate operations that can be performed on scene
       objects.
     */
    class ObjectOperation
    {
    public:
        friend class Object;

        /*--- Public member functions ---*/

        /**
           Called from Object::apply before the operation is queued.

           @return True iff this operation can be applied in the scene
           object provided.
           @internal
        */
        virtual bool accept(const Scene::Object& object) const = 0;

    protected:
        /*--- Protected declarations ---*/

        /** @internal */
        class Impl;

        /*--- Protected constructors/destructor ---*/

        virtual ~ObjectOperation();

    public:
        /** @internal */
        virtual std::shared_ptr<Impl> getImplementation() = 0;
    };
    typedef boost::shared_ptr<ObjectOperation> ObjectOperationPtr;

    /**
       Abstract interface for a handler to a target or model added to the
       scene.

       This handler can be used to remove the objects associated with the
       target/model from the scene or to modify its attributes if that's
       allowed. The attribute map is copied from the original map passed
       when the objects were added to the scene.
     */
    class Object
    {
    public:
        /*--- Public member functions ---*/

        /**
            Returns the attribute map in which dynamic attributes can
            be changed.

            Attribute changes won't take effect until update is called, the
            update is not instantaneous but queued for processing until the
            next update traversal is performed.

            Throws an exception if no attribute can be changed.
            Setting attributes that are not understood will also throw an
            exception before any modification occurs.
        */
        virtual AttributeMap& getAttributes() = 0;

        /**
          Issues and update operation on the scene object managed by
          this handler.

          Attributes are copied internally so it's safe to modify the
          attributes after an update, however those changes won't take effect
          until update is called again.
        */
        virtual void update() = 0;

        /**
           Return a copy of the object passed to the add method that returned
           this handler.

           \if pybind
           These objects will be:
           - A read-only numpy array of u4 for neurons, with the neuron GIDs.
           - A brain.Synapses container for add{A,E}fferentNeurons
           - A string with the model name for addModel
           - None for addGeometry
           \else
           These objects will be:
           - A uint32_ts for addNeurons, with the neuron GIDs.
           - A brain.Synapses container for add{A,E}fferentNeurons
           - A std::string with the model name for addModel.
           - An empty boost::any for geometry objects.
           \endif
        */
        virtual boost::any getObject() const = 0;

        /**
           Return a handler to a subset of the entities from this object.

           The function shall throw if any of the ids does not identify
           any entity handled by this object.

           Attribute changes on the subset handler will affect only the
           entities selected.
           Attribute changes on the parent handler will still affect all
           the entities. The child handler attributes are also updated when
           the parent attributes are modified. However, attribute updates on
           a child handler are not propagated to the attributes of any
           other children (regardless of having overlapping subsets).
           Nevertheless, when calling update() changes in the attributes are
           always made effective when needed.

           The lifetime of the returned object is independent of the source
           one, but the returned object will be invalidated (operations will
           throw) when the source one is deallocated or invalidated.
           If this function is called recursively, the new returned objects
           will depend on the parent of the called object.

           Subset handlers are not part of the objects returned by
           Scene::getObjects().

           Beware that attribute updates on subhandlers may be discarded
           if the parent object has not been fully integrated in the
           scene yet (i.e., no frame including it has been rendered).

           This method may not be implemented by all objects.
         */
        virtual std::shared_ptr<Object> query(const uint32_ts& ids,
                                              bool checkIds = false);

        /**
           Applies an operation to this scene object.

           The operation is applied immediately, there is no need to call
           update(). The operation is distributed to all the nodes participating
           in the Equalizer configuration.

           @throw std::runtime_error if the operation does not accept this
           object as input.
         */
        virtual void apply(const ObjectOperationPtr& operation) = 0;

    protected:
        /*--- Protected constructor/destructor ---*/

        virtual ~Object();
    };
    typedef std::shared_ptr<Object> ObjectPtr;

    /** @internal */
    class _Impl;

    /*--- Public constructors/destructor ---*/

    ~Scene();

    Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    /*--- Public member functions (wrapped in Python) ---*/

    /** @name C++ public interface wrapped in Python */
    ///@{

    /**
       The runtime configurable attribute map.

       The modifiable attributes are:
       - *alpha_blending* (AttributeMap):
       The attribute map with options for transparency algorithms. See Scene
       class documentation
       - *em_shading* (bool): See Scene class documentation
       - *auto_update* (bool): Whether scene modifications automatically
       trigger a dirty signal or not.
       - *inflatable_neurons* (bool): Enable/disable neuron membrame
       inflation along the surface normal. The inflation factor is specified
       as a view attribute called *inflation_factor*.
    */
    AttributeMap& getAttributes();

    /**
       Set the brain::Circuit to be used for this scene.
       Throws if the scene already contains neurons or synapses.
    */
    void setCircuit(const CircuitPtr& circuit);

    /**
       Get the current brain::Circuit to be used for this scene
    */
    CircuitPtr getCircuit() const;

    /**
       Add a set of neurons to the scene.

       This is an asynchronous operation.
       The neuron container as well as the attribute map are copied
       internally so it is safe to modify them afterwards.
       Thread safe with regard to the rendering loop.

       \if pybind
       @param gids The GIDs of the neurons to add.
       \else
       @param gids An iterable or numpy array convertible to a GID set.
       \endif

       @param attributes Neuron display attributes:
       - mode (RepresentationMode):
         How to display neurons. Neurons added with SOMA or NO_AXON modes
         cannot be switched to WHOLE_NEURON later on.
       - color_scheme (ColorScheme):
         Coloring method to use. SOLID_COLOR by default is not provided
       - color (floatx4):
         RGBA tuple to be used as base color for SOLID_COLOR and
         BY_WIDTH_COLORS color schemes.
       - colormaps (AttributeMap):
         Optional submap with target specific color maps. These color maps
         override the color maps from the view. The supported color maps are:
         - *by_distance_to_soma*: The color map to use for the
            \if pybind
            BY_DISTANCE_TO_SOMA coloring scheme.
            \else
            BY_DISTANCE_TO_SOMA_COLORS coloring scheme.
            \endif
         - *by_width*: The color map to use for the
            \if pybind
            BY_WIDTH coloring scheme.
            \else
            BY_WIDTH_COLORS coloring scheme.
            \endif
         - *compartments*: The color map to use for compartmental simulation
           data.
         - *spikes*: The color map to use for spike rendering. The range of
           this color map must be always [0, 1], otherwise the rendering
           results are undefined.
       - primary_color (floatx4):
         An alias of the above.
       - secondary_color (floatx4):
         RGBA tuple to be used as secondary color for BY_WIDTH_COLORS.
       - max_visible_branch_order (int):
         Changes the maximum branching order of visible sections.
         Use -1 to make all braches visible and 0 to make only the soma visible.

       @return An object handler to the neuron set added.
    */
    ObjectPtr addNeurons(const GIDSet& gids,
                         const AttributeMap& attributes = AttributeMap());

    /**
       Adds a set of synapse glyphs at their post-synaptic locations to
       the scene.

       Thread safe with regard to the rendering loop.

       @param synapses The synapse container.
       @param attributes Synapse display attributes:
        - radius (float)
        - color (floatx4)
        - surface (bool):
          If true, the synapses are placed on the surfaces of the geometry,
          or in the center otherwise
       @return An object handler to the synapse set added.
    */
    ObjectPtr addAfferentSynapses(
        const brain::Synapses& synapses,
        const AttributeMap& attributes = AttributeMap());

    /**
       Adds a set of synapse glyphs at their pre-synaptic locations to
       the scene.

       Exactly the same as the function addAfferentSynapses but for efferent
       synapses.

       @return An object handler to the synapse set added.
    */
    ObjectPtr addEfferentSynapses(
        const brain::Synapses& synapses,
        const AttributeMap& attributes = AttributeMap());

    /**
       Loads a 3D model from a file and adds it to the scene.

       @param filename Model file to load.
       @param transform An affine transformation to apply to the model.
       @param attributes Model attributes:
        - color (floatx4)
          The diffuse color to be applied to the model to all parts that don't
          specify any material already.
        - flat (bool):
          If true and the model doesn't include it's own shaders, a shader to
          render facets with flat shading will be applied. If false, it has no
          effect.

       @return An object handler to the model added.

       \warning Additional models are not divided up in DB decompositions.
    */
    ObjectPtr addModel(const char* filename, const Matrix4f& transform,
                       const AttributeMap& attributes = AttributeMap());

    /**
       Convenience overload of the function above.

       @param filename Model file to load.
       @param transform A sequence of affine transformations. The sequence
       is specified as a colon separated string of 3 possible transformations:
       - rotations "r@x,y,z,angle"
       - scalings "s@x,y,z"
       - translations "t@x,y,z"
       @param attributes See function above.

       @return An object handler to the model added.

       \warning Additional models are not divided up in DB decompositions
     */
    ObjectPtr addModel(const char* filename, const char* transform = "",
                       const AttributeMap& attributes = AttributeMap());

    /** Adds geometry described as vertices and faces to the scene.

        \if pybind
        @param vertices A Nx4 numpy array of floats or a list of 4-element lists
               for adding points with size/radii to the scene. A Nx3 array or
               3-element lists for all other cases (points with a single radius
        @param primitive If None, the vertices will be added as points to the
               scene. For adding lines or triangles this parameter must be a MxI
               numpy array of integeres or N list of I-element lists, where I
               is 2 for lines and 3 for triangles.
        @param colors An Nx4 optional numpy array or N lists of 4-element
               iterables for per vertex colors, or a single 4-element iterable
               for a global color. If not provided a default color will be used.
        @param normals An optional Nx3 numpy array or a list of 3-element
               iterables with per vertex normals. Not used for points and
               spheres.
        \else
        @param vertices A Vec4Array for points with radius, a Vec3Array for
                        all the other cases.
        @param primitive The primitive indices for triangles or lines, not used
                         for displaying points.
        @param colors An optional per vertex color array for per vertex color
                      binding or a single element array for single overall
                      color. If not provided a default color will be used.
        @param normals An optional array with per vertex normals.
        \endif
        @param attributes Optional attributes concerning shading details
        - flat (bool):
          If true, the normal array is ignored and flat shading is used
          instead. If false and no normal array is provided, vertex normals
          are computed on the fly. Flat shading is only meaningful for
          triangle meshes.
        - line_width (float):
          Line width. Only for line primitives.
        - point_size (float):
          For points without individual size/radius this is the overall size.
          Its interpretation depends on the point style. For spheres, it's the
          radius. For points or circles, this is the screen size in pixels.
          If not specified, it will default to 1.
        - point_style (string):
          Use "spheres" to add real 3D spheres to the scene, "points" to add
          round points sprites and "circles" to add circles with 1 pixel of line
          width (the last two use regular GL_POINTS style). The default style
          if not specified is points.

        @return An object handler to the model added.
    */
#ifndef DOXYGEN_TO_BREATHE
    ObjectPtr addGeometry(const osg::ref_ptr<osg::Array>& vertices,
                          const osg::ref_ptr<osg::DrawElementsUInt>& primitive =
                              osg::ref_ptr<osg::DrawElementsUInt>(),
                          const osg::ref_ptr<osg::Vec4Array>& colors =
                              osg::ref_ptr<osg::Vec3Array>(),
                          const osg::ref_ptr<osg::Vec3Array>& normals =
                              osg::ref_ptr<osg::Vec3Array>(),
                          const AttributeMap& attributes = AttributeMap());
#else
    ObjectPtr addGeometry(object vertices, object primitive = None,
                          object colors = None, object normals = None,
                          const AttributeMap& attributes = AttributeMap());
#endif

    /**
       To use when auto_update is false to trigger the scene update.

       If the auto_update attribute is false, adding/removing objects from
       the scene or changing attributes that modify the rendering style
       will not trigger a new frame and consequent scene update.
       This function can be used to trigger it manually.
     */
    void update();

    /**
       Returns the handlers to all objects added to the scene.

       \if pybind
       @return A list of object handlers.
       \else
       @return An STL vector of object handlers.
       \endif
     */
#ifdef DOXYGEN_TO_BREATHE
    list getObjects();
#else
    std::vector<ObjectPtr> getObjects();
#endif

    /**
       Removes a target/model from the scene given its handler.
    */
    void remove(const ObjectPtr& object);

    /**
       Removes all the objects from the scene.

       Clipping planes are removed.

       To be called only from the application node.
    */
    void clear();

    /**
       Toggle highlighing of a cell set

       To be called only from the application node.
       \if pybind
       @param gids The set of cells to toggle.
       \else
       @param gids An iterable or numpy array convertible to a GID set.
       \endif
       @param on True to highlight the cell, false otherwise.
     */
    void highlight(const GIDSet& gids, bool on);

    /**
       Sets the set of unselectable cells.

       \if pybind
       @return A numpy array of u4 copied from the internal list.
       \endif
    */
    void setNeuronSelectionMask(const GIDSet& gids);

    /**
       Gets the set of unselectable cells.

       This mask affects the results of pick() functions.

       \if pybind
       @return A numpy array of u4 with the unslectable neurons.
       \endif
    */
    const GIDSet& getNeuronSelectionMask() const;

    /**
       Return the gids of the highlighted neurons.

       \if pybind
       @return A numpy array of u4 copied from the internal list.
       \endif
    */
    const GIDSet& getHighlightedNeurons() const;

    /**
       Thread safe with regard to the rendering loop.
    */
    void setSimulation(const CompartmentReportPtr& report);

    /**
        Thread safe with regard to the rendering loop.
    */
    void setSimulation(const SpikeReportReaderPtr& report);

#ifdef DOXYGEN_TO_BREATHE
    /* This function is added to the Python wrapping because there's
       no other way to remove the simulation reports otherwise. */
    /**
       Clear any simulation report from the scene.
    */
    void clearSimulation();
#endif

    /**
     * Intersection test between the pointer ray and the scene elements.
     *
     * May emit cellSelected or synapseSelected signals if a scene object
     * was hit.
     *
     * A signal is used to communicate the result to allow decoupling the
     * GUI event handling code from selection action callbacks.
     *
     * @param origin world space origin of the pick ray
     * @param direction pick ray direction, does not need to be normalized
     */
    void pick(const Vector3f& origin, const Vector3f& direction) const;

    /**
     * Intersection test of the space region selected by a rectangular area
     * projected using the camera from the given view.
     *
     * The implementation distinguishes between perspective and orthographic
     * projections.
     * Will emit a cellSetSelected signal with the group of somas intersected
     * by the projection of the rectangle (both towards infinite and the
     * camera position).
     *
     * A signal is used to communicate the result to allow decoupling the
     * GUI event hanlding code from selection action callbacks.
     *
     * @param view
     * @param left Normalized position (in [0,1]) of the left side of the
     *             rectangle relative to the camera projection frustum/prism.
     * @param right Normalized position (in [0,1]) of the right side of the
     *              rectangle relative to the camera projection frustum/prism.
     * @param bottom Normalized position (in [0,1]) of the bottom side of the
     *               rectangle relative to the camera projection frustum/prism.
     * @param top Normalized position (in [0,1]) of the top side of the
     *            rectangle relative to the camera projection frustum/prism.
     */
    void pick(const View& view, float left, float right, float bottom,
              float top) const;

    /**
     * Adds or modifies a clipping plane.
     *
     * Clipping planes are only applied to subscenes that have no spatial
     * decomposition, otherwise they are silently ignored.
     *
     * @param index Number of plane to be set. The maximum number of clipping
     *        planes is 8.
     * @param plane Plane equation of the clipping plane.
     * @throw runtime_error if index is >= 8.
     */
    void setClipPlane(unsigned int index, const Vector4f& plane);

    /**
     * Queries the clip plane with a given index.
     *
     * @throw runtime_error if no plane has been assigned in that index.
     */
    Vector4f getClipPlane(unsigned int index) const;

    void clearClipPlanes();

    /*--- Public signals ---*/

    /* Selection */

    /**
       Signal emitted when a cell is selected by pick.
     */
    CellSelectedSignal cellSelected;

    /**
       Signal emitted when a group of cells is selected by pick.
     */
    CellSetSelectedSignal cellSetSelected;

    /**
       Signal emitted when a synapse is selected by pick.
       \if pybind
       Do not store the synapse argument passed to the callback.
       \endif
    */
    SynapseSelectedSignal synapseSelected;

/**
   @deprecated use getSomasBoundingSphere() instead
   @if pybind
   Deprecated, use getSomasBoundingSphere instead.
   @return A tuple with the scene center and radius.
   @endif
*/
#ifdef DOXGEN_TO_BREATHE
    tuple getCircuitSceneBoundingSphere() const;
#else
    osg::BoundingSphere getCircuitSceneBoundingSphere() const;
#endif

/**
 * @return The center and radius around the somas of the scene.
 */
#ifdef DOXGEN_TO_BREATHE
    tuple getSomasBoundingSphere() const;
#else
    osg::BoundingSphere getSomasBoundingSphere() const;
#endif

/**
 * @return The center and radius around the synapses of the scene.
 */
#ifdef DOXGEN_TO_BREATHE
    tuple getSynapsesBoundingSphere() const;
#else
    osg::BoundingSphere getSynapsesBoundingSphere() const;
#endif

    osg::BoundingSphere getExtraModelsBoundingSphere() const;

    ///@}

    /*--- Public C++ only functions ---*/

    /** @name C++ only public interface */
    ///@{

    /**
       Applies the simulation timestamp to be available for the given
       frame number.

       The simulation is applied to all subscenes assuming that they will
       be needed to complete the frame (e.g. 2D/DB decompositions).

       If the mapper is working on the given timestamp, this function waits
       for it to finish. In other case it cancel the current operation,
       triggers the mapping of the timestamp and waits for it to finish.

       \todo The frame number will be used in time multiplexing.
       \todo Efficient implementation if DB load balancing is dynamic.
    */
    void mapSimulation(uint32_t frameNumber, float millisecond);

    /**
       Trigger the simulation data mapper to map simulation data to
       be used for a given frame.

       This function does essentially the same as above but it doesn't wait for
       the mapper to finish, neither cancels its current operation.

       @param frameNumber
       @param millisecond
       @return The number of mapping threads that will participate and need
       to be waited. This is equivalent to how many times simulationUpdated
       will be signalled.
    */
    unsigned int prepareSimulation(uint32_t frameNumber, float millisecond);

    /** @internal */
    const CompartmentReportPtr& getCompartmentReport() const;
    /** @internal */
    const core::SpikeReportPtr& getSpikeReport() const;

    //! Returns the scene interface to be used by Equalizer
    std::shared_ptr<osgEq::Scene> getSceneProxy();

    /*--- Public signals ---*/

    /* Progress */
    /**
       Emitted as scene loading/creation advances.
     */
    ProgressSignal progress;

    /* Updates */
    DirtySignal dirty;

    SimulationUpdatedSignal simulationUpdated;

    ///@}

protected:
    /*--- Protected constructors/destructor ---*/

    /**
       Constructor to be invoked by RTNeuron class
    */
    Scene(const RTNeuronPtr& application,
          const AttributeMap& attributes = AttributeMap());

private:
    /*--- Private member attributes ---*/

    std::shared_ptr<_Impl> _impl;
    std::weak_ptr<RTNeuron> _application;

    /*--- Private member functions ---*/

    /**
       Called from the application objects when it's going to be destroyed.
       If the wrapping is holding this object after this occurs, any
       operation will throw an exception to handle on the python side.
       This function deallocates any resources held by this object.
    */
    void _invalidate();

    /**
       Checks if invalidate was called and throws a Bad_Operation exception in
       that case.
     */
    void _checkValid() const;

    /**
     * Pick objects that are located inside a covex region defined by the
     * negative hemispaces of a list of planes.
     *
     * Will emit a cellSetSelected signal with the group of somas intersected
     * by the pyramid.
     *
     * @param planes A list of planes defined by the normal equation
     */
    void _pick(const std::vector<Vector4f>& planes) const;
};
}
}
#endif
