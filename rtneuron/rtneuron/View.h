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

#ifndef RTNEURON_API_VIEW_H
#define RTNEURON_API_VIEW_H

#include "types.h"

#include <osg/Version>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class View;
}

// clang-format off

/**
   This class represents a view on a scene.

   A view holds together a scene, a camera and the visual attributes that are
   not bound to a scene, e.g. level of detail bias, simulation color map,
   stereo correction.
   A view can also have a camera manipulator and a selection pointer. Cameras
   are view specific and cannot be shared.

   There is a one to one mapping between RTNeuron views and Equalizer views.

   \note Currently the simulation report is bound to the scene but this will
   be moved to the view in the future. The same applies to
   enabling/disabling alpha blending at runtime.
 */
/*
   \todo Consider adding a setCamera method to share cameras between views.
   This feature needs to be considered also in the context of a future
   integration with vizjam.
 */
class View
{
public:
    /*--- Public declarations ---*/
    friend class RTNeuron;
    friend class Camera;
    friend class Scene;

    typedef void FrameGrabbedSignalSignature(const std::vector<uint8_t>&);
    typedef boost::signals2::signal<FrameGrabbedSignalSignature>
        FrameGrabbedSignal;

    /*--- Public constructors/destructor ---*/
    ~View();

    View() = default;
    View(const View&) = delete;
    View& operator=(const View&) = delete;

    /*--- Public member functions ---*/

    /** @name C++ public interface wrapped in Python */
    ///@{

    /**
       Attribute map with runtime configurable attributes for a View.

       Existing attributes are:
       - General:
         - *background* (floatx4):
           Background color. The alpha channel of the background is considered
           by frame grabbing functions. If alpha equals to 1, the output
           images will have no alpha channel.
         - *use_roi* (float):
           Compute and use regions of interest for frame readback in parallel
           rendering configurations.
       - Appearance:
         - *clod_threshold* (float):
           When using continuous LOD, the unbiased distance at which the
           transition from pseudocylinders to tublets occurs for branches
           of radius 1. This value is modulated by the lod_bias. During
           rendering, the distance of a segment is divided by its radius
           before comparing it to the clod_threshold.
         - *colormaps* (AttributeMap):
           A map of ColorMap objects. The currently supported color maps are:
           - *compartments*: The color map to use for compartmental simulation
             data.
           - *spikes*: The color map to use for spike rendering. This range of
             this color map must be always [0, 1], otherwise the rendering
             results are undefined.
         - *display_simulation* (bool):
           Show/hide simulation data.
         - *idle_AA_steps* (int):
           Number of frames to accumulate in idle anti-aliasing
         - *highlight_color* (floatx4):
           The color applied to make highlighted neurons stand out. The
           highlight color replaces the base color when *display_simulation*
           is disabled. When *display_simulation* is enabled, the highlight
           color is added to the color obtained from simulation data mapping.
         - *inflation_factor* (float):
           Sets the offset in microns by which neuron membrane surfaces will be
           displaced along their normal direction. This parameter has effect
           only on those scenes whose *inflatable_neurons* attribute is set
           to true.
         - *lod_bias* (float):
           A number between 0 and 1 that specifies the bias in LOD selection.
           0 goes for the lowest LOD and 1 for the highest.
         - *probe_color* (floatx4):
           The color to apply to those parts of a neuron whose simulation value
           is above the threshold if simulation display is enabled.
         - *probe_threshold* (float):
           The simulation value above which the probe color will be applied to
           neuron surfaces if simulation display is enabled.
         - *spike_tail* (float):
           Time in millisecond during which the visual representation of
           spikes will be still visible.
       - Frame capture
         - *snapshot_at_idle* (bool):
           If true, take snapshots only when the rendering thread becomes idle
           (e.g. antialias accumulation done). Otherwise, the snapshot is taken
           at the very next frame.
         - *output_file_prefix* (string):
           Prefix for file written during recording.
         - *output_file_format* (string):
           File format extension (without dot) to
           use during frame recording. Supported extensions are those for
           which OSG can find a pluging.
       - Cameras and stereo
         - *auto_compute_home_position* (bool):
           If true, the camera manipulator home position is recomputed
           automatically when the scene object is changed or when the scene
           emits its dirty signal.
         - *auto_adjust_model_scale* (bool):
           If true, every time the scene is changed the ratio between world and
           model scales is adjusted.
         - *depth_of_field* (AttributeMap):
           Attributes to enable and configure depth of field effect.
           - *enabled* (bool)
           - *focal_distance* (float):
             Distance to the camera in world units at which objects are in
             focus
           - *focal_range* (float):
             Distance from the focal point within objects remain in focus.
         - *model_scale* (bool) :
           Size hint used by Equalizer to setup orthographic projections and
           stereo projections. Set to 1 in order to use world coordinates in
           orthographic camera frustums.
         - *stereo_correction* (float):
           Multiplier of the scene size in relation to the observer for stereo
           adjustment.
         - *stereo* (bool) :
           Enables/disables stereoscopic rendering.
         - *zero_parallax_distance* (float):
           In stereo rendering, the distance from the
           camera in meters at which left and right eye projections converge
           into the same image (only meaningful for fixed position screens).
       All valid attributes are initialized to their default values.
    */
    /*
       Transparency could be added here but this requires studying this:

       http://forum.openscenegraph.org/viewtopic.php?t=5441
       http://forum.openscenegraph.org/viewtopic.php?t=6072
    */
    AttributeMap& getAttributes();
    const AttributeMap& getAttributes() const;

    /**
       Sets the scene to be displayed.
     */
    void setScene(const ScenePtr& scene);

    ScenePtr getScene() const;

    CameraPtr getCamera() const;

    /**
       Sets the manipulator that controls de camera.

       The camera manipulator will receive input events from this view and
       process them into a model matrix.
       At construction, a trackball manipulator is created by default.

       \todo Explain what happens when multiple views share a camera and
       how the wrapping will look like
     */
    void setCameraManipulator(const CameraManipulatorPtr& manipulator);
    CameraManipulatorPtr getCameraManipulator() const;

    /**
       Compute the home position for the current scene and set it to the camera
       manipulator.

       The camera position is also reset to new home position.
    */
    void computeHomePosition();

    void setPointer(const PointerPtr& pointer);
    PointerPtr getPointer();

    /**
       Enable or disable frame grabbing.

       @param enable If true, rendered images will be written to files
       starting from next frame on.

       \if pybind
       @sa File naming attributes from View.attributes
       \else
       @sa File naming attributes from getAttributes
       \endif
     */
    void record(bool enable);

    /**
       Triggers a frame and emit the frameGrabbed signal.

       This method does not wait until the frame is rendered.

       When idle AA is enabled and the *snapshot_at_idle* attribute is set,
       the signal is emitted when the frame accumulation is finished.
     */
    void grabFrame();

    /**
       Triggers a frame and writes the rendered image to a file.

       This method waits until the image has been written unless
       waitForCompletion is false, in which case it returns inmediately.

       When idle AA is enabled and the *snapshot_at_idle* attribute is set,
       the snapshot is taken when frame accumulation is finished.

       Throws if fileName is empty.

       @param fileName Filename including extension. If the filename include
                       the squence "%c" all destination channels will be
                       captured, replacing "%c" with the channel name in the
                       output file. Notice that this option is meaningless
                       for the offscreen snapshot functions.
       @param waitForCompletion if true, locks until the image has been
       written to a file.
     */
    void snapshot(const std::string& fileName, bool waitForCompletion = true);

    /**
       Triggers a frame on an auxiliary off-screen window and writes the
       rendered image to a file.

       The off-screen window can have a different size than the
       windows in which this view resides.
       The vertical field of view of the camera will be preserved.

       This method waits until the image has been written.

       When idle AA is enabled and the *snapshot_at_idle* attribute is set,
       the snapshot is taken when frame accumulation is finished.

       Throws if fileName is empty or scale is negative or zero.

       @param fileName Filename including extension.
       @param scale Scale factor that will be uniformly applied to the original
       view to obtain the final image.
     */
    void snapshot(const std::string& fileName, float scale);

    /**
       Triggers a frame on an auxiliary off-screen window and writes the
       rendered image to a file.

       The off-screen window can have a different size than the
       windows in which this view resides.
       The vertical field of view of the camera will be preserved.

       This method waits until the image has been written.

       When idle AA is enabled and the *snapshot_at_idle* attribute is set,
       the snapshot is taken when frame accumulation is finished.

       Throws if fileName is empty or if any of the resolution components is 0.

       @param fileName Filename including extension.
       \if pybind
       @param resolution Tuple containing the horizontal and vertical resolution
       that will be used to generate the final image.
       \else
       @param resX Horizontal resolution that will be used to generate the final
       image.
       @param resY Vertical resolution that will be used to generate the final
       image.
       \endif
     */
#ifdef DOXYGEN_TO_BREATHE
    void snapshot(const std::string& fileName, const tuple& resolution);
#else
    void snapshot(const std::string& fileName, size_t resX, size_t resY);
#endif

    /** Emitted after grabFrame() is done */
    FrameGrabbedSignal frameGrabbed;

    ///@}

    /*--- Public C++ only member functions ---*/

    /** @name C++ only public interface */
    ///@{

    osgEq::View* getEqView();

    const osgEq::View* getEqView() const;

    /** @internal */
    void disconnectAllSlots();

    ///@}

protected:
    /*--- Protected member attributes ---*/

    class _Impl;
    _Impl* _impl;
    std::weak_ptr<RTNeuron> _application;

    /*--- Protected  constructors/destructor ---*/

    View(osgEq::View* view, const RTNeuronPtr& application,
         const AttributeMap& style);

private:
    /*--- Private member functions ---*/

    /**
       Called when the backend Equalizer object is going to be released.
       If the wrapping is holding this object after this occurs, any
       operation will throw an exception to handle on the python side.
       This function deallocates any resources held by this object.
    */
    void _invalidate();

    /**
       Checks if invalidate was called and throws an exception in that case.
     */
    void _checkValid() const;
};
}
}
#endif
