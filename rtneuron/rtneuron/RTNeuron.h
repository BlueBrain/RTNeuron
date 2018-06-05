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

#ifndef RTNEURON_API_RTNEURON_H
#define RTNEURON_API_RTNEURON_H

#include "AttributeMap.h"
#include "types.h"

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#include <string>

class QObject;
class QOpenGLContext;

namespace boost
{
namespace python
{
class list;
}
}

namespace bbp
{
namespace rtneuron
{
/**
   Parameters to configure the generation of movies by RTNeuron::record
 */
class RecordingParams
{
public:
    /**
      Parameters used for movie recording

      The default values are:
      - simulationStart: 0
      - simulationEnd: 0
      - simulationDelta: 1 (1s simulation time -> 40s@25fps video time)
      - cameraPathDelta(0) (Use the default value from the camera)
      - filePrefix: "frame_"
      - fileFormat: "png"
      - stopAtCameraPathEnd: false
      - frameCount: 0
    */
    RecordingParams();

    /** Start timestamp in milliseconds. */
    double simulationStart;
    /** End timestamp in milliseconds. */
    double simulationEnd;
    /** \brief Delta time milliseconds in which the simulation is advanced
        each frame. */
    double simulationDelta;

    /** Delta time in milliseconds which the camera path is advanced
        each frame.
        If 0 real time will be used. */
    double cameraPathDelta;
    /** The camera path to be used during rendering.
        If not assigned each view will keep its own camera manipulator. */
    CameraPathPtr cameraPath;

    /** Prefix to append to the output files. */
    std::string filePrefix;
    /** Extension (without dot) of the file format to use.
        File formats supported are those for which an OSG plugin is avaiable. */
    std::string fileFormat;

    /** Set to true to stop recording at the end of the camera path
        if cameraPathDelta is a positive number.
        The camera path time interval is considered open at the right. */
    bool stopAtCameraPathEnd;

    /** If different from 0, sets the number of frames to render before
        stop recording. */
    uint32_t frameCount;
};

// clang-format off

/**
   The main application class.

   This class manages the Equalizer configuration and is the factory for
   other classes that are tied to a configuration (e.g. the scenes).
 */
class RTNeuron : public std::enable_shared_from_this<RTNeuron>
{
public:
    /*--- Public declarations ---*/

    typedef void FrameIssuedSignalSignature();
    typedef boost::signals2::signal<FrameIssuedSignalSignature>
        FrameIssuedSignal;

    typedef void TextureUpdatedSignalSignature(unsigned);
    typedef boost::signals2::signal<TextureUpdatedSignalSignature>
        TextureUpdatedSignal;

    typedef void EventProcessorUpdatedSignalSignature();
    typedef boost::signals2::signal<EventProcessorUpdatedSignalSignature>
        EventProcessorUpdatedSignal;

    typedef void IdleSignalSignature();
    typedef boost::signals2::signal<IdleSignalSignature> IdleSignal;

    typedef void ExitedSignalSignature();
    typedef boost::signals2::signal<ExitedSignalSignature> ExitedSignal;

    friend class SimulationPlayer;

/*--- Public constructors/destructor */

    /**
       \if pybind
       @param argv The command line argument list.
       \else
       The command line arguments (argc, argv) are internally stored to
       be used at initLocal invokation when the view is created.
       @param argc
       @param argv
       \endif
       @param attributes Global application options, including:
       - *afferent_syn_color* (floatx3[+1]):
         Default color to use for afferent synapse glyphs
       - *autoadjust_simulation_window* (bool)
         Whether the simulation player window should be adjusted automatically.
         Simulation window adjustment occurs when:
         - SimulationPlayer::setTimestamp is called
         - SimulationPlayer::play is called
         - A new simulation timestamp has been mapped and is ready for
           displaying.
         Auto-adjustment will not try to obtain the latest simulation data if
         it can lead to a deadlock (e.g. when the engine is already trying
         to do it for a previously requested timestamp).
       - *efferent_syn_color* (floatx3[+1]):
         Default color to use for efferent synapse glyphs
       - *has_gui* (bool):
         True to indicate the RTNeuron object that it's running inside a
         QT application.
       - *neuron_color* (floatx3[+1]):
         Default color for neurons
       - *soma_radii* (AttributeMap):
         An attribute map indexed with morphology type names as attribute
         names and radii as attribute values.
       - *soma_radius* (float):
         Default soma radius to use if no additional information is available.
       - *profile* (AttributeMap):
         An attribute map with options for profiling:
         - enable (bool):
           Enable frame time profiling
         - logfile (string):
           Log file name to write frame times.
         - compositing (bool):
           False to disable frame compositing, True or non present otherwise.
         .
       - *view* (AttributeMap):
         An attribute map with the default view parameters (e.g. background,
         lod_bias, ...).
       - *window_width* (int):
         Width of the application window in pixels.
       - *window_height* (int):
         Height of the application window in pixels.
       .
       @version 2.4 required for the *profile* attribute.
     */
#ifndef DOXYGEN_TO_BREATHE
    RTNeuron(const int argc, char* argv[],
             const AttributeMap& attributes = AttributeMap());
#else
    RTNeuron(list argv, const AttributeMap& attributes = AttributeMap());
#endif

    RTNeuron(const RTNeuron&) = delete;
    RTNeuron& operator=(const RTNeuron&) = delete;

    ~RTNeuron();

    /*--- Public member functions (wrapped in Python) ---*/

    /**
       \brief Creates the view (windows, rendering threads, event processing...)
       Throws if there's a view already created.

       In parallel rendering configurations this function needs to call
       eq::client::initLocal to launch the rendering clients. Instead of
       blocking forever inside this function, a new thread will be created
       for the client loop.

       This function blocks until the application (or rendering client)
       loop is guaranteed to have been started.
       The rendering loop starts paused RTNeuron::resume must be called
       to start the rendering.

       Do not call from outside the main thread.

       @param config Path to an Equalizer configuration file or hwsd session
                     name.
    */
    void init(const std::string& config = "");

    /**
       Deprecated, use init instead.

       @deprecated
    */
    void createConfig(const std::string& config = "");

    /**
       \brief Stops all rendering and cleans up all the resource from the
       current Equalizer configuration.

       The exited signal will be emitted when the config is considered to be
       done. Rendering is not stopped yet at that point.

       Do not call from outside the main thread.
    */
    void exit();

    /**
       Deprecated, use exit instead.
       @deprecated
    */
    void exitConfig();

    /**
       \brief Change the Equalizer layout.

       Throws if the layout does not exist or if there is no configuration
       initialized.
       Do not call outside the main thread.
     */
    void useLayout(const std::string& layoutName);

    /**
       \brief Returns the vector of active views.
    */
    std::vector<ViewPtr> getViews();

    /**
       \brief Returns the vector of active or inactive views which belong
              to any layout.
    */
    std::vector<ViewPtr> getAllViews();

    /**
       \brief Creates a Scene to be used in this application.

       The attribute map includes scene attributes that are passed to
       the scene constructor. See \ref Scene_details "here" for details.

       Currently, all scenes to be used inside a config must be created in
       all nodes before init is called.

       The application does not hold any reference to the returned scene.
       If the caller gets rid of the returned reference and no view holds
       the scene, the scene will be deallocated.

       Do not call from outside the main thread.

       @sa Scene::Scene
     */
    ScenePtr createScene(const AttributeMap& attributes);

    /**
       \brief Returns the interface object to simulation playback.

       A single player by default. May become an external object and shared by
       different views.
    */
    SimulationPlayerPtr getPlayer();

    /**
       \brief High level function to dump the rendered frames to files.

       When called, the rendering loop is resumed if paused.

       If a camera path is given, a new camera path manipulator is created
       and assigned to all active views. Any previous camera manipulator
       will be overriden.

       Recording is stopped automatically when:
       - The camera path end is reached.
       - The end of the simulation window is reached.
       - The maximum number of frames to render is reached.
       whichever occurs first.

       This function does not wait until all frames are finished (you can
       set a callback to frameIssued to count frames or use waitRecord).
       If the simulation delta is set to 0, no simulation playback will be
       performed (in that case, the current simulation window remains
       unmodified).

       If the simulation window is invalid (start >= stop) it will not be
       considered to stop the recording.

       Idle anti-alias appears disabled when using this function.

       Changing simulation playback parameters in the player API will
       interfere with the results of this function.

       Do not call from outside the main thread.
    */
    void record(const RecordingParams& params);

    /**
       \brief Block rendering loop after the current frame finishes

       Do not call from outside the main thread in the application node.
     */
    void pause();

    /**
       \brief Resume the rendering loop

       Do not call from outside the main thread in the application node.
    */
    void resume();

    /**
       \brief Trigger a frame

       If the rendering is paused, triggers the rendering of exactly
       one frame.

       If the rendering loop is running, triggers a redraw request if the
       rendering loop was waiting for this event.

       Do not call from outside the main thread in the application node.
    */
    void frame();

    /**
       \brief Wait for a new frame to be finished

       Returns inmediately if no config is active. Exiting the config will
       also unlock the caller.
       Do not call from outside the main thread in the application node.
    */
    void waitFrame();

    /**
       \brief Wait for at least n frames to be finished.

       This function resumes the rendering loop.
       Returns inmediately if no config is active. Exiting the config will
       also unlock the caller.
       More frames may be generated before the function returns.
       Do not call from outside the main thread in the application node.
    */
    void waitFrames(const unsigned frames);

    /**
       \brief Wait for the Equalizer application loop to exit.

       This function returns inmediately if not config is active. Otherwise
       it blocks until some event makes the application loop to exit.

       While a thread is blocked in this function, calls to init.
       or waitRecord will block.
     */
    void wait();

    /**
       \brief Wait for the last frame of a movie recording to be issued.

       Do not call outside the main thread.
    */
    void waitRecord();

    /**
       \brief Returns a modifiable attribute map.

       These attributes can be modified at runtime.
    */
    AttributeMap& getAttributes();

    /**
       \brief Returns the version string to print out.
    */
    static std::string getVersionString();

    /**
       @internal
       @return the object to which send GUI events for the active view.
     */
    QObject* getActiveViewEventProcessor() const;

    /**
       @internal Set the Qt Open GL context to share the rendering with.
     */
    void setShareContext(QOpenGLContext* shareContext);

    /*--- Public signals ---*/

    /**
       \brief Emitted after the internal rendering loop has finished issuing
       a frame.

       The frame is not necessarily finished at this point, but all the
       distributed objects are guaranteed to have been committed.
       This can be used for animations.
    */
    FrameIssuedSignal frameIssued;

    /**
       Emitted after the texture that captures the appNode rendering was
       updated.

       This signal is tied to the existence of the GUI widget which takes this
       notification to render the UI and the texture.
    */
    TextureUpdatedSignal textureUpdated;

    /**
       @internal
       Emitted to notify the QT GUI to which object does it have to forward
       events.
    */
    EventProcessorUpdatedSignal eventProcessorUpdated;

    /**
       Emitted when the application is idle, e.g. waiting for (user)events.
    */
    IdleSignal idle;

    /**
       Emitted while the config is done in Config::setDone.

       Frame rendering is not finished at the point. The signal indicates that
       the lifetime of objects that might be referenced outside of the library
       (e.g. in python) is about to end, so object destructions can be scheduled
       accordingly.
    */
    ExitedSignal exited;

protected:
    /*--- Protected member functions ---*/

    /**
       This is the actual destructor.
    */
    void finalize();

    /**
       Disconnects all slots from all signals, including the simulation player.
     */
    void disconnectAllSlots();

private:
    /*--- Private member declarations ---*/

    class _Impl;
    _Impl* _impl;
};
}
}
#endif
