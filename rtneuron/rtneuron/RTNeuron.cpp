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

#include <eq/gl.h>

#include "InitData.h"
#include "RTNeuron.h"
#include "Scene.h"
#include "SceneImpl.h"
#include "SimulationPlayer.h"
#include "View.h"
#include "detail/Configurable.h"
#include "detail/RTNeuronEvent.h"
#include "ui/CameraPath.h"
#include "ui/CameraPathManipulator.h"

#include "config/Globals.h"
#include "config/constants.h"
#include "data/Neuron.h"
#include "data/SpikeReport.h"
#include "util/attributeMapHelpers.h"
#include "util/log.h"
#include "viewer/osgEq/Application.h"
#include "viewer/osgEq/Channel.h"
#include "viewer/osgEq/Client.h"
#include "viewer/osgEq/Config.h"
#include "viewer/osgEq/ConfigEvent.h"
#include "viewer/osgEq/FrameData.h"
#include "viewer/osgEq/InitData.h"
#include "viewer/osgEq/Node.h"
#include "viewer/osgEq/Pipe.h"
#include "viewer/osgEq/Scene.h"
#include "viewer/osgEq/View.h"
#include "viewer/osgEq/Window.h"

#include <rtneuron/version.h>

#include <brain/compartmentReport.h>

#include <eq/admin/init.h>
#include <eq/eventICommand.h>

#include <lunchbox/scopedMutex.h>

#include <osg/Geode>
#include <osg/ShapeDrawable>

#include <unordered_map>

#define QUOTE(string) STRINGIFY(string)
#define STRINGIFY(foo) #foo

namespace bbp
{
namespace rtneuron
{
namespace
{
const int DEFAULT_WINDOW_WIDTH = 1280;
const int DEFAULT_WINDOW_HEIGHT = 800;

const double NaN = std::numeric_limits<double>::quiet_NaN();

class MergeSimulationWindows
{
public:
    MergeSimulationWindows(double& start, double& end, const bool updateStreams)
        : _start(start)
        , _end(end)
        , _updateStreams(updateStreams)
        , _changed(false)
    {
    }

    void operator()(core::SpikeReport& report) const
    {
        if (_updateStreams)
            report.tryUpdate();
        _merge(report.getStartTime(), report.getEndTime());
    }

    void operator()(brain::CompartmentReport& report) const
    {
        const auto& metadata = report.getMetaData();
        _merge(metadata.startTime, metadata.endTime);
    }

    double& _start;
    double& _end;
    const bool _updateStreams;
    mutable bool _changed;

private:
    void _merge(const double start, const double end) const
    {
        /* By using the negation of the comparison, the case in which _end
           and _start are NaN is also covered. */
        if (!(end <= _end))
        {
            _changed = true;
            _end = end;
        }
        if (!(start >= _start))
        {
            _changed = true;
            _start = start;
        }
    }
};

void _setEqWindowIAttribute(const AttributeMap& attributes, const char* eqName,
                            const char* attrName, int defaultValue)
{
    int value = 0;
    if (attributes.get(attrName, value) != 1)
    {
        /* If no attribute has been provided check if the environmental
           variable has already been set or provide a default otherwise */
        if (getenv(eqName))
            return;
        value = defaultValue;
    }
    std::stringstream str;
    str << value;
    ::setenv(eqName, str.str().c_str(), 1);
}

std::string _defaultEQConfig;
}

using namespace core;

typedef std::vector<ViewPtr> ViewPtrs;
typedef std::vector<ScenePtr> ScenePtrs;

// RTNeuron::_Impl -------------------------------------------------------------

/**
  This class will do all the initialization that is external to the class
  inheriting from eq::Client (quite similar to main.cpp in the eqPly example).
  At the same time, this class is the eq::NodeFactory so we can keep track
  of all the objects created by Equalizer (in particular we are interested
  in the views).
*/
class RTNeuron::_Impl : public detail::Configurable,
                        public eq::NodeFactory,
                        public osgEq::Application
{
    friend class RTNeuron;

public:
    /*--- Public declarations */

    class Notifier;

    /*--- Public constructors/destructor */

    _Impl(const int argc, char* argv[], const AttributeMap& attributes,
          RTNeuron* parent);

    ~_Impl();

    /*--- Public member functions ---*/

    void createEqualizerConfig(const std::string& configFileName);

    void exitConfig();

    void useLayout(const std::string& name);

    ViewPtrs getViews(const bool onlyActive = true);

    void registerScene(const ScenePtr& scene);

    void pause();

    void resume();

    void frame();

    void waitFrame();

    void waitFrames(const unsigned int frames);

    void waitRecord();

    void wait();

    void record(const RecordingParams& params);

    /* Simulation playback */

    void setSimulationTimestamp(double milliseconds);

    double getSimulationTimestamp() const;

    void setSimulationWindow(double start, double stop);

    void getSimulationWindow(double& start, double& stop) const;

    void adjustSimulationWindow();

    void setSimulationBeginTime(double milliseconds);

    double getSimulationBeginTime() const;

    void setSimulationEndTime(double milliseconds);

    double getSimulationEndTime() const;

    void setSimulationDelta(double milliseconds);

    double getSimulationDelta() const;

    void resyncSimulation();

    void playSimulation();

    void pauseSimulation();

    /* osgEq::Application methods */

    void preFrame() final;

    void postFrame() final;

    void preNodeUpdate(const uint32_t frameNumber) final;

    void preNodeDraw(const uint32_t /* frameNumber */) final {}
    void init(osgEq::Config* config) final;

    void exit(osgEq::Config* config) final;

    void configInit(eq::Node* node) final;

    void configExit(eq::Node* node) final;

    bool handleEvent(eq::EventICommand command) final;

    void onActiveViewTextureUpdated(const unsigned textureID) final;

    void onActiveViewEventProcessorUpdated(QObject* processor) final;

    void onFrameGrabbed(const eq::uint128_t& viewID,
                        const std::vector<uint8_t>& data) final;

    void onIdle() final;

    void onDone() final;

    /* This function propagates the scene of the old layout to the new one,
       but is does only so if all views in the old layout had the same scene. */
    void onLayoutChanged(const eq::Layout* oldLayout,
                         const eq::Layout* newLayout) final;

    /* Node factory methods */
    virtual osgEq::Config* createConfig(eq::ServerPtr parent);
    virtual osgEq::Node* createNode(eq::Config* parent);
    virtual eq::Pipe* createPipe(eq::Node* parent);
    virtual eq::Window* createWindow(eq::Pipe* parent);
    virtual eq::Channel* createChannel(eq::Window* parent);
    virtual eq::View* createView(eq::Layout* parent);

    virtual void releaseNode(eq::Node* node);
    virtual void releaseView(eq::View* view);

private:
    /*--- Private declarations ---*/
    static bool s_instantiated;

    typedef std::map<eq::View*, ViewPtr> Views;
    typedef Views::iterator ViewsIter;
    typedef std::vector<osgEq::Node*> Nodes;
    typedef std::vector<SceneWeakPtr> Scenes;

    /*--- Private member attributes ---*/

    RTNeuron* _parent;
    NodeFactory _factory;

    std::vector<char*> _argv;
    std::vector<std::string> _args;

    mutable std::mutex _lock;

    std::mutex _exitMutex;
    std::condition_variable _exitCondition;
    bool _exitingConfig;
    unsigned _releaseAtExit;

    Views _views;
    Nodes _nodes;
    Scenes _scenes;

    osgEq::ClientPtr _client;
    AttributeMap _initAttributes;
    InitDataPtr _initData;

    lunchbox::Monitor<bool> _isRecording;

    CameraPathPtr _recordingCameraPath;
    double _cameraPathDelta;
    size_t _recordedPathFrames;
    size_t _recordFrames;
    osg::Timer _recordingTimer;

    double _simulationStart;
    double _simulationEnd;
    double _simulationDelta;
    double _nextTimestamp;
    double _timestampReady;
    double _displayedTimestamp;
    SimulationPlayer::PlaybackState _playbackState;
    bool _autoAdjustSimulationWindow;

    typedef std::map<co::uint128_t, unsigned int> SimulationMappersPerNode;
    SimulationMappersPerNode _mappersPerNode;
    typedef std::map<double, unsigned int> NotifiedTimestampCounts;
    NotifiedTimestampCounts _notifiedThreads;

    SimulationPlayerPtr _player;

    typedef std::unordered_map<View*, int> ViewIdleStepsMap;
    ViewIdleStepsMap _idleAAState;

    QObject* _activeViewEventProcessor;
    QOpenGLContext* _shareContext;

    LB_TS_VAR(_controlThread);

    /*--- Private member functions ---*/

    virtual void onAttributeChangingImpl(
        const AttributeMap& attributes, const std::string& name,
        const AttributeMap::AttributeProxy& parameters);

    virtual void onAttributeChangedImpl(const AttributeMap& map,
                                        const std::string& name);

    ViewPtrs _getViews(const bool onlyActive);

    /** Applies an operator to the set of scenes from active views */
    template <typename Functor>
    void _visitScenes(const Functor& functor);

    /** Applies an operator to the reports attached to the scenes from active
        views.

        The functor must define two operator() overloads, one for
        CompartmentBasedReports and the other for SpikeBasedReports
    */
    template <typename Functor>
    void _visitReports(const Functor& functor, bool onlyStreamBased = false);

    /** Checks that the timesamp is valid and throws otherwise.

        A timestamp is valid if:
        - It's within the current simulation window.
        - It's not beyond the last timestamp received when there is
          a stream-based report attached to any active view.
    */
    void _checkTimestamp(const double timestamp);

    /** Sets a requested simulation timestamp into the frame data and requests
        a redraw. */
    void _setSimulationTimestamp(const double timestamp);

    /** Update the simulation on all active subscenes. */
    void _processSimulationTimestamp(const uint32_t frameNumber);

    void _onSimulationUpdated(const double timestamp);

    /** Receives the notification from a client meaning that the simulation
        for the given timestamp is ready for displaying.

        Adjusts the simulation window, schedules the next timestamp to
        display and triggers a new frame as needed.

        This method calls _notifyTimestamp, _adjustSimulationWindow and
        _scheduleNextTimestamp, handling the locking externally to them.
    */
    void _simulationReady(const double timestamp);

    /** Receives the notification from a client node meaning that the simulation
        for the given timestamp is ready for displaying.

        When all clients have notified a timestamp, the FrameData attribute
        "next_simulation_timestamp" is set with the timestamp.

        This function does not send any external simulation player signal,
        that occurs at postFrame.

        @param timestamp The simulation timestamp for which the client has
        simulation data ready.

        @return True if all the notifications for the input timestamp have
        been received, false in other case.
    */
    bool _notifyTimestamp(const double timestamp);

    /** Adjust the current simulation window considering stream based
        reports of active scenes.

        @return True if the window has been adjusted.
    */
    bool _adjustSimulationWindow();

    /** Set the next simulation timestamp to be prepared if needed.

        If recording is off and simulation playback is on, a new frame
        will be triggerred.
        If none of the simulation playback edges is reached, the next
        timestamp is assigned a FrameData attribute named
        next_simulation_timestamp. Otherwise, the frame attribute
        "playback_finished" is assigned.
     */
    void _scheduleNextTimestamp(const double current);

    void _findActiveScenes(const std::vector<unsigned int>& ids,
                           std::set<ScenePtr>& scenes);

    void _checkRecordingDone();

    bool _continueRecording();

    void _saveIdleAAState();

    void _restoreIdleAAState();
};

/**
   Class for emitting signals at context exit.

   The purpose of this class is to reduce copy-paste code for emitting
   signal outside of the scope of critical sections.

   Since all state changes handled by this class are synchronized, in order
   to reduce typing and potential errors (locking before declaring the
   notifier, which could lead to trivial deadlocks during signal emission)
   this class locks the application instance mutex in the constructor and
   releases it before the signals are emitted.
*/
class RTNeuron::_Impl::Notifier
{
public:
    enum SignalBits
    {
        PLAYER_WINDOW_CHANGED = 1,
        PLAYER_STATE_CHANGED = 2,
        PLAYER_DELTA_CHANGED = 4,
    };

    Notifier(_Impl* app)
        : _lock(app->_lock)
        , _app(app)
        , _window(_app->_simulationStart, _app->_simulationEnd)
        , _state(_app->_playbackState)
        , _delta(_app->_simulationDelta)
    {
    }

    ~Notifier()
    {
        if (!_app->_player)
            return;

        int flags = 0;
        if (_delta != _app->_simulationDelta)
        {
            flags |= PLAYER_DELTA_CHANGED;
            _delta = _app->_simulationDelta;
        }
        if (_state != _app->_playbackState)
        {
            flags |= PLAYER_STATE_CHANGED;
            _state = _app->_playbackState;
        }
        if (_window.first != _app->_simulationStart ||
            _window.second != _app->_simulationEnd)
        {
            flags |= PLAYER_WINDOW_CHANGED;
            _window =
                std::make_pair(_app->_simulationStart, _app->_simulationEnd);
        }

        _lock.unlock();

        if (flags & PLAYER_WINDOW_CHANGED)
            _app->_player->windowChanged(_window.first, _window.second);

        if (flags & PLAYER_STATE_CHANGED)
        {
            if (_state == SimulationPlayer::FINISHED)
                _app->_player->finished();
            _app->_player->playbackStateChanged(_state);
        }

        if (flags & PLAYER_DELTA_CHANGED)
            _app->_player->simulationDeltaChanged(_delta);
    }

    lunchbox::ScopedWrite _lock;
    _Impl* _app;
    std::pair<double, double> _window;
    SimulationPlayer::PlaybackState _state;
    double _delta;
};

bool RTNeuron::_Impl::s_instantiated = false;

RTNeuron::_Impl::_Impl(const int argc, char* argv[],
                       const AttributeMap& attributes, RTNeuron* parent)
    : _parent(parent)
    , _exitingConfig(false)
    , _releaseAtExit(0)
    , _isRecording(false)
    , _recordFrames(0)
    , _simulationStart(NaN)
    , _simulationEnd(NaN)
    , _simulationDelta(0.1)
    , _nextTimestamp(0)
    , _timestampReady(NaN)
    , _displayedTimestamp(NaN)
    , _playbackState(SimulationPlayer::PAUSED)
    , _autoAdjustSimulationWindow(true)
    , _activeViewEventProcessor(0)
    , _shareContext(0)
{
    if (s_instantiated)
    {
        throw std::runtime_error(
            "RTNeuron: multiple RTNeuron instances"
            " are not allowed.");
    }

    /* Copying arguments internally. They shouldn't be modified from
       outside, but this is harder to guarantee in the wrapping. */
    _args.reserve(argc);
    _argv.reserve(argc);
    for (int i = 0; i != argc; ++i)
    {
        _args.push_back(argv[i]);
        _argv.push_back(const_cast<char*>(&_args.back()[0]));
    }

    /* Applying some global options */
    AttributeMap& attr = getAttributes();
#define COPY_ATTRIBUTE(str)             \
    try                                 \
    {                                   \
        attr.set(str, attributes(str)); \
    }                                   \
    catch (...)                         \
    {                                   \
    }
    COPY_ATTRIBUTE("auto_adjust_simulation_window")
    COPY_ATTRIBUTE("has_gui")
    COPY_ATTRIBUTE("neuron_color")
    COPY_ATTRIBUTE("afferent_syn_color")
    COPY_ATTRIBUTE("efferent_syn_color")
    COPY_ATTRIBUTE("soma_radius")
    COPY_ATTRIBUTE("soma_radii")
    COPY_ATTRIBUTE("profile")
#undef COPY_ATTRIBUTE

    /* Storing the initialization attributes. */
    _initAttributes.merge(attributes);

    /* With the GUI widget, this is not needed + it messes up the aspect ratio
       with dual-screen setups */
    if (!_initAttributes("has_gui", false))
    {
        _setEqWindowIAttribute(attributes, "EQ_WINDOW_IATTR_HINT_WIDTH",
                               "window_width", DEFAULT_WINDOW_WIDTH);
        _setEqWindowIAttribute(attributes, "EQ_WINDOW_IATTR_HINT_HEIGHT",
                               "window_height", DEFAULT_WINDOW_HEIGHT);
    }

    /* Restoring the default Equalizer config to its saved value before
       parsing the command line options */
    if (!_defaultEQConfig.empty())
        eq::Global::setConfig(_defaultEQConfig);

    /* Initializing Equalizer */
    if (!eq::init(_argv.size(), &_argv[0], this))
    {
        throw std::runtime_error("RTNeuron: Equalizer init failed");
    }
    if (!eq::admin::init(_argv.size(), &_argv[0]))
    {
        std::cerr << "RTNeuron: Could not initialize the Equalizer "
                     "administrative layer. Runtime reconfiguration will not "
                     "be possible."
                  << std::endl;
    }

    s_instantiated = true;
}

RTNeuron::_Impl::~_Impl()
{
    LB_TS_THREAD(_controlThread);
    exitConfig();

    eq::admin::exit();
    // cppcheck-suppress unreachableCode
    eq::exit();

    s_instantiated = false;
}

void RTNeuron::_Impl::createEqualizerConfig(const std::string& config)
{
    LB_TS_THREAD(_controlThread);

    /* Exiting any previous equalizer configuration */
    exitConfig();

    /* Create and initialize the Equalizer client */
    try
    {
        if (!config.empty())
        {
            /* Storing the current default config file to restore it later on */
            _defaultEQConfig = eq::Global::getConfig();
            eq::Global::setConfig(config);
        }
        _client = new osgEq::Client(_argv.size(), &_argv[0], this);
        _client->start();
    }
    catch (std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void RTNeuron::_Impl::exitConfig()
{
    LB_TS_THREAD(_controlThread);
    if (_client)
    {
        _client->exitConfig();
        /* No mutex is needed here because no internal threading
           should be working at this point */
        _client = 0;
    }
    assert(_views.empty());
}

void RTNeuron::_Impl::useLayout(const std::string& name)
{
    LB_TS_THREAD(_controlThread);
    if (!_client)
        throw std::runtime_error("No configuration ready");
    osgEq::Config* config = _client->getConfig();
    config->useLayout(name);
}

void RTNeuron::_Impl::registerScene(const ScenePtr& scene)
{
    LB_TS_THREAD(_controlThread);

    /* In mutual exclusion with createNode and init */
    lunchbox::ScopedWrite mutex(_lock);

    /* Cleaning up unused scenes references */
    for (Scenes::reverse_iterator s = _scenes.rbegin(); s != _scenes.rend();)
    {
        Scenes::reverse_iterator t = s++;
        if (!t->lock())
            _scenes.erase(--t.base());
    }
    _scenes.push_back(scene);

    scene->simulationUpdated.connect(
        boost::bind(&_Impl::_onSimulationUpdated, this, _1));

    /* Adding the scene to the nodes */
    for (Nodes::iterator n = _nodes.begin(); n != _nodes.end(); ++n)
        (*n)->registerScene(scene->getSceneProxy());
}

void RTNeuron::_Impl::pause()
{
    LB_TS_THREAD(_controlThread);
    if (_client)
        _client->pause();
}

void RTNeuron::_Impl::resume()
{
    LB_TS_THREAD(_controlThread);
    if (_client)
        _client->resume();
}

void RTNeuron::_Impl::frame()
{
    LB_TS_THREAD(_controlThread);
    if (_client)
        _client->frame();
}

void RTNeuron::_Impl::waitFrame()
{
    LB_TS_THREAD(_controlThread);
    if (_client)
        _client->waitFrame();
}

void RTNeuron::_Impl::waitFrames(const unsigned int frames)
{
    LB_TS_THREAD(_controlThread);
    if (_client)
        _client->waitFrames(frames);
}

void RTNeuron::_Impl::wait()
{
    LB_TS_THREAD(_controlThread);
    if (_client)
        _client->wait();
}

void RTNeuron::_Impl::waitRecord()
{
    LB_TS_THREAD(_controlThread);
    _isRecording.waitEQ(false);
}

void RTNeuron::_Impl::record(const RecordingParams& params)
{
    LB_TS_THREAD(_controlThread);
    /* Freezing the rendering during the preparation of the recording */

    pause();
    waitFrame();

    {
        lunchbox::ScopedWrite mutex(_lock);
        _saveIdleAAState();
    }

    /* Setting up the camera manipulator for the camera path */
    CameraPathManipulatorPtr manip;
    if (params.cameraPath)
    {
        manip.reset(new CameraPathManipulator());
        manip->setFrameDelta(params.cameraPathDelta);
        manip->setPath(params.cameraPath);

        if (params.stopAtCameraPathEnd)
        {
            _recordingTimer.setStartTick(osg::Timer::instance()->tick());
            _recordedPathFrames = 0;
            _cameraPathDelta = params.cameraPathDelta;
            _recordingCameraPath = params.cameraPath;
        }
    }

    LBLOG(LOG_FRAME_RECORDING) << "Start recording" << std::endl;

    _recordFrames = params.frameCount;

    /** \bug What if the layout is changed while recording? The operation
        above only sets record to true to active views */
    const ViewPtrs views = getViews();
    for (const ViewPtr& view : views)
    {
        if (manip)
            view->setCameraManipulator(manip);
        AttributeMap& attributes = view->getAttributes();
        if (!params.filePrefix.empty())
        {
            if (views.size() == 1)
                attributes.set("output_file_prefix", params.filePrefix);
            else
                attributes.set("output_file_prefix",
                               params.filePrefix + "_" +
                                   view->getEqView()->getName());
        }
        if (!params.fileFormat.empty())
            attributes.set("output_file_format", params.fileFormat);

        view->record(true);
    }

    if (params.simulationDelta != 0)
    {
        Notifier notifier(this);
        _simulationStart = params.simulationStart;
        _simulationEnd = params.simulationEnd;
        _simulationDelta = params.simulationDelta;
        if (_playbackState == SimulationPlayer::FINISHED)
            _playbackState = SimulationPlayer::PLAYING;
        _setSimulationTimestamp(params.simulationStart);
    }

    _isRecording = true;

    resume();
}

void RTNeuron::_Impl::setSimulationTimestamp(const double milliseconds)
{
    Notifier notifier(this); /* Acquires lock */

    if (_autoAdjustSimulationWindow)
        _adjustSimulationWindow();

    _checkTimestamp(milliseconds);

    if (_playbackState == SimulationPlayer::FINISHED)
        _playbackState = SimulationPlayer::PLAYING;

    _setSimulationTimestamp(milliseconds);
}

double RTNeuron::_Impl::getSimulationTimestamp() const
{
    lunchbox::ScopedRead mutex(_lock);
    return _displayedTimestamp;
}

void RTNeuron::_Impl::setSimulationWindow(double start, double end)
{
    Notifier notifier(this); /* Acquires lock */

    _simulationStart = start;
    _simulationEnd = end;
    const double nextTimestamp = std::min(end, std::max(start, _nextTimestamp));
    if (nextTimestamp != _nextTimestamp)
        _setSimulationTimestamp(nextTimestamp);
    getAttributes().set("auto_adjust_simulation_window", false);
}

void RTNeuron::_Impl::getSimulationWindow(double& start, double& end) const
{
    lunchbox::ScopedRead mutex(_lock);
    start = _simulationStart;
    end = _simulationEnd;
}

void RTNeuron::_Impl::adjustSimulationWindow()
{
    Notifier notifier(this); /* Acquires lock */
    double start = NaN;
    double end = NaN;
    bool updateStreams = _playbackState == SimulationPlayer::PAUSED;
    MergeSimulationWindows operation(start, end, updateStreams);
    _visitReports(operation);

    if (std::isnan(start))
    {
        /* This can only happen if no active scene has reports, the window
           is undefined. */
        throw std::runtime_error(
            "No report found to adjust the simulation window");
    }

    if (_simulationStart != start || _simulationEnd != end)
    {
        _simulationStart = start;
        _simulationEnd = end;
    }
}

void RTNeuron::_Impl::setSimulationBeginTime(const double milliseconds)
{
    lunchbox::ScopedWrite mutex(_lock);
    _simulationStart = milliseconds;
    _nextTimestamp =
        std::min(_simulationEnd, std::max(_simulationStart, _nextTimestamp));
}

double RTNeuron::_Impl::getSimulationBeginTime() const
{
    lunchbox::ScopedRead mutex(_lock);
    return _simulationStart;
}

void RTNeuron::_Impl::setSimulationEndTime(const double milliseconds)
{
    lunchbox::ScopedWrite mutex(_lock);
    _simulationEnd = milliseconds;
    _nextTimestamp =
        std::min(_simulationEnd, std::max(_simulationStart, _nextTimestamp));
}

double RTNeuron::_Impl::getSimulationEndTime() const
{
    lunchbox::ScopedRead mutex(_lock);
    return _simulationEnd;
}

void RTNeuron::_Impl::setSimulationDelta(double milliseconds)
{
    {
        Notifier notifier(this); /* Acquires lock */

        if (!_client || _simulationDelta == milliseconds)
            return;

        _simulationDelta = milliseconds;
        if (_playbackState != SimulationPlayer::PAUSED)
        {
            /* If playback was finished (not paused), it is restarted
               because the new simulation delta can make it possible to
               advance in the opposite direction to that in which the window
               edge was reached. */
            _playbackState = SimulationPlayer::PLAYING;
            LBLOG(LOG_SIMULATION_PLAYBACK)
                << "triggering frame with next_simulation_timestamp "
                << _nextTimestamp << std::endl;
            osgEq::Config* config = _client->getConfig();
            osgEq::FrameData& frameData = config->getFrameData();
            frameData.setFrameAttribute("next_simulation_timestamp",
                                        _nextTimestamp);
        }
    }

    /* Signals are sent before triggering redraw */

    /* Triggering redraw */
    {
        lunchbox::ScopedWrite mutex(_lock);
        osgEq::Config* config = _client->getConfig();
        config->sendEvent(osgEq::ConfigEvent::REQUEST_REDRAW);
    }
}

double RTNeuron::_Impl::getSimulationDelta() const
{
    lunchbox::ScopedRead mutex(_lock);
    return _simulationDelta;
}

void RTNeuron::_Impl::playSimulation()
{
    Notifier notifier(this); /* Acquires lock */

    if (_playbackState == SimulationPlayer::PLAYING)
        return;

    _saveIdleAAState();

    _playbackState = SimulationPlayer::PLAYING;

    if (!_client)
        return;

    if (_autoAdjustSimulationWindow)
        _adjustSimulationWindow();

    LBLOG(LOG_SIMULATION_PLAYBACK)
        << "triggering frame with next_simulation_timestamp " << _nextTimestamp
        << std::endl;
    osgEq::Config* config = _client->getConfig();
    osgEq::FrameData& frameData = config->getFrameData();
    frameData.setFrameAttribute("next_simulation_timestamp", _nextTimestamp);
    config->sendEvent(osgEq::ConfigEvent::REQUEST_REDRAW);
}

void RTNeuron::_Impl::pauseSimulation()
{
    Notifier notifier(this); /* Acquires lock */

    if (_playbackState == SimulationPlayer::PAUSED)
        return;

    _playbackState = SimulationPlayer::PAUSED;
    _restoreIdleAAState();
}

void RTNeuron::_Impl::preFrame()
{
    if (_client->getConfig()->isDone())
        return;

    assert(_client->isApplicationNode());

    /* Storing the active scene IDs in the frame data */
    osgEq::FrameData& frameData = _client->getConfig()->getFrameData();
    ViewPtrs views = getViews();
    frameData.activeScenes.clear();
    ScenePtrs activeScenes;
    for (const auto& view : views)
    {
        ScenePtr scene = view->getScene();
        if (scene)
        {
            activeScenes.push_back(scene);
            frameData.activeScenes.push_back(scene->getSceneProxy()->getID());
        }
    }

    /* Committing dirty scenes and annotating the commit number in the
       frame data */
    for (const auto& scene : activeScenes)
        scene->_impl->commit(frameData);
}

void RTNeuron::_Impl::postFrame()
{
    if (_client->getConfig()->isDone())
        return;

    assert(_client->isApplicationNode());

    /* Checking if the camera path or simulation window end has been reached
       during movie recording or if the requested number of frames has been
       rendererd. */
    _checkRecordingDone();

    {
        bool timestampChanged = false;
        lunchbox::ScopedWrite mutex(_lock);

        /** \bug This is not true as long there is latency between
            the invocation of this function and the actual frame display. */
        if (_displayedTimestamp != _timestampReady)
        {
            _displayedTimestamp = _timestampReady;
            timestampChanged = true;
        }

        const osgEq::Node* node = _client->getLocalNode();
        const osgEq::FrameData& frameData = node->getFrameData();
        bool dummy = false;
        const bool finish =
            _playbackState == SimulationPlayer::PLAYING &&
            frameData.getFrameAttribute("playback_finished", dummy);
        if (finish)
            _playbackState = SimulationPlayer::FINISHED;

        if (_player)
        {
            mutex.unlock();
            if (timestampChanged)
                _player->timestampChanged(_displayedTimestamp);
            if (finish)
                _player->playbackStateChanged(_playbackState);
        }
    }
    _parent->frameIssued();
}

void RTNeuron::_Impl::preNodeUpdate(const uint32_t frameNumber)
{
    /* For time multiplexing the simulation may not need to be updated
       in this frame but we are not considering this situation yet.
       For 2D, subscenes may be shared, that's the reason for not doing
       the update in the channel or the pipe but at node level. */
    _processSimulationTimestamp(frameNumber);
}

void RTNeuron::_Impl::init(osgEq::Config* config)
{
    lunchbox::ScopedWrite mutex(_lock);

    /* Registering the distributed objects of the scenes */
    for (Scenes::reverse_iterator s = _scenes.rbegin(); s != _scenes.rend();)
    {
        Scenes::reverse_iterator t = s++;
        ScenePtr scene = t->lock();
        /* Cleaning up unused scenes references */
        if (!scene)
            _scenes.erase(--t.base());
        else
        {
            scene->_impl->registerDistributedObjects(config, *_initData);
        }
    }
}

void RTNeuron::_Impl::exit(osgEq::Config*)
{
    /* We don't want this function to finish while some other one depending
       on the config is going on (e.g. _onSimulationUpdated). However we
       must give the opportunity to some functions to exit to avoid
       deadlocks. */
    {
        lunchbox::ScopedWrite mutex(_lock);
        for (Scenes::reverse_iterator s = _scenes.rbegin();
             s != _scenes.rend();)
        {
            Scenes::reverse_iterator t = s++;
            ScenePtr scene = t->lock();
            if (scene)
                scene->simulationUpdated.disconnect(
                    boost::bind(&_Impl::_onSimulationUpdated, this, _1));
        }
    }

    {
        lunchbox::ScopedWrite lock(_exitMutex);
        _exitingConfig = true;
        _exitCondition.wait(lock, [&] { return _releaseAtExit == 0; });
    }

    lunchbox::ScopedWrite mutex(_lock);

    /* Clearing all scene data, distributed objects and caches.
       Making sure that all references to OSG objects are removed before the
       GL contexts are finalized. */
    for (const auto& v : _views)
    {
        /* This is done to make sure that we can clear all the colormap
           textures during OpenGL context destruction. */
        ViewPtr view = v.second;
        view->getAttributes().clear();
        const ScenePtr scene = view->getScene();
        view->setScene(ScenePtr());
    }

    for (auto s : _scenes)
    {
        auto scene = s.lock();
        if (!scene)
            continue;
        scene->_impl->unmapDistributedObjects();
        scene->_impl->deregisterDistributedObjects(*_initData);
        scene->_invalidate();
    }
    _scenes.clear();

    Neuron::clearCaches();
}

void RTNeuron::_Impl::configInit(eq::Node* node)
{
    node->getConfig()->sendEvent(RTNeuronEvent::NODE_INIT) << node->getID();

    /* Mapping distributed objects for scenes */
    for (Scenes::reverse_iterator s = _scenes.rbegin(); s != _scenes.rend();)
    {
        Scenes::reverse_iterator t = s++;
        ScenePtr scene = t->lock();
        /* Cleaning up unused scenes references */
        if (!scene)
            _scenes.erase(--t.base());
        else
        {
            scene->_impl->mapDistributedObjects(
                static_cast<osgEq::Config*>(node->getConfig()));
        }
    }
}

void RTNeuron::_Impl::configExit(eq::Node* node)
{
    node->getConfig()->sendEvent(RTNeuronEvent::NODE_EXIT) << node->getID();

    if (_client)
    {
        /* Unmapping distributed objects for scenes. In the application
           node this has already being executed by exit when this point is
           reached. */
        for (Scenes::reverse_iterator s = _scenes.rbegin();
             s != _scenes.rend();)
        {
            Scenes::reverse_iterator t = s++;
            ScenePtr scene = t->lock();
            /* Cleaning up unused scenes references */
            if (!scene)
                _scenes.erase(--t.base());
            else
            {
                scene->_impl->unmapDistributedObjects();
            }
        }
    }
}

bool RTNeuron::_Impl::handleEvent(eq::EventICommand command)
{
    switch (command.getEventType())
    {
    case RTNeuronEvent::SIMULATION_READY:
    {
        _simulationReady(command.read<double>());
        break;
    }
    case RTNeuronEvent::PREPARING_SIMULATION:
    {
        const co::uint128_t nodeID = command.read<co::uint128_t>();
        unsigned int mappers = command.read<unsigned int>();
        lunchbox::ScopedWrite mutex(_lock);
        _mappersPerNode[nodeID] = mappers;
        LBLOG(LOG_SIMULATION_PLAYBACK) << "PREPARING_SIMULATION event, node "
                                       << nodeID << ", mappers " << mappers
                                       << std::endl;
        break;
    }
    case RTNeuronEvent::NODE_INIT:
    {
        const co::uint128_t nodeID = command.read<co::uint128_t>();
        lunchbox::ScopedWrite mutex(_lock);
        /* Annotating the node as contributing to the simulation mapping */
        _mappersPerNode[nodeID] = LB_UNDEFINED_UINT32;
        break;
    }
    case RTNeuronEvent::NODE_EXIT:
    {
        const co::uint128_t nodeID = command.read<co::uint128_t>();
        lunchbox::ScopedWrite mutex(_lock);
        _mappersPerNode.erase(nodeID);
        break;
    }
    default:;
    }
    return false;
}

void RTNeuron::_Impl::onActiveViewTextureUpdated(const unsigned textureID)
{
    _parent->textureUpdated(textureID);
}

void RTNeuron::_Impl::onActiveViewEventProcessorUpdated(QObject* processor)
{
    _activeViewEventProcessor = processor;
    _parent->eventProcessorUpdated();
}

void RTNeuron::_Impl::onFrameGrabbed(const eq::uint128_t& viewID,
                                     const std::vector<uint8_t>& data)
{
    for (auto view : getViews())
    {
        if (view->getEqView()->getID() == viewID)
            view->frameGrabbed(data);
    }
}

void RTNeuron::_Impl::onIdle()
{
    _parent->idle();
}

void RTNeuron::_Impl::onDone()
{
    _parent->exited();
}

void RTNeuron::_Impl::onLayoutChanged(const eq::Layout* oldLayout,
                                      const eq::Layout* newLayout)
{
    lunchbox::ScopedRead mutex(_lock);
    if (!oldLayout || oldLayout->getViews().empty())
        return;

    /* Checking if all the old views were using the same scene. */
    const osgEq::View* oldView =
        static_cast<const osgEq::View*>(oldLayout->getViews().front());
    const unsigned int sceneID = oldView->getSceneID();
    for (const auto view : oldLayout->getViews())
    {
        if (static_cast<const osgEq::View*>(view)->getSceneID() != sceneID)
        {
            LBWARN << "Layout changed, but the previous layout had "
                      "multiple scenes. The new views will be empty"
                   << std::endl;
            return;
        }
    }

    /* Finding the actual Scene object. */
    ScenePtr scene;
    ViewPtrs allViews = _getViews(false);
    for (const auto& view : allViews)
    {
        if (view->getEqView() == oldView)
        {
            scene = view->getScene();
            break;
        }
    }

    /* Updating all new views. */
    for (const auto v : newLayout->getViews())
    {
        const osgEq::View* eqView = static_cast<const osgEq::View*>(v);
        for (const auto& view : allViews)
        {
            if (view->getEqView() == eqView)
            {
                view->setScene(scene);
                break;
            }
        }
    }
}

ViewPtrs RTNeuron::_Impl::getViews(const bool onlyActive)
{
    lunchbox::ScopedRead mutex(_lock);
    return _getViews(onlyActive);
}

osgEq::Config* RTNeuron::_Impl::createConfig(eq::ServerPtr parent)
{
    /* Creating the init data and registering the distributed objects
       for scenes into it */
    _initData.reset(new InitData(_initAttributes));
    _initData->shareContext = _shareContext;

    osgEq::Config* config = new osgEq::Config(parent);
    config->setInitData(_initData);

    return config;
}

osgEq::Node* RTNeuron::_Impl::createNode(eq::Config* parent)
{
    lunchbox::ScopedWrite mutex(_lock);

    osgEq::Node* node = new osgEq::Node(parent);
    _nodes.push_back(node);

    /* Adding all known scenes to this node */
    for (Scenes::reverse_iterator s = _scenes.rbegin(); s != _scenes.rend();)
    {
        Scenes::reverse_iterator t = s++;
        ScenePtr scene = t->lock();
        /* Cleaning up unused scenes references */
        if (!scene)
            _scenes.erase(--t.base());
        else
            node->registerScene(scene->getSceneProxy());
    }

    return node;
}

eq::Pipe* RTNeuron::_Impl::createPipe(eq::Node* parent)
{
    return new osgEq::Pipe(parent);
}

eq::Window* RTNeuron::_Impl::createWindow(eq::Pipe* parent)
{
    return new osgEq::Window(parent);
}

eq::Channel* RTNeuron::_Impl::createChannel(eq::Window* parent)
{
    return new osgEq::Channel(parent);
}

eq::View* RTNeuron::_Impl::createView(eq::Layout* parent)
{
    lunchbox::ScopedWrite mutex(_lock);

    osgEq::View* view = new osgEq::View(parent);
    /* If the view is native, an RTNeuron API view is created and linked
       to the eq::View.
       No rtneuron::View object is created for osgEq::Views from pipes or
       channels because: 1. they are not needed at this level, 2. it has some
       undersired side effects with regard to view attribute handling.
    */
    if (!parent)
        return view;

    AttributeMapPtr viewAttributes = _initAttributes("view", AttributeMapPtr());
    _views[view] =
        ViewPtr(new View(view, _parent->shared_from_this(),
                         viewAttributes ? *viewAttributes : AttributeMap()));
    return view;
}

void RTNeuron::_Impl::releaseNode(eq::Node* node)
{
    {
        lunchbox::ScopedWrite mutex(_lock);

        Nodes::iterator n = std::find(_nodes.begin(), _nodes.end(),
                                      static_cast<osgEq::Node*>(node));
        assert(n != _nodes.end());
        _nodes.erase(n);
    }
    delete node;
}

void RTNeuron::_Impl::releaseView(eq::View* view)
{
    if (view->getLayout())
    {
        lunchbox::ScopedWrite mutex(_lock);
        /* The python wrapping can still be holding a (smart) pointer to
           the view we are deallocating so we need to invalidate the
           rtneuron::View object so it doesn't access its internal
           eq::View */
        ViewsIter v = _views.find(view);
        assert(v != _views.end());
        v->second->_invalidate();
        _views.erase(v);
        _idleAAState.erase(v->second.get());
    }
    delete view;
}

void RTNeuron::_Impl::onAttributeChangingImpl(
    const AttributeMap&, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    using namespace AttributeMapHelpers;

    if (name == "auto_adjust_simulation_window")
    {
        (void)(bool) parameters;
    }
    else if (name == "neuron_color")
    {
        std::string value;
        try
        {
            std::string t = parameters;
            value = t; /* Direct assignment fails to compile */
        }
        catch (...)
        {
            osg::Vec4 color;
            getRequiredColor(parameters, color);
            return;
        }
        if (value != "random" && value != "allrandom")
            throw std::runtime_error("Invalid symbolic value for color");
    }
    else if (name == "afferent_syn_color" || name == "efferent_syn_color")
    {
        osg::Vec4 color;
        getRequiredColor(parameters, color);
    }
    else if (name == "soma_radius")
    {
        (void)(double) parameters;
    }
    else if (name == "soma_radii")
    {
        AttributeMapPtr radii = parameters;
        for (AttributeMap::const_iterator i = radii->begin(); i != radii->end();
             ++i)
        {
            (void)(double) i->second;
        }
    }
    else if (name == "profile")
    {
        (void)(bool) parameters;
    }
    else
    {
        throw std::runtime_error("Unknown or inmmutable attribute: " + name);
    }
}

void RTNeuron::_Impl::onAttributeChangedImpl(const AttributeMap& map,
                                             const std::string& name)
{
    using namespace core::AttributeMapHelpers;

    if (name == "auto_adjust_simulation_window")
    {
        _autoAdjustSimulationWindow = map(name);
    }
    if (name == "neuron_color")
    {
        Globals::setDefaultNeuronColor(map);
    }
    else if (name == "efferent_syn_color")
    {
        osg::Vec4 color;
        getRequiredColor(map, name, color);
        Globals::setDefaultEfferentSynapseColor(color);
    }
    else if (name == "afferent_syn_color")
    {
        osg::Vec4 color;
        getRequiredColor(map, name, color);
        Globals::setDefaultAfferentSynapseColor(color);
    }
    else if (name == "soma_radius")
    {
        Globals::setDefaultSomaRadius((double)map(name));
    }
    else if (name == "soma_radii")
    {
        AttributeMapPtr radii = map(name);
        LBASSERT(radii);
        Globals::setDefaultSomaRadii(*radii);
    }
    else if (name == "profile")
    {
        AttributeMapPtr profile = map(name);
        LBASSERT(profile);
        Globals::setProfilingOptions(*profile);
    }
}

ViewPtrs RTNeuron::_Impl::_getViews(const bool onlyActive)
{
    ViewPtrs views;
    for (const auto& viewPair : _views)
    {
        eq::View* eqView = viewPair.first;
        if (!onlyActive || eqView->isActive())
            views.push_back(viewPair.second);
    }
    return views;
}

template <typename F>
void RTNeuron::_Impl::_visitScenes(const F& functor)
{
    ViewPtrs views = _getViews(true);
    typedef std::set<ScenePtr> Scenes;
    Scenes scenes;
    for (const auto& view : views)
    {
        const ScenePtr scene = view->getScene();
        if (scene)
            scenes.insert(scene);
    }
    for (const auto& scene : scenes)
        functor(*scene);
}

template <typename F>
void RTNeuron::_Impl::_visitReports(const F& functor, bool onlyStreamBased)
{
    _visitScenes([functor, onlyStreamBased](Scene& scene) {
        const core::SpikeReportPtr spikeReport = scene.getSpikeReport();
        const CompartmentReportPtr compartmentReport =
            scene.getCompartmentReport();

        if (onlyStreamBased)
        {
            if (spikeReport && spikeReport->hasEnded())
                functor(*spikeReport);
        }
        else
        {
            if (spikeReport)
                functor(*spikeReport);
            if (compartmentReport)
                functor(*compartmentReport);
        }
    });
}

class CheckTimestamp
{
public:
    CheckTimestamp(const double timestamp)
        : _timestamp(timestamp)
    {
    }

    void operator()(const SpikeReport& report) const
    {
        if (_timestamp > report.getEndTime())
            throw std::runtime_error(
                "Simulation timestamp out of report window");
    }

    void operator()(const brain::CompartmentReport& report) const
    {
        if (_timestamp > report.getMetaData().endTime)
            throw std::runtime_error(
                "Simulation timestamp out of report window");
    }
    const double _timestamp;
};

void RTNeuron::_Impl::_checkTimestamp(const double timestamp)
{
    if (std::isnan(_simulationStart) || timestamp < _simulationStart ||
        timestamp > _simulationEnd)
    {
        throw std::runtime_error(
            "Simulation timestamp out of the current simulation window");
    }
    _visitReports(CheckTimestamp(timestamp), true);
}

void RTNeuron::_Impl::_setSimulationTimestamp(const double timestamp)
{
    if (!_client)
        return;
    osgEq::Config* config = _client->getConfig();
    if (!config)
        return;
    osgEq::FrameData& frameData = config->getFrameData();
    frameData.setFrameAttribute("requested_simulation_timestamp", timestamp);
    _nextTimestamp = timestamp;

    config->sendEvent(osgEq::ConfigEvent::REQUEST_REDRAW);
}

void RTNeuron::_Impl::_processSimulationTimestamp(const uint32_t frameNumber)
{
    lunchbox::ScopedWrite mutex(_lock);

    osgEq::Node* node = _client->getLocalNode();
    const osgEq::FrameData& frameData = node->getFrameData();

    std::set<ScenePtr> scenes;
    _findActiveScenes(frameData.activeScenes, scenes);

    /* Checking if there's a current simulation timestamp to apply */
    double timestamp = 0;
    if (frameData.getFrameAttribute("requested_simulation_timestamp",
                                    timestamp) ||
        frameData.getFrameAttribute("current_simulation_timestamp", timestamp))
    {
        /* Applying requested timestamp */
        LBLOG(LOG_SIMULATION_PLAYBACK) << "map simulation data " << timestamp
                                       << std::endl;

        _timestampReady = timestamp;
        /* Cleared, so calls to _simulationReady triggerred by
           setSimulationTimestamp won't confuse playback if started. */
        _notifiedThreads.erase(timestamp);

        /* Update simulation on all active scenes.
           mapSimulation calls _onSimulationUpdated, unlocking the mutex
           to avoid a trivial deadlock */
        _lock.unlock();
        for (const auto& scene : scenes)
            scene->mapSimulation(frameNumber, timestamp);

        _lock.lock();
    }
    /* Trigger next update if a prediction is given */
    if (frameData.getFrameAttribute("next_simulation_timestamp", timestamp))
    {
        unsigned int mappers = 0;
        for (const auto& scene : scenes)
            mappers += scene->prepareSimulation(frameNumber + 1, timestamp);

        LBLOG(LOG_SIMULATION_PLAYBACK) << "prepare simulation data, timestamp "
                                       << timestamp << ", mappers " << mappers
                                       << std::endl;

        /* Notify how many mappers are working on this time stamp to know
           how many simulationUpdated signals will be sent.
           This number will be 0 for nodes which don't do any rendering. */
        node->getConfig()->sendEvent(RTNeuronEvent::PREPARING_SIMULATION)
            << node->getID() << mappers;
    }
}

void RTNeuron::_Impl::_onSimulationUpdated(const double timestamp)
{
    struct Monitor
    {
        Monitor(std::mutex& mutex, std::condition_variable& condition,
                unsigned int& counter_)
            : cond(condition)
            , lock(mutex)
            , counter(counter_)
        {
            ++counter;
        }

        ~Monitor()
        {
            --counter;
            if (counter == 0)
                cond.notify_all();
        }
        std::condition_variable& cond;
        lunchbox::ScopedWrite lock;
        unsigned int& counter;
    };
    Monitor monitor(_exitMutex, _exitCondition, _releaseAtExit);

    lunchbox::ScopedWrite mutex(_lock);
    if (!_client || _exitingConfig)
        return;

    /* _client->getConfig() can't be used in client nodes */
    osgEq::Config* config = _client->getLocalNode()->getConfig();

    LBLOG(LOG_SIMULATION_PLAYBACK) << "Simulation updated to " << timestamp
                                   << std::endl;

    config->sendEvent(RTNeuronEvent::SIMULATION_READY) << timestamp;
}

void RTNeuron::_Impl::_simulationReady(const double timestamp)
{
    Notifier notifier(this); /* Acquires lock */

    if (!_notifyTimestamp(timestamp))
        return;

    if (_autoAdjustSimulationWindow)
        _adjustSimulationWindow();

    _scheduleNextTimestamp(timestamp);
}

bool RTNeuron::_Impl::_notifyTimestamp(const double timestamp)
{
    unsigned int notifications = ++_notifiedThreads[timestamp];
    /* Checking if all nodes have notified have many mapper threads are
       using (this code has still some potential race conditions). */
    unsigned int expected = 0;
    for (SimulationMappersPerNode::const_iterator i = _mappersPerNode.begin();
         i != _mappersPerNode.end(); ++i)
    {
        if (i->second == LB_UNDEFINED_UINT32)
            return false; /* Some node didn't tell its number of mappers
                             yet, i.e. mapping didn't started. */
        expected += i->second;
    }

    LBLOG(LOG_SIMULATION_PLAYBACK) << "Simulation ready notified: timestamp "
                                   << timestamp << " notification "
                                   << notifications << '(' << expected << ')'
                                   << std::endl;

    /* Checking if all notifications have arrived. */
    if (notifications < expected)
        return false;

    _notifiedThreads.clear();

    /* Even if a new frame is not triggered, the current timestamp has
       to be made available to swap buffers and unblock mapper threads.
       Otherwise playback doesn't start again once it's paused.
       With setTimestamp, setting current_simulation_timestamp twice in
       consecutive frames is harmless. */
    osgEq::Config* config = _client->getConfig();
    osgEq::FrameData& frameData = config->getFrameData();
    frameData.setFrameAttribute("current_simulation_timestamp", timestamp);

    return true;
}

bool RTNeuron::_Impl::_adjustSimulationWindow()
{
    MergeSimulationWindows operation(_simulationStart, _simulationEnd,
                                     _playbackState ==
                                         SimulationPlayer::PAUSED);
    _visitReports(operation);

    return operation._changed;
}

void RTNeuron::_Impl::_scheduleNextTimestamp(const double current)
{
    /* Preparing the prefetching of next simulation timestamp if playback is
       on but only if not recording if off because recording requires timely
       simulation mapping. */
    if (_isRecording || _playbackState != SimulationPlayer::PLAYING)
        return;

    osgEq::Config* config = _client->getConfig();
    osgEq::FrameData& frameData = config->getFrameData();

    double finalTimestamp = current;
    /* If the user has made the timestamp jump, then the requested
       timestamp will be honored. */
    frameData.getFrameAttribute("requested_simulation_timestamp",
                                finalTimestamp);

    /* During streaming, the simulation window may have been adjusted in such
       a way that the current timestamp is outside it. The timestamp
       will be fixed if playing forward and only for the start time.
       When the timestamp has been given by the user _checkTimestamp
       ensures that it already falls within the simulation window.
    */
    if (_simulationDelta > 0 && finalTimestamp < _simulationStart)
        finalTimestamp = _simulationStart;

    const double next = finalTimestamp + _simulationDelta;
    const bool finished = next >= _simulationEnd || next < _simulationStart;
    LBLOG(LOG_SIMULATION_PLAYBACK) << finalTimestamp << ' ' << _simulationDelta
                                   << ' ' << _simulationStart << ' '
                                   << _simulationEnd << std::endl;

    if (!finished)
    {
        LBLOG(LOG_SIMULATION_PLAYBACK) << "Advancing simulation to: " << next
                                       << std::endl;
        frameData.setFrameAttribute("next_simulation_timestamp", next);
        _nextTimestamp = next;
    }
    else
    {
        /* Not setting the playback state to finished yet, otherwise it's
           reported too early. */
        LBLOG(LOG_SIMULATION_PLAYBACK) << "Window end reached" << std::endl;
        frameData.setFrameAttribute("playback_finished", true);
    }

    config->sendEvent(osgEq::ConfigEvent::REQUEST_REDRAW);
}

void RTNeuron::_Impl::_findActiveScenes(const std::vector<unsigned int>& ids,
                                        std::set<ScenePtr>& scenes)
{
    std::set<unsigned int> sceneIDs(ids.begin(), ids.end());
    for (Scenes::reverse_iterator s = _scenes.rbegin(); s != _scenes.rend();)
    {
        Scenes::reverse_iterator t = s++;
        ScenePtr scene = t->lock();
        /* Cleaning up unused scenes references */
        if (!scene)
            _scenes.erase(--t.base());
        else if (sceneIDs.find(scene->getSceneProxy()->getID()) !=
                 sceneIDs.end())
            scenes.insert(scene);
    }
}

void RTNeuron::_Impl::_checkRecordingDone()
{
    lunchbox::ScopedWrite mutex(_lock);

    /* This will be the new recording state. */
    const bool continueRecording = _continueRecording();

    /* Stop recording only if previous state was recording and new state is
       not recording. Otherwise user modifications of the record state of
       individual views will be overriden. */
    if (_isRecording.get() && !continueRecording)
    {
        LBLOG(LOG_FRAME_RECORDING) << "Stop recording" << std::endl;

        for (const auto& viewPair : _views)
        {
            eq::View* eqView = viewPair.first;
            if (eqView->getLayout() != 0 && eqView->isActive())
                viewPair.second->record(false);
        }
        _recordingCameraPath.reset();
        _recordFrames = 0;

        _restoreIdleAAState();
    }

    /* Issue another frame if recording */
    if (continueRecording)
    {
        osgEq::Config* config = _client->getConfig();
        config->sendEvent(osgEq::ConfigEvent::REQUEST_REDRAW);
    }

    _isRecording = continueRecording;
}

bool RTNeuron::_Impl::_continueRecording()
{
    bool recording = _isRecording.get();

    LBLOG(LOG_FRAME_RECORDING)
        << std::endl
        << "    frames left: " << _recordFrames << std::endl
        << "    simulation: [" << _simulationStart << ", " << _simulationEnd
        << "], delta: " << _simulationDelta << std::endl;

    /* Checking if the camera path end has been reached */
    if (_recordingCameraPath)
    {
        const double length = (_recordingCameraPath->getStopTime() -
                               _recordingCameraPath->getStartTime());

        LBLOG(LOG_FRAME_RECORDING)
            << "   path: length: " << length
            << ", recorded frames: " << _recordedPathFrames
            << ", delta: " << _cameraPathDelta << std::endl;

        ++_recordedPathFrames;
        /* The camera path interval is open at the right */
        if (_cameraPathDelta != 0)
            recording =
                length * 1000.0 / _cameraPathDelta > _recordedPathFrames;
        else
            recording = _recordingTimer.time_s() < length;
    }
    if (!recording)
        return false;

    /* Checking if we are doing simulation playback while recording */
    if (_simulationDelta != 0 && _simulationStart < _simulationEnd)
    {
        /* Continue recording if next timestamp is within the simulation
           window. */
        recording = !std::isnan(_nextTimestamp) &&
                    (_nextTimestamp + _simulationDelta < _simulationEnd &&
                     _nextTimestamp + _simulationDelta > _simulationStart);

        if (recording)
        {
            _nextTimestamp += _simulationDelta;
            /* Can't directly use setSimulationTimestamp due to threading
               asserts. No need for mutex because _continueRecording is
               already called with a lock. */
            osgEq::Config* config = _client->getConfig();
            osgEq::FrameData& frameData = config->getFrameData();
            frameData.setFrameAttribute("current_simulation_timestamp",
                                        _nextTimestamp);
        }
    }
    if (!recording)
        return false;

    if (_recordFrames != 0)
        recording = --_recordFrames != 0;

    return recording;
}

void RTNeuron::_Impl::_saveIdleAAState()
{
    LBINFO << "Setting idle anti-aliasing steps to 0 temporarily" << std::endl;

    for (const auto& v : _views)
    {
        ViewPtr view = v.second;
        AttributeMap& attributes = view->getAttributes();
        int steps = attributes("idle_AA_steps");
        attributes.set("idle_AA_steps", 0);
        _idleAAState[view.get()] = steps;
    }
}

void RTNeuron::_Impl::_restoreIdleAAState()
{
    LBINFO << "Restoring idle anti-aliasing state" << std::endl;
    for (auto viewSteps : _idleAAState)
    {
        View* view = viewSteps.first;
        view->getAttributes().set("idle_AA_steps", viewSteps.second);
    }
    _idleAAState.clear();
}

// RecordingParams -------------------------------------------------------------

RecordingParams::RecordingParams()
    : simulationStart(0)
    , simulationEnd(0)
    , simulationDelta(1)
    , cameraPathDelta(0)
    , filePrefix(core::DEFAULT_FRAME_FILE_NAME_PREFIX)
    , fileFormat(core::DEFAULT_FRAME_FILE_FORMAT)
    , stopAtCameraPathEnd(false)
    , frameCount(0)
{
}

// RTNeuron --------------------------------------------------------------------

/*
  Constructors
*/

RTNeuron::RTNeuron(const int argc, char* argv[], const AttributeMap& attributes)
    : _impl(new _Impl(argc, argv, attributes, this))
{
}

/*
  Destructor
*/

RTNeuron::~RTNeuron()
{
    finalize();
}

/*
  Member functions
*/

void RTNeuron::disconnectAllSlots()
{
    /* In boost::signals2 connections are not really destroyed when
       disconnect_all_slots is called. It actually happens during signal
       invokation (there might be a good reason, but it looks like a micro
       optimization).
       If we want to make sure that the signals are cleaned up we have to
       destroy them. This can be enforced with the swap trick below. */

    FrameIssuedSignal().swap(frameIssued);
    TextureUpdatedSignal().swap(textureUpdated);
    EventProcessorUpdatedSignal().swap(eventProcessorUpdated);
    IdleSignal().swap(idle);
    ExitedSignal().swap(exited);
    if (_impl->_player)
        _impl->_player->disconnectAllSlots();
}

void RTNeuron::finalize()
{
    delete _impl;
    _impl = 0;
}

void RTNeuron::init(const std::string& config)
{
    _impl->createEqualizerConfig(config);
}

void RTNeuron::createConfig(const std::string& config)
{
    init(config);
}

void RTNeuron::exit()
{
    _impl->exitConfig();
}

void RTNeuron::exitConfig()
{
    exit();
}

void RTNeuron::useLayout(const std::string& layoutName)
{
    _impl->useLayout(layoutName);
}

ViewPtrs RTNeuron::getViews()
{
    return _impl->getViews(true);
}

ViewPtrs RTNeuron::getAllViews()
{
    return _impl->getViews(false);
}

ScenePtr RTNeuron::createScene(const AttributeMap& attributes)
{
    LB_TS_THREAD(_impl->_controlThread);

    if (_impl->_client)
        throw std::runtime_error(
            "New scenes cannot be created if there is an "
            "active configuration");

    ScenePtr scene(new Scene(shared_from_this(), attributes));
    _impl->registerScene(scene);
    return scene;
}

SimulationPlayerPtr RTNeuron::getPlayer()
{
    lunchbox::ScopedWrite mutex(_impl->_lock);

    /* Created on demand because otherwise we cannot safely call
       shared_from_this. */
    if (!_impl->_player)
        _impl->_player.reset(new SimulationPlayer(shared_from_this()));
    return _impl->_player;
}

void RTNeuron::record(const RecordingParams& params)
{
    _impl->record(params);
}

void RTNeuron::pause()
{
    _impl->pause();
}

void RTNeuron::resume()
{
    _impl->resume();
}

void RTNeuron::frame()
{
    _impl->frame();
}

void RTNeuron::waitFrame()
{
    _impl->waitFrame();
}

void RTNeuron::waitFrames(const unsigned int frames)
{
    _impl->waitFrames(frames);
}

void RTNeuron::waitRecord()
{
    _impl->waitRecord();
}

AttributeMap& RTNeuron::getAttributes()
{
    return _impl->getAttributes();
}

std::string RTNeuron::getVersionString()
{
    std::stringstream version;
    version << "RTNeuron " << RTNEURON_VERSION_MAJOR << "."
            << RTNEURON_VERSION_MINOR << "." << RTNEURON_VERSION_PATCH
            << " (c) 2006-2016 Universidad Politécnica de Madrid,"
               " Blue Brain Project";
    return version.str();
}

QObject* RTNeuron::getActiveViewEventProcessor() const
{
    return _impl->_activeViewEventProcessor;
}

void RTNeuron::setShareContext(QOpenGLContext* shareContext)
{
    _impl->_shareContext = shareContext;
}

void RTNeuron::wait()
{
    _impl->wait();
}

// SimulationPlayer::_Impl -----------------------------------------------------

class SimulationPlayer::_Impl
{
public:
    /*--- Public constructors/destructor */

    _Impl(const RTNeuronPtr& application)
        : _application(application)
    {
    }

    /*--- Public member functions ---*/

    RTNeuronPtr getApplication()
    {
        RTNeuronPtr app = _application.lock();
        if (!app)
            throw std::runtime_error(
                "RTNeuron: application already deallocated");
        return app;
    }

private:
    /*--- Private member attributes ---*/

    std::weak_ptr<RTNeuron> _application;
};

// SimulationPlayer ------------------------------------------------------------

/*
  Constructors/destructor
*/

SimulationPlayer::SimulationPlayer(const RTNeuronPtr& application)
    : _impl(new _Impl(application))
{
}

SimulationPlayer::~SimulationPlayer()
{
    delete _impl;
}

void SimulationPlayer::setTimestamp(const double milliseconds)
{
    _impl->getApplication()->_impl->setSimulationTimestamp(milliseconds);
}

double SimulationPlayer::getTimestamp() const
{
    return _impl->getApplication()->_impl->getSimulationTimestamp();
}

void SimulationPlayer::setWindow(const double begin, const double end)
{
    _impl->getApplication()->_impl->setSimulationWindow(begin, end);
}

void SimulationPlayer::getWindow(double& begin, double& end) const
{
    _impl->getApplication()->_impl->getSimulationWindow(begin, end);
}

void SimulationPlayer::adjustWindow()
{
    _impl->getApplication()->_impl->adjustSimulationWindow();
}

void SimulationPlayer::setBeginTime(const double milliseconds)
{
    _impl->getApplication()->_impl->setSimulationBeginTime(milliseconds);
}

double SimulationPlayer::getBeginTime() const
{
    return _impl->getApplication()->_impl->getSimulationBeginTime();
}

void SimulationPlayer::setEndTime(const double milliseconds)
{
    _impl->getApplication()->_impl->setSimulationEndTime(milliseconds);
}

double SimulationPlayer::getEndTime() const
{
    return _impl->getApplication()->_impl->getSimulationEndTime();
}

void SimulationPlayer::setSimulationDelta(const double milliseconds)
{
    _impl->getApplication()->_impl->setSimulationDelta(milliseconds);
}

double SimulationPlayer::getSimulationDelta() const
{
    return _impl->getApplication()->_impl->getSimulationDelta();
}

void SimulationPlayer::play()
{
    _impl->getApplication()->_impl->playSimulation();
}

void SimulationPlayer::pause()
{
    _impl->getApplication()->_impl->pauseSimulation();
}

void SimulationPlayer::disconnectAllSlots()
{
    /* Using the same trick that in RTNeuron::disconnectAllSlots. */
    PlaybackFinishedSignal().swap(finished);
    PlaybackStateChangedSignal().swap(playbackStateChanged);
    SimulationDeltaChangedSignal().swap(simulationDeltaChanged);
    WindowChangedSignal().swap(windowChanged);
    TimestampChangedSignal().swap(timestampChanged);
}
}
}
