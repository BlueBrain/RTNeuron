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

#include <QApplication>
#include <eq/gl.h> // Don't move down glew must be included before GL

#include "Application.h"
#include "Client.h"
#include "Config.h"
#include "ConfigEvent.h"
#include "NodeViewer.h"
#include "config/Globals.h"
#include "util/log.h"

#include <eq/eq.h>

#include <boost/filesystem/operations.hpp>
#ifndef WIN32
#include <boost/tokenizer.hpp> // For LD_PRELOAD filtering
#include <sys/resource.h>
#include <sys/time.h>
#endif
#include <boost/date_time/posix_time/posix_time.hpp>

#include <fstream>
#include <stdexcept>

namespace bp = boost::posix_time;

extern char** environ;

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*
  Static definitions
*/

/*
  Helper classes
*/
namespace hidden_to_doxygen
{
enum NodeCommand
{
    CMD_REGISTER_CLIENT_NODE = eq::fabric::CMD_CLIENT_CUSTOM
};
}
using namespace hidden_to_doxygen;

/*
  Constructors
*/

Client::Client(const int argc, char* argv[], Application* application)
    : _application(application)
    , _config(0)
    , _localNode(0)
    , _isRenderingClient(false)
    , _innerLoopRunning(false)
    , _currentFrame(0)
    , _finishedFrame(0)
    , _frameStamp(new osg::FrameStamp())
{
    for (int i = 1; i < argc; ++i)
    {
        /* Detecting if this is a client node */
        if (!strcmp(argv[i], "--eq-client"))
            _isRenderingClient = true;
    }

    _argc = argc;
    _argv = argv;
}

/*
  Destructor
*/

Client::~Client()
{
    exitConfig();
    _thread.join();
}

/*
  Member functions
*/

bool Client::start()
{
    _thread = std::thread(&Client::_run, this);

    /* Wait until the client or application threads are started. */
    if (QApplication::instance())
    {
        /* as we are in the main thread here and we need to wait for Qt events
           for the QGLWidget creation in Equalizer, give Qt a chance to handle
           those events */
        while (!_innerLoopRunning.timedWaitEQ(true, 100))
            QApplication::instance()->processEvents();
    }
    else
        _innerLoopRunning.waitEQ(true);

    return true;
}

void Client::wait()
{
    _innerLoopRunning.waitEQ(false);
}

void Client::pause()
{
    assert(!_isRenderingClient);
    _config->sendEvent(osgEq::ConfigEvent::REDRAW_MASK) << true;
}

void Client::resume()
{
    assert(!_isRenderingClient);
    _config->sendEvent(osgEq::ConfigEvent::REDRAW_MASK) << false;
}

void Client::frame()
{
    assert(!_isRenderingClient);
    lunchbox::ScopedWrite lock(_frameMutex);

    if (!_config)
        /* This can happen if the configuration has been exited pressing ESC. */
        return;

    const unsigned int current = _config->getCurrentFrame();
    _config->sendEvent(osgEq::ConfigEvent::FORCE_REDRAW);

    /* Waiting until a new frame is triggered.
       \todo Will make this wait until a previous frame is finished? */
    _frameCondition.wait(lock, [&] { return current != _currentFrame; });
}

void Client::waitFrame()
{
    assert(!_isRenderingClient);
    lunchbox::ScopedWrite lock(_frameMutex);
    _frameCondition.wait(lock, [&] {
        if (_config)
            _config->sendEvent(osgEq::ConfigEvent::FINISH_ALL_FRAMES);
        return !_config || _finishedFrame == _currentFrame;
    });
}

void Client::waitFrames(const unsigned int frames)
{
    assert(!_isRenderingClient);
    lunchbox::ScopedWrite lock(_frameMutex);
    unsigned int currentFinished = _finishedFrame;
    _config->sendEvent(osgEq::ConfigEvent::REDRAW_MASK) << false;
    _frameCondition.wait(lock, [&] {
        return !_config || _finishedFrame >= currentFinished + frames;
    });
}

void Client::exitConfig()
{
    if (!_config)
        return;

    if (!_isRenderingClient)
        _config->setDone();
    wait();
    assert(_config == 0);
}

void Client::exitClient()
{
    bool ret = exitLocal();
    LBINFO << "Exit " << lunchbox::className(this) << " process used "
           << getRefCount() << std::endl;
    if (!ret)
        LBWARN << "Client process didn't exit cleanly" << std::endl;

    /* Now it's safe to unlock the application thread waiting in
       Client::wait(). */
    _innerLoopRunning = false;

    /* eq::exit is expected to be called from the main thread */
}

void Client::_applicationNodeInit()
{
    if (!initLocal(_argc, _argv))
        throw std::runtime_error(
            "RTNeuron: Couldn't initialize the master node networking");

    /* Connecting to the server. */
    _server = new eq::Server();
    if (!connectServer(_server))
        throw std::runtime_error("RTNeuron: Can't open Equalizer server");

    /* Adding a GPU filter to the configuration parameters. */
    eq::fabric::ConfigParams configParams;
    const char* gpuFilter = ::getenv("RTNEURON_AUTOCONFIG_GPU_FILTER");
    if (gpuFilter)
        configParams.setGPUFilter(gpuFilter);
    configParams.setRenderClientEnvPrefixes({"RTNEURON_", "OSGTRANSPARENCY_"});

    /* Getting the Config object. */
    _config = static_cast<Config*>(_server->chooseConfig(configParams));

    if (_config == 0)
    {
        /* Couldn't connect to the server */
        disconnectServer(_server);
        throw std::runtime_error("RTNeuron: Bad configuration");
    }

    if (!_config->init(eq::uint128_t()))
    {
        _server->releaseConfig(_config);
        disconnectServer(_server);
        _config = 0;
#if EQ_VERSION_LT(1, 7, 1)
        std::stringstream msg;
        msg << "RTNeuron:: Config initialization failed: "
            << _config->getError();
        throw std::runtime_error(msg.str().c_str());
#else
        throw std::runtime_error("RTNeuron:: Config initialization failed");
#endif
    }
}

void Client::_renderingClientInit()
{
#if !defined NDEBUG && defined __unix__
    /* Enabling core dumps in UNIX systems */
    const struct rlimit rlp = {RLIM_INFINITY, RLIM_INFINITY};
    ::setrlimit(RLIMIT_CORE, &rlp);
#endif
    if (!initLocal(_argc, _argv))
        throw std::runtime_error(
            "RTNeuron: Couldn't initialize "
            "the rendering client networking");
}

void Client::_run()
{
    /* Overriding the default value of draw serialization. Otherwise
       multihread configurations won't scale as expected. */
    osg::DisplaySettings::instance()->setSerializeDrawDispatch(false);

    if (_isRenderingClient)
    {
        _renderingClientInit();
    }
    else
    {
        /* We would like to do the init outside this thread, but this doesn't
           work in debug build due to all the thread safety checks present
           in config object. */
        _applicationNodeInit();
        _applicationLoop();
    }
}

void Client::clientLoop()
{
    /* Unlocking renderingClientInit */
    _innerLoopRunning = true;
    eq::Client::clientLoop();

    /* _innerLoopRunning will be set to false once exitClient has finished
       otherwise the application may start destroying objects too early. */
}

void Client::_applicationLoop()
{
    _innerLoopRunning = true;

    bp::ptime startTime = bp::microsec_clock::local_time();
    std::ofstream profileLog;
    if (core::Globals::isProfilingEnabled())
    {
        profileLog.open(core::Globals::getProfileLogfile());
        if (!profileLog)
        {
            LBWARN << "Couldn't create " << core::Globals::getProfileLogfile()
                   << " file to store profile measures " << std::endl;
        }
    }

    while (!_config->isDone())
    {
        /* Processing events before rendering.
           This is done to process the redraw request sent by RTNeuron::frame
           and clear it at frame start. */
        _waitForRedrawRequest();

        if (_config->isDone())
            break;

        _application->preFrame();
        {
            lunchbox::ScopedWrite lock(_frameMutex);
            /* We need mutual exclusion between waitFrame functions and
               startFrame because the current frame counter is increased
               inside the function after the frame has been emitted.
               In principle, startFrame does a round trip to the server but
               shouldn't wait indefinitely for any event. */
            _config->startFrame(eq::uint128_t());
            _currentFrame = _config->getCurrentFrame();
            _frameCondition.notify_all();
        }
        _config->finishFrame();
        _application->postFrame();

        /* Some inner functions may call handleEvents, so it's possible that
           by the end of the iteration finishAllFrames was called
           due to a FINISH_ALL_FRAMES event. Updating the last finished
           frame counter. */
        {
            lunchbox::ScopedWrite lock(_frameMutex);
            _finishedFrame = _config->getFinishedFrame();
            _frameCondition.notify_all();
        }

        if (core::Globals::isProfilingEnabled())
        {
            bp::ptime now = bp::microsec_clock::local_time();
            const float time = (now - startTime).total_milliseconds();
            startTime = now;
            profileLog << time << std::endl;
            lunchbox::enableHeader(profileLog);
        }
    }

    _config->finishAllFrames();

    if (core::Globals::isProfilingEnabled())
    {
        bp::ptime now = bp::microsec_clock::local_time();
        const float time = (now - startTime).total_milliseconds();
        profileLog << time << std::endl;
    }

    _application->exit(_config);
    _config->exit();

    /* Shutting down the server and disconnecting. */
    _server->releaseConfig(_config);
    disconnectServer(_server);
    _server = 0;
    _config = 0;

    exitLocal();

    _innerLoopRunning = false;

    /* This will unlock caller of waitFrames or waitFrame (which is needed
       because no new frames will be produced. */
    _frameCondition.notify_all();
}

void Client::_waitForRedrawRequest()
{
    while (!_config->needsRedraw())
    {
        if (hasCommands())
        {
            /* execute non-critical pending commands */
            processCommand();
        }
        else if (!_continueRendering())
        {
            /* Block on until a single user event occurs. Resuming the
               rendering loop also unblocks this thread. */
            const eq::EventICommand& event = _config->getNextEvent(1000);
            if (!event.isValid())
                _application->onIdle();
            else if (!_config->handleEvent(event))
                LBVERB << "Unhandled " << event << std::endl;
        }
        /* Process all pending events without blocking. */
        _config->handleEvents();

        /* handleEvents can finish all the frames */
        {
            lunchbox::ScopedWrite lock(_frameMutex);
            /* Signal only if _finishedFrame has changed */
            const unsigned int currenFrame = _finishedFrame;
            _finishedFrame = _config->getFinishedFrame();
            if (currenFrame != _finishedFrame)
                _frameCondition.notify_all();
        }

        if (_continueRendering())
            return;
    }
}

bool Client::_continueRendering() const
{
    /* This condition reads as follows:
       Continue rendering if continuous rendering is enabled and:
       - The client loop is not paused
       - or being paused an unmaskable redraw was requested. */
    return core::Globals::doContinuousRendering() &&
           (!_config->isRedrawMasked() || _config->needsRedraw());
}
}
}
}
