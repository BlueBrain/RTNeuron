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

#ifndef RTNEURON_OSGEQ_CLIENT_H
#define RTNEURON_OSGEQ_CLIENT_H

#include <osg/ref_ptr>

#include <eq/eq.h>
#include <lunchbox/monitor.h>

#include <list>
#include <thread>

namespace osg
{
class FrameStamp;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Layout;
class Config;
class Node;
class Application;

//! The application definition of the local node eq::Client
class Client : public eq::Client
{
public:
    friend class osgEq::Node;

    /*--- Public constructors/destructor ---*/

    /**
       Init the Equalizer configuration.

       If no error occurs, this function returns when the application (or
       rendering client loop) is started.

       A reference to argv is stored internally and shouldn't be modified
       from outside.
     */
    Client(const int argc, char* argv[], Application* application);

    virtual ~Client();

    /*--- Public member functions ---*/

    /**
       Returns true if this Client is representing the master application node.
     */
    bool isApplicationNode() { return !_isRenderingClient; }
    bool start();

    /**
       Block until the application (or rendering client) loop is finished.

       To avoid race conditions this function should only be called by
       the thread that creates the Client.
     */
    void wait();

    /**
       Mask redraws requests so they cannot trigger frames and finish all
       pending frames.
     */
    void pause();

    /**
       Unmask redraw requests events and trigger a redraw if a request was
       pending.
     */
    void resume();

    void frame();

    /**
       Waits until a new frame is finished or the configuration is exited.
     */
    void waitFrame();

    /**
       Waits until n frames are finished or the configuration is exited.

       This function call implies a call to resume.
     */
    void waitFrames(const unsigned int frames);

    /**
       In the application node it finishes the rendering loop and exits
       the configuration, in rendering clients it has no effect.
       This methods blocks until all resources are freed.

       Do not call any other methods after this one.
    */
    void exitConfig();

    const osg::FrameStamp* getFrameStamp() const { return _frameStamp.get(); }
    /**
       In the application node it returns the config object if already running.
       In rendering clients it always returns 0
       (Don't use this method to distingish between application and client
       nodes use Client::isApplicationNode instead).
    */
    Config* getConfig() { return _config; }
    Application* getApplication() { return _application; }
    osgEq::Node* getLocalNode() { return _localNode; }
    const osgEq::Node* getLocalNode() const { return _localNode; }
protected:
    /*--- Protected member attributes ---*/

    Application* _application;
    Config* _config;
    osgEq::Node* _localNode;

    eq::ServerPtr _server;

    bool _isRenderingClient;
    lunchbox::Monitor<bool> _innerLoopRunning;

    std::mutex _frameMutex;
    std::condition_variable _frameCondition;

    /* Read/write from config within mutual exclusions sections for
       synchronization of frame and waitFrame */
    unsigned int _currentFrame;
    unsigned int _finishedFrame;

    std::thread _thread;

    /* Only used for initLocal in client nodes */
    int _argc;
    char** _argv;

    lunchbox::Monitor<bool> _applicationLoopRunning;

    osg::ref_ptr<osg::FrameStamp> _frameStamp;

    /*--- Protected member functions ---*/

    /* Overridden to not call eq::exit and ::exit from the internal
       thread and let the main Python thread do it instead. */
    virtual void exitClient();

private:
    /*--- Private member functions ---*/

    void _applicationNodeInit();

    void _renderingClientInit();

    void _run();

    /** Overriden only to know when renderingClientInit has finished the
        initialization and the client loop has been started. */
    void clientLoop();

    void _applicationLoop();

    void _waitForRedrawRequest();

    bool _continueRendering() const;
};

typedef lunchbox::RefPtr<Client> ClientPtr;
}
}
}
#endif
