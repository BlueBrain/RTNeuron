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

#include "View.h"
#include "ConfigEvent.h"
#include "DummyPointer.h"
#include "EventAdapter.h"
#include "Scene.h"
#include "SceneDecorator.h"

#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "render/ViewStyle.h"
#include "ui/EventHandler.h"

#include <osgGA/TrackballManipulator>

#include <eq/admin/admin.h>
#include <eq/channel.h>
#include <eq/config.h>
#include <eq/pipe.h>

#include <lunchbox/scopedMutex.h>

#include <boost/serialization/shared_ptr.hpp>

#include <condition_variable>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
const unsigned int DEFAULT_IDLE_AA_STEPS = 16;

/*
  Helper classes
*/

typedef std::map<co::uint128_t, unsigned int> PerPipeCounts;

// DestinationChannelCounter --------------------------------------------------

class DestinationChannelCounter : public eq::admin::NodeVisitor
{
public:
    DestinationChannelCounter(eq::View* view, PerPipeCounts& counts)
        : _view(view)
        , _counts(counts)
        , _count(0)
    {
    }

    typedef std::map<co::uint128_t, unsigned int> PerPipeCounts;

    virtual eq::admin::VisitorResult visit(eq::admin::Channel* channel)
    {
        if (channel->getViewVersion().identifier != _view->getID())
            return eq::admin::TRAVERSE_CONTINUE;

        if (channel->isDestination())
            ++_count;

        return eq::admin::TRAVERSE_CONTINUE;
    }

    virtual eq::admin::VisitorResult visitPre(eq::admin::Pipe*)
    {
        _count = 0;
        return eq::admin::TRAVERSE_CONTINUE;
    }

    virtual eq::admin::VisitorResult visitPost(eq::admin::Pipe* pipe)
    {
        _counts[pipe->getID()] = _count;
        return eq::admin::TRAVERSE_CONTINUE;
    }

private:
    eq::View* const _view;
    PerPipeCounts& _counts;
    unsigned int _count;
};

// View::SnapshotHelper -------------------------------------------------------

class View::SnapshotHelper
{
public:
    SnapshotHelper(View* view)
        : _view(view)
        , _completedPipes(0)
        , _done(false)
    {
    }

    void setFilename(const std::string& filename)
    {
        _filename = filename;
        /* If an all channel snapshot was requested we have to find out the
           number of destination channels per pipe. Otherwise we assume it's
           one. */
        if (_filename.find("%c") != std::string::npos)
            _countPerPipeChannels();

        _done = false;
        _completedPipes = 0;
    }

    std::string getFilename(const std::string& channelName) const
    {
        const size_t pos = _filename.find("%c");
        if (pos == std::string::npos)
            return _filename;

        std::string filename = _filename;
        filename.replace(pos, 2, channelName);
        return filename;
    }

    void snapshotDone(const eq::Channel& channel)
    {
        if (_channelCounts.empty())
        {
            _filename = "";
            return;
        }
        co::uint128_t pipeID = channel.getPipe()->getID();
        if (--_channelCounts[pipeID] == 0)
        {
            /* No more channels to capture in this pipe. */
            _filename = "";
        }
    }

    void pipeCompleted()
    {
        lunchbox::ScopedWrite lock(_mutex);

        if (_channelCounts.empty() ||
            ++_completedPipes == _channelCounts.size())
        {
            _done = true;
            _condition.notify_all();
        }
    }

    void wait()
    {
        lunchbox::ScopedWrite lock(_mutex);
        _condition.wait(lock, [&] { return _done; });
    }

    void serialize(co::DataOStream& os) const
    {
        os << _filename;
        os << _channelCounts.size();
        for (const auto& pipeCount : _channelCounts)
            os << pipeCount.first << pipeCount.second;
    }

    void deserialize(co::DataIStream& is)
    {
        is >> _filename;
        size_t counts;
        is >> counts;
        if (counts)
        {
            for (size_t i = 0; i != counts; ++i)
            {
                co::uint128_t pipe;
                unsigned int count;
                is >> pipe >> count;
                _channelCounts[pipe] = count;
            }
        }
    }

private:
    std::mutex _mutex;
    std::condition_variable _condition;

    /** Distributed members */
    std::string _filename;
    typedef std::map<co::uint128_t, unsigned int> PerPipeCounts;
    PerPipeCounts _channelCounts;

    /* Master only members */
    View* _view;

    eq::admin::ClientPtr _client;
    eq::admin::ServerPtr _server;
    size_t _completedPipes;
    bool _done;

    void _initAdminConnection()
    {
        if (_client)
            return;

        _client = new eq::admin::Client;
        if (!_client->initLocal(0, 0))
        {
            _client = 0;
            LBTHROW(std::runtime_error("Cannot initialize admin client"));
        }
        _server = new eq::admin::Server;
        if (!_client->connectServer(_server))
        {
            _client->exitLocal();
            _client = 0;
            _server = 0;
            LBTHROW(
                std::runtime_error("Cannot open connection to "
                                   "administrate server"));
        }
    }

    void _closeAdminConnection()
    {
        if (_client)
        {
            _client->disconnectServer(_server);
            _client->exitLocal();
            _client = 0;
        }
    }

    void _countPerPipeChannels()
    {
        _initAdminConnection();

        _channelCounts.clear();
        const eq::admin::Configs& configs = _server->getConfigs();
        /* This view comes from a configuration, so it's assumed that configs
           is not empty. */
        assert(!configs.empty());
        eq::admin::Config* config = configs.front();
        const eq::admin::Nodes& nodes = config->getNodes();
        /* The same goes for the nodes */
        assert(!nodes.empty());

        _channelCounts.clear();
        DestinationChannelCounter counter(_view, _channelCounts);
        for (eq::admin::Node* node : nodes)
            node->accept(counter);

        _closeAdminConnection();
    }
};

// View::Proxy ----------------------------------------------------------------

void View::Proxy::serialize(co::DataOStream& os, const uint64_t dirtyBits)
{
    if (dirtyBits & DIRTY_SCENE_ID)
        os << _view._sceneID;

    if (dirtyBits & DIRTY_MODELMATRIX)
        for (int i = 0; i != 16; ++i)
            os << _view._matrix.ptr()[i];

    if (dirtyBits & DIRTY_CLEARCOLOR)
        os << _view._clearColor[0] << _view._clearColor[1]
           << _view._clearColor[2] << _view._clearColor[3];

    if (dirtyBits & DIRTY_USE_DOF)
        os << _view._useDOF;

    if (dirtyBits & DIRTY_FOCAL_DISTANCE)
        os << _view._focalDistance;

    if (dirtyBits & DIRTY_FOCAL_RANGE)
        os << _view._focalRange;

    if (dirtyBits & DIRTY_USE_ROI)
        os << _view._useROI;

    if (dirtyBits & DIRTY_IDLE)
    {
        os << _view._maxIdleSteps;
        os << _view._snapshotAtIdle;
    }
    if (dirtyBits & DIRTY_DECORATOR_OBJECT)
    {
        net::DataOStreamArchive aos(os);
        aos << _view._decorator;
    }
    /* The previous bit excludes this one */
    else if (dirtyBits & DIRTY_DECORATOR)
    {
        net::DataOStreamArchive aos(os);
        /* There must be a way to avoid the static cast without creating
           a new derived object at deserialization, but I don't know it
           yet. */
        aos << static_cast<core::ViewStyle&>(*_view._decorator);
    }
    if (dirtyBits & DIRTY_POINTER)
    {
        if (_view._pointer)
        {
            os << true;
            _view._pointer->serialize(os);
        }
        else
        {
            os << false;
        }
    }
    if (dirtyBits & DIRTY_WRITE_FRAMES)
        os << _view._writeFrames;

    if (dirtyBits & DIRTY_GRAB_FRAME)
        os << _view._grabFrame;

    if (dirtyBits & DIRTY_SNAPSHOT)
        _view._snapshotHelper->serialize(os);

    if (dirtyBits & DIRTY_FILE_PREFIX)
        os << _view._filePrefix;

    if (dirtyBits & DIRTY_FILE_FORMAT)
        os << _view._fileFormat;

    if (dirtyBits & DIRTY_ORTHO)
        os << _view._useOrtho;
}

void View::Proxy::deserialize(co::DataIStream& is, const uint64_t dirtyBits)
{
    if (dirtyBits & DIRTY_SCENE_ID)
        is >> _view._sceneID;

    if (dirtyBits & DIRTY_MODELMATRIX)
        for (int i = 0; i != 16; ++i)
            is >> _view._matrix.ptr()[i];

    if (dirtyBits & DIRTY_CLEARCOLOR)
        is >> _view._clearColor[0] >> _view._clearColor[1] >>
            _view._clearColor[2] >> _view._clearColor[3];

    if (dirtyBits & DIRTY_USE_DOF)
        is >> _view._useDOF;

    if (dirtyBits & DIRTY_FOCAL_DISTANCE)
        is >> _view._focalDistance;

    if (dirtyBits & DIRTY_FOCAL_RANGE)
        is >> _view._focalRange;

    if (dirtyBits & DIRTY_USE_ROI)
        is >> _view._useROI;

    if (dirtyBits & DIRTY_IDLE)
    {
        is >> _view._maxIdleSteps;
        is >> _view._snapshotAtIdle;
        if (isMaster())
            setDirty(DIRTY_IDLE); /* Redistribute to slaves */
    }
    if (dirtyBits & DIRTY_DECORATOR_OBJECT)
    {
        net::DataIStreamArchive ais(is);
        if (_view._decoratorProtected)
        {
            /* The data still needs to be deserialized or the stream will
               be corrupted. */
            SceneDecoratorPtr tmp;
            ais >> tmp;
        }
        else
        {
            ais >> _view._decorator;
        }
    }
    /* The previous bit excludes this one */
    else if (dirtyBits & DIRTY_DECORATOR)
    {
        net::DataIStreamArchive ais(is);
        /* This deserialization is assumed to be harmless when the decorator
           is protected because clients and the application node must use
           the same decorator. */
        ais >> static_cast<core::ViewStyle&>(*_view._decorator);
    }
    if (dirtyBits & DIRTY_POINTER)
    {
        bool usePointer;
        is >> usePointer;

        if (usePointer)
        {
            if (!_view._pointer)
                /* Creating a dummy pointer to receive the updates. No signal
                   is connected because this object is just used to have the
                   geometry. */
                _view._pointer.reset(new DummyPointer());
            _view._pointer->deserialize(is);
        }
        else
        {
            _view._pointer.reset();
        }
    }
    if (dirtyBits & DIRTY_WRITE_FRAMES)
    {
        is >> _view._writeFrames;
        _view._resetFrameCounter = _view._writeFrames;
    }
    if (dirtyBits & DIRTY_GRAB_FRAME)
        is >> _view._grabFrame;

    if (dirtyBits & DIRTY_SNAPSHOT)
        _view._snapshotHelper->deserialize(is);

    if (dirtyBits & DIRTY_FILE_PREFIX)
        is >> _view._filePrefix;

    if (dirtyBits & DIRTY_FILE_FORMAT)
        is >> _view._fileFormat;

    if (dirtyBits & DIRTY_SNAPSHOT_DONE)
        _view._snapshotDoneOnPipe();

    if (dirtyBits & DIRTY_ORTHO)
        is >> _view._useOrtho;
}

// View -----------------------------------------------------------------------

View::View(eq::Layout* parent)
    : eq::View(parent)
    , _proxy(*this)
    , _eventQueue(new osgGA::EventQueue())
    , _resetManipulator(false)
    , _requiresContinuousUpdate(false)
    , _useOrtho(false)
    , _sceneID(UNDEFINED_SCENE_ID)
    , _useDOF(false)
    , _focalDistance(100.f)
    , _focalRange(200.f)
    , _useROI(false)
    , _maxIdleSteps(DEFAULT_IDLE_AA_STEPS)
    , _snapshotAtIdle(true)
    , _writeFrames(false)
    , _grabFrame(false)
    , _resetFrameCounter(false)
    , _fileFormat("png")
    , _snapshotHelper(new SnapshotHelper(this))
    , _decoratorProtected(false)
{
    setUserData(&_proxy);
}

View::~View()
{
    delete _snapshotHelper;
    setUserData(0);
}

void View::copyAttributes(const View& other)
{
    lunchbox::ScopedWrite mutex(&_lock);
    lunchbox::ScopedRead mutex2(&other._lock);

    /* Scene ID. */
    _sceneID = other._sceneID;

    /* Camera attributes. */
    _useOrtho = other._useOrtho;
    _matrix = other._matrix;
    setModelUnit(other.getModelUnit());
    changeMode(other.getMode());

    /* Decoration attributes */
    _replaceDecorator(other._decorator->clone());
    _clearColor = other._clearColor;

    /* Rendering technique attributes */
    _useROI = other._useROI;
    _maxIdleSteps = other._maxIdleSteps;
    _snapshotAtIdle = other._snapshotAtIdle;

    _proxy.setDirty(Proxy::DIRTY_SCENE_ID | Proxy::DIRTY_MODELMATRIX |
                    Proxy::DIRTY_CLEARCOLOR | Proxy::DIRTY_USE_ROI |
                    Proxy::DIRTY_ORTHO | Proxy::DIRTY_IDLE |
                    Proxy::DIRTY_DECORATOR_OBJECT);
}

void View::pushEvent(EventAdapter* event)
{
    LB_TS_THREAD(_thread);
    if (event->getEventType() == osgGA::GUIEventAdapter::RESIZE)
        getConfig()->sendEvent(ConfigEvent::REQUEST_REDRAW);
    _eventQueue->addEvent(event);
}

void View::pushFrameEvent(const double timestamp)
{
    LB_TS_THREAD(_thread);
    _eventQueue->frame(timestamp);
}

void View::addEventHandler(const EventHandlerPtr& eventHandler)
{
    lunchbox::ScopedRead mutex(&_lock);
    _eventHandlers.push_back(eventHandler);
}

bool View::processEvents()
{
    LB_TS_THREAD(_thread);

    osgGA::EventQueue::Events events;
    OSGCameraManipulatorPtr manipulator;
    PointerPtr pointer;
    EventHandlers handlers;

    bool viewDirty = false;
    /* To avoid trivial dead-locks due to double locks processing the events,
       all the data needed to process them is copied and the process is
       done outside the critical section. */
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _eventQueue->takeEvents(events);
        manipulator = _manipulator;
        pointer = _pointer;
        handlers = _eventHandlers;

        if (manipulator)
        {
            if (_resetManipulator)
            {
                manipulator->home(_eventQueue->getTime());
                _matrix = manipulator->getInverseMatrix();
                viewDirty = true;
                _proxy.setDirty(Proxy::DIRTY_MODELMATRIX);
                _resetManipulator = false;
            }
        }
    }

    for (osgGA::EventQueue::Events::const_iterator e = events.begin();
         e != events.end(); ++e)
    {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 0)
        osgGA::Event* event = *e;
#else
        osgGA::GUIEventAdapter* event = *e;
#endif
        _handleEvent(static_cast<EventAdapter*>(event), handlers, manipulator);
    }

    _updatePointer(pointer, manipulator);

    if (viewDirty)
        modelviewDirty();

    /* The dirty bits of the proxy will be unset automatically once the
       changes are propagated to the rendering clients. */
    lunchbox::ScopedRead mutex(&_lock);
    return (_requiresContinuousUpdate ||
            _proxy.getDirty() != Proxy::DIRTY_NONE);
}

void View::dirty()
{
    lunchbox::ScopedWrite mutex(&_lock);
    /* The bit to set is arbitrary. */
    _proxy.setDirty(Proxy::DIRTY_MODELMATRIX);
    /** \todo call _requestRedraw()? */
}

void View::requestRedraw()
{
    bool viewDirty = false;
    {
        lunchbox::ScopedWrite mutex(&_lock);
        if (_manipulator.get())
        {
            osg::Matrix matrix = _manipulator->getInverseMatrix();
            if (matrix != _matrix)
            {
                _matrix = matrix;
                viewDirty = true;
            }
        }
        /* Here this bit is used to force a redraw, it doesn't really imply
           that the matrix has changed. */
        _proxy.setDirty(Proxy::DIRTY_MODELMATRIX);
    }
    _requestRedraw();
    if (viewDirty)
        modelviewDirty();
}

void View::setCameraManipulator(const OSGCameraManipulatorPtr& manipulator)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);

        _resetManipulator = true;
        /* Transferring the home position from the old manipulator to the
           new one. */
        if (_manipulator && manipulator)
        {
            osg::Vec3d eye, center, up;
            _manipulator->getHomePosition(eye, center, up);
            manipulator->setHomePosition(eye, center, up);
            /* Manipulator reset (home) delayed until event processing to know
               the current timestamp. */
        }
        _manipulator = manipulator;
        if (!manipulator)
            return;

        _manipulator->setAutoComputeHomePosition(false);
        /* Getting the initial position from the manipulator.
           (when there was already a manipulator set, the matrix may be
           different from the old one if the camera manipulator is a camera
           path) */
        _matrix = _manipulator->getInverseMatrix();
        _proxy.setDirty(Proxy::DIRTY_MODELMATRIX);
    }
    _requestRedraw();
    modelviewDirty();
}

OSGCameraManipulatorPtr View::getCameraManipulator()
{
    lunchbox::ScopedRead mutex(&_lock);
    return _manipulator;
}

void View::setUseOrtho(bool ortho)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _proxy.setDirty(Proxy::DIRTY_ORTHO);
        _useOrtho = ortho;
    }
    _requestRedraw();
}

bool View::isOrtho() const
{
    return _useOrtho;
}

void View::setOrthoWall(const eq::Wall& wall)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _proxy.setDirty(Proxy::DIRTY_ORTHO);
    _useOrtho = true;
    Frustum::setWall(wall);
}

void View::setPerspectiveWall(const eq::Wall& wall)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _proxy.setDirty(Proxy::DIRTY_ORTHO);
    _useOrtho = false;
    Frustum::setWall(wall);
}

void View::setProjection(const eq::Projection& projection)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _proxy.setDirty(Proxy::DIRTY_ORTHO);
    _useOrtho = false;
    Frustum::setProjection(projection);
}

void View::setPointer(const PointerPtr& pointer)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        if (_pointer)
            _pointer->dirty.disconnect(
                boost::bind(&View::_onPointerDirty, this));
        _pointer = pointer;
        _pointer->dirty.connect(boost::bind(&View::_onPointerDirty, this));
        _proxy.setDirty(Proxy::DIRTY_POINTER);
    }
    _requestRedraw();
}

PointerPtr View::getPointer()
{
    lunchbox::ScopedRead mutex(&_lock);
    return _pointer;
}

void View::setHomePosition(const osg::Vec3d& eye, const osg::Vec3d& center,
                           const osg::Vec3d& up)
{
    lunchbox::ScopedWrite mutex(&_lock);

    if (_manipulator.get())
    {
        _manipulator->setHomePosition(eye, center, up, false);
        /* Reset delayed until event processing to know the current timestamp */
        _resetManipulator = true;
        _requestRedraw();
    }
}

void View::setSceneID(unsigned int id)
{
    {
        /* Lock may not be needed if RefPtr is thread-safe */
        lunchbox::ScopedWrite mutex(&_lock);
        _sceneID = id;
        _proxy.setDirty(Proxy::DIRTY_SCENE_ID);
    }
    _requestRedraw();
}

void View::setClearColor(const osg::Vec4& color)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _clearColor = color;
        _proxy.setDirty(Proxy::DIRTY_CLEARCOLOR);
    }
    _requestRedraw();
}

const osg::Vec4& View::getClearColor() const
{
    /* The mutex ensures that the object is copied atomically */
    lunchbox::ScopedRead mutex(&_lock);
    return _clearColor;
}

void View::useDOF(const bool use)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _useDOF = use;
        _proxy.setDirty(Proxy::DIRTY_USE_DOF);
    }
    _requestRedraw();
}

void View::setFocalDistance(const float value)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _focalDistance = value;
        _proxy.setDirty(Proxy::DIRTY_FOCAL_DISTANCE);
    }
    _requestRedraw();
}

void View::setFocalRange(const float value)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _focalRange = value;
        _proxy.setDirty(Proxy::DIRTY_FOCAL_RANGE);
    }
    _requestRedraw();
}

void View::setDecorator(const SceneDecoratorPtr& decorator)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _replaceDecorator(decorator);
        _proxy.setDirty(Proxy::DIRTY_DECORATOR_OBJECT);
    }
    _requestRedraw();
}

const osg::Matrixd& View::getModelMatrix() const
{
    lunchbox::ScopedRead mutex(&_lock);
    return _matrix;
}

void View::setModelMatrix(const osg::Matrixd& matrix, const bool emitDirty)
{
    bool different;
    {
        lunchbox::ScopedWrite mutex(&_lock);
        different = _matrix != matrix;
        if (different)
        {
            _matrix = matrix;
            if (_manipulator.get() != 0)
            {
                _manipulator->setByInverseMatrix(matrix);
                /* This is not a fully satisfactory solution but it avoids
                   the problem of the camera position being overwritten
                   at the first frame. */
                _resetManipulator = false;
            }
            _proxy.setDirty(Proxy::DIRTY_MODELMATRIX);
        }
    }

    if (different)
    {
        _requestRedraw();
        if (emitDirty)
            modelviewDirty();
    }
}

void View::useROI(const bool use)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _useROI = use;
    _proxy.setDirty(Proxy::DIRTY_USE_ROI);
}

void View::setMaxIdleSteps(const unsigned int idle_)
{
    {
        const unsigned int idle = idle_ == 1 ? 0 : idle_;
        lunchbox::ScopedWrite mutex(&_lock);
        if (_maxIdleSteps == idle)
            return;
        _maxIdleSteps = idle;
        _proxy.setDirty(Proxy::DIRTY_IDLE);
    }
    _requestRedraw();
}

void View::setSnapshotAtIdle(const bool enable)
{
    lunchbox::ScopedWrite mutex(&_lock);
    if (_snapshotAtIdle == enable)
        return;
    _snapshotAtIdle = enable;
    _proxy.setDirty(Proxy::DIRTY_IDLE);
}

void View::setWriteFrames(const bool enable)
{
    lunchbox::ScopedWrite mutex(&_lock);
    if (_writeFrames != enable)
    {
        _writeFrames = enable;
        _proxy.setDirty(Proxy::DIRTY_WRITE_FRAMES);
    }
}

bool View::getWriteFrames() const
{
    lunchbox::ScopedRead mutex(&_lock);
    return _writeFrames;
}

void View::setGrabFrame(const bool enable)
{
    lunchbox::ScopedRead mutex(&_lock);
    if (_grabFrame != enable)
    {
        _grabFrame = enable;
        _proxy.setDirty(Proxy::DIRTY_GRAB_FRAME);
    }
}

bool View::getGrabFrame() const
{
    lunchbox::ScopedRead mutex(&_lock);
    return _grabFrame;
}

bool View::resetFrameCounter()
{
    lunchbox::ScopedWrite mutex(&_lock);
    bool reset = _resetFrameCounter;
    _resetFrameCounter = false;
    return reset;
}

void View::grabFrame()
{
    setGrabFrame(true);

    lunchbox::ScopedWrite mutex(&_lock);
    if (isAttached())
    {
        /* Assuming the Config is not ready for sending if the view is not
           attached */
        getConfig()->sendEvent(ConfigEvent::FORCE_REDRAW);
    }
}

void View::snapshot(const std::string& fileName, const bool waitForCompletion)
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _snapshotHelper->setFilename(fileName);

        _proxy.setDirty(Proxy::DIRTY_SNAPSHOT);

        if (isAttached())
        {
            /* Assuming the Config is not ready for sending if the view is not
               attached */
            getConfig()->sendEvent(ConfigEvent::FORCE_REDRAW);
        }
    }

    if (waitForCompletion)
        _snapshotHelper->wait();
}

std::string View::getSnapshotName(const std::string& channelName) const
{
    lunchbox::ScopedRead mutex(&_lock);
    return _snapshotHelper->getFilename(channelName);
}

void View::snapshotDone(const eq::Channel& channel)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _snapshotHelper->snapshotDone(channel);
    _proxy.setDirty(Proxy::DIRTY_SNAPSHOT_DONE);
}

void View::setFilePrefix(const std::string& prefix)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _filePrefix = prefix;
    _proxy.setDirty(Proxy::DIRTY_FILE_PREFIX);
}

std::string View::getFilePrefix() const
{
    lunchbox::ScopedRead mutex(&_lock);
    return _filePrefix;
}

void View::setFileFormat(const std::string& extension)
{
    lunchbox::ScopedWrite mutex(&_lock);
    _fileFormat = extension;
    _proxy.setDirty(Proxy::DIRTY_FILE_FORMAT);
}

std::string View::getFileFormat() const
{
    lunchbox::ScopedRead mutex(&_lock);
    return _fileFormat;
}

co::uint128_t View::commit(const uint32_t incarnation)
{
    lunchbox::ScopedWrite mutex(&_lock);
    co::uint128_t ret = eq::View::commit(incarnation);
    return ret;
}

void View::notifyFrustumChanged()
{
    /* This shouldn't be needed, but somehow, the redraw doesn't occur
       automatically when the frustum change is notified. */
    eq::View::notifyFrustumChanged();
    _requestRedraw();
}

void View::_requestRedraw()
{
    if (!isAttached())
        /* Assuming the Config is not ready for sending if the view is not
           attached */
        return;
    getConfig()->sendEvent(ConfigEvent::REQUEST_REDRAW);
}

void View::_onPointerDirty()
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _proxy.setDirty(Proxy::DIRTY_POINTER);
    }
    _requestRedraw();
}

void View::_snapshotDoneOnPipe()
{
    _snapshotHelper->pipeCompleted();
}

void View::_onDecoratorDirty()
{
    {
        lunchbox::ScopedWrite mutex(&_lock);
        _proxy.setDirty(Proxy::DIRTY_DECORATOR);
    }
    _requestRedraw();
}

void View::_handleEvent(EventAdapter* event, const EventHandlers& handlers,
                        const OSGCameraManipulatorPtr& manipulator)
{
    for (EventHandlers::const_iterator i = handlers.begin();
         i != handlers.end(); ++i)
    {
        if ((*i)->handle(*event, *this))
            return;
    }

    if (manipulator)
        manipulator->handle(*event, *this);
}

void View::_updatePointer(PointerPtr pointer,
                          const OSGCameraManipulatorPtr& manipulator)
{
    if (pointer)
    {
        if (manipulator)
            pointer->setManipMatrix(manipulator->getMatrix());
        pointer->update();
    }
}

void View::_replaceDecorator(const SceneDecoratorPtr& decorator)
{
    if (_decorator)
        _decorator->dirty.disconnect(
            boost::bind(&View::_onDecoratorDirty, this));
    _decorator = decorator;
    _decorator->dirty.connect(boost::bind(&View::_onDecoratorDirty, this));
    _decoratorProtected = true;
}
}
}
}
