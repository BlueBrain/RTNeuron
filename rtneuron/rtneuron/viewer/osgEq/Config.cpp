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

#include "Config.h"
#include "Application.h"
#include "Client.h"
#include "ConfigEvent.h"
#include "EventAdapter.h"
#include "InitData.h"
#include "NodeViewer.h"
#include "View.h"

#ifdef RTNEURON_USE_VRPN
#include "Tracker.h"
#endif

#include <osg/Matrix>

#include <eq/admin/client.h>
#include <eq/admin/server.h>
#include <eq/eq.h>
#include <eq/gl.h>

#include <boost/assert.hpp>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*
  Constructors
*/
Config::Config(eq::ServerPtr parent)
    : eq::Config(parent)
    , _done(false)
    , _lastFocusedView(0)
    , _needsRedraw(false)
    , _needsRedrawUnmaskable(false)
    , _maskRedraw(true)
    , _remainingAAFrames(0)
{
}

/*
  Destructor
*/
Config::~Config()
{
}

/*
  Member functions
*/
bool Config::init(const eq::uint128_t&)
{
    /* This is only executed by the application node.
       The values of _initData have already been initialized in the node
       factory function. */

    registerObject(&_frameData);
    _initData->frameDataID = _frameData.getID();

    getClient()->getApplication()->init(this);

    registerObject(_initData.get());
    /* _initData->getID() is passed to Node::configInit at client nodes
       so they can map the distributed objects. */
    bool result = eq::Config::init(_initData->getID());

    if (!result)
    {
        deregisterObject(_initData.get());
        deregisterObject(&_frameData);
        return false;
    }

/* Creating the tracker object is required */
#ifdef RTNEURON_USE_VRPN
    try
    {
        if (!_initData->trackerDeviceName.empty())
            _tracker.reset(new Tracker(_initData->trackerDeviceName));
    }
    catch (...)
    {
        _tracker.reset(0);
    }
#else
    if (!_initData->trackerDeviceName.empty())
        std::cerr << "RTNeuron compiled without VRPN support, "
                     "head tracking not available"
                  << std::endl;
#endif

    _timer.setStartTick(osg::Timer::instance()->tick());

    return result;
}

bool Config::exit()
{
    const bool ret = eq::Config::exit();

    _initData->frameDataID = 0;
    deregisterObject(_initData.get());
    deregisterObject(&_frameData);

    return ret;
}

void Config::setDone()
{
    /* We close the admin server and client from here because doing it from
       exit is not thread safe (the admin objects are created from the main
       control thread and exit is called from the internal thread from Client */
    if (_adminServer)
    {
        _adminClient->disconnectServer(_adminServer);
        _adminClient->exitLocal();
        _adminClient = 0;
        _adminServer = 0;
    }

    _done = true;
    sendEvent(osgEq::ConfigEvent::FORCE_REDRAW);
    getClient()->getApplication()->onDone();
}

bool Config::mapDistributedObjects(const eq::uint128_t& initDataID)
{
    assert(_initData);
    if (_initData->isAttached())
    {
        /* This is executed only by the application node */
        _initData->isClient = false;
        assert(_initData->getID() == initDataID);
        return true;
    }

    /* This code is only run by clients */
    _initData->isClient = true;
    return syncObject(_initData.get(), initDataID, getApplicationNode());
}

void Config::unmapDistributedObjects()
{
}

uint32_t Config::startFrame(const eq::uint128_t&)
{
    _pushFrameEventToViews();

#ifdef RTNEURON_USE_VRPN
    /* Update VRPN tracker position. */
    if (_tracker.get() != 0)
        _setHeadMatrix(_tracker->sample());
#endif
    _frameData.isIdle =
        /** \todo Check all views to see if any has idle AA enabled. */
        !_needsRedraw && _remainingAAFrames > 0;

    _remainingAAFrames = 0;

    const eq::uint128_t commitID = _frameData.commit();
    _frameData.commits.clear();

#ifdef RTNEURON_USE_VRPN
    /* Continuous redraw if using a head tracker */
    _needsRedraw = _tracker.get() != 0;
#else
    _needsRedraw = false;
#endif
    _needsRedrawUnmaskable = false;

    return eq::Config::startFrame(commitID);
}

void Config::setInitData(const InitDataPtr& initData)
{
    _initData = initData;
}

Client* Config::getClient()
{
    return static_cast<Client*>(eq::Config::getClient().get());
}

bool Config::handleEvent(eq::EventICommand command)
{
    LB_TS_THREAD(_clientThread);

    if (command.getEventType() >= ConfigEvent::APPLICATION_EVENT)
        return getClient()->getApplication()->handleEvent(command);

    if (getClient()->getApplication()->handleEvent(command))
        return true;

    switch (command.getEventType())
    {
    case ConfigEvent::IDLE_AA_LEFT:
    {
        const int32_t steps = command.read<int32_t>();
        _remainingAAFrames = std::max(_remainingAAFrames, steps);
        return true;
    }
    case ConfigEvent::RECONFIGURE_REQUEST:
    {
        /* Update the Equalizer configuration */
        const bool success = update();
        /* Signal that the reconfiguration is complete. */
        if (command.read<bool>())
        {
            const uint32_t requestID = command.read<uint32_t>();
            getLocalNode()->serveRequest(requestID, success);
        }
        return true;
    }
    case ConfigEvent::REDRAW_MASK:
    {
        _maskRedraw = command.read<bool>();
        finishAllFrames();
        return true;
    }
    case ConfigEvent::FINISH_ALL_FRAMES:
        finishAllFrames();
        return true;
    case ConfigEvent::FORCE_REDRAW:
        _needsRedrawUnmaskable = true;
        return true;
    case ConfigEvent::REQUEST_REDRAW:
        _needsRedraw = true;
        return true;
    case ConfigEvent::ACTIVE_VIEW_EVENT_PROCESSOR:
    {
        QObject* processor = command.read<QObject*>();
        getClient()->getApplication()->onActiveViewEventProcessorUpdated(
            processor);
        return true;
    }
    case ConfigEvent::ACTIVE_VIEW_TEXTURE:
    {
        const unsigned textureID = command.read<unsigned>();
        getClient()->getApplication()->onActiveViewTextureUpdated(textureID);
        return true;
    }
    case ConfigEvent::GRAB_FRAME:
    {
        const eq::uint128_t& viewID = command.read<eq::uint128_t>();
        const uint64_t size = command.read<uint64_t>();
        if (size == 0)
        {
            getClient()->getApplication()->onFrameGrabbed(
                viewID, std::vector<uint8_t>());
            return true;
        }
        const uint8_t* data =
            reinterpret_cast<const uint8_t*>(command.getRemainingBuffer(size));
        const std::vector<uint8_t> dataVec(data, data + size);
        getClient()->getApplication()->onFrameGrabbed(viewID, dataVec);
        return true;
    }
    default:
    {
        const bool ret = eq::Config::handleEvent(command);
        _handleUserInputEvent(command);
        return ret;
    }
    }
}

bool Config::handleEvent(const eq::EventType type, const eq::KeyEvent& event)
{
    if (type != eq::EVENT_KEY_PRESS)
        return eq::Config::handleEvent(type, event);

    switch (event.key)
    {
    case eq::KC_ESCAPE:
    {
        setDone();
        _needsRedrawUnmaskable = true;
        return true;
    }
    case 'S':
        _frameData.statistics = !_frameData.statistics;
        if (_frameData.statistics)
            _needsRedraw = true;
        return true;
#ifndef NDEBUG
    case 'b':
        _frameData.drawViewportBorder = !_frameData.drawViewportBorder;
        if (_frameData.drawViewportBorder)
            _frameData.drawViewportBorder = true;
        return true;
#endif
    case 'l':
        _switchLayout(1);
        return true;
    case 'L':
        _switchLayout(-1);
        return true;

    default:
        return eq::Config::handleEvent(type, event);
    }
}

bool Config::handleEvent(const eq::EventType, const eq::PointerEvent& event)
{
    if (!event.context.view)
        return false;

    /* Using osgEq::View as template parameter causes a link error. */
    _lastFocusedView =
        static_cast<View*>(find<eq::View>(event.context.view.identifier));
    assert(_lastFocusedView);
    _lastFocusedContext = event.context;
    return false;
}

bool Config::handleEvent(const eq::EventType type, const eq::Event& event)
{
    switch (type)
    {
    case eq::EVENT_EXIT:
        setDone();
        _needsRedrawUnmaskable = true;
        return true;

    default:
        return eq::Config::handleEvent(type, event);
    }
}

void Config::_pushFrameEventToViews()
{
    /* Pushing a frame event to the queue in all active views and
       processing it. */
    const double timestamp = _timer.time_s();
    const eq::Layouts& layouts = getLayouts();
    for (auto layout : layouts)
    {
        if (layout->isActive())
        {
            for (auto v : layout->getViews())
            {
                View* view = static_cast<View*>(v);
                view->pushFrameEvent(timestamp);
                view->processEvents();
            }
        }
    }
}

void Config::_handleUserInputEvent(const eq::EventICommand& command)
{
    View* view = _lastFocusedView;
    if (!view && !getLayouts().empty() &&
        !getLayouts().front()->getViews().empty())
    {
        view = static_cast<View*>(getLayouts().front()->getViews().front());
    }
    if (view)
    {
        EventAdapter* e = new EventAdapter(command, _lastFocusedContext);
        view->pushEvent(e);
    }
}

void Config::handleEvents()
{
    eq::Config::handleEvents();

    /* Updating all views from active layouts */
    const eq::Layouts& layouts = getLayouts();
    for (auto layout : layouts)
        if (layout->isActive())
            for (auto view : layout->getViews())
                _needsRedraw |= static_cast<View*>(view)->processEvents();
}

bool Config::needsRedraw() const
{
    return (_needsRedrawUnmaskable || (!_maskRedraw && _needsRedraw) ||
            _remainingAAFrames > 0);
}

void Config::useLayout(const std::string& name)
{
    const eq::Canvases& canvases = getCanvases();
    if (canvases.empty())
        throw std::runtime_error("No canvas available");

    eq::Canvas* canvas = canvases.front();
    const eq::Layouts& layouts = canvas->getLayouts();
    for (uint32_t i = 0; i != layouts.size(); ++i)
    {
        if (layouts[i]->getName() == name)
        {
            const eq::Layout* old = canvas->getActiveLayout();
            if (canvas->useLayout(i))
                getClient()->getApplication()->onLayoutChanged(old, layouts[i]);
            return;
        }
    }
    throw std::runtime_error("Layout not found: " + name);
}

eq::admin::Config* Config::getAdminConfig()
{
    if (!_adminServer)
    {
        /* Creating server connection */
        _adminClient = new eq::admin::Client;
        if (!_adminClient->initLocal(0, 0))
            LBTHROW(std::runtime_error("Cannot initialize admin client"));

        _adminServer = new eq::admin::Server;
        if (!_adminClient->connectServer(_adminServer))
        {
            _adminClient->exitLocal();
            _adminClient = 0;
            _adminServer = 0;
            LBTHROW(
                std::runtime_error("Cannot open connection to "
                                   "administrate server"));
        }
    }
    /* Find first config */
    const auto& configs = _adminServer->getConfigs();
    if (configs.empty())
    {
        _adminClient->exitLocal();
        _adminServer = 0;
        LBTHROW(std::runtime_error("No configuration defined on the server"));
    }
    return configs.front();
}

void Config::_setHeadMatrix(const eq::Matrix4f& matrix)
{
    for (auto observer : getObservers())
        observer->setHeadMatrix(matrix);
}

const eq::Matrix4f& Config::_getHeadMatrix() const
{
    const eq::Observers& observers = getObservers();
    static const eq::Matrix4f identity;
    if (observers.empty())
        return identity;
    return observers[0]->getHeadMatrix();
}

void Config::_switchLayout(int32_t increment)
{
    const eq::Canvases& canvases = getCanvases();
    if (canvases.empty())
        return;

    eq::Canvas* canvas = canvases.front();
    int64_t index = canvas->getActiveLayoutIndex() + increment;
    const eq::Layouts& layouts = canvas->getLayouts();
    LBASSERT(!layouts.empty());

    const eq::Layout* oldLayout = canvas->getActiveLayout();

    index = (index % layouts.size());
    canvas->useLayout(uint32_t(index));

    const eq::Layout* layout = layouts[index];
    getClient()->getApplication()->onLayoutChanged(oldLayout, layout);

    std::ostringstream stream;
    stream << "Layout ";
    if (layout)
    {
        const std::string& name = layout->getName();
        if (name.empty())
            stream << index;
        else
            stream << name;
    }
    else
        stream << "NONE";

    stream << " active";
    std::cout << stream.str() << std::endl;
}
}
}
}
