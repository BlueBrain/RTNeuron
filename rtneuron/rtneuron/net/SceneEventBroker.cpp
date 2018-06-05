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

#include "SceneEventBroker.h"

#include "../Scene.h"

#include <brain/types.h>

#include <zeroeq/zeroeq.h>

#include <mutex>
#include <thread>

namespace bbp
{
namespace rtneuron
{
namespace net
{
#define COMMON_INITIALIZERS \
    : _done(false)                                                           \
    , _trackState(false)                                                     \
    , _parent(parent)

class SceneEventBroker::_Impl
{
public:
    _Impl(SceneEventBroker* parent) COMMON_INITIALIZERS { _init(); }
    _Impl(const std::string& session,
          SceneEventBroker* parent) COMMON_INITIALIZERS,
        _publisher(session), _subscriber(session)
    {
        _init();
    }

    _Impl(const zeroeq::URI& publisher, const zeroeq::URI& subscriber,
          SceneEventBroker* parent) COMMON_INITIALIZERS,
        _publisher(publisher), _subscriber(subscriber)
    {
        _init();
    }

    ~_Impl()
    {
        ScenePtr scene = _scene.lock();
        if (scene)
        {
            scene->cellSelected.disconnect(
                boost::bind(&_Impl::_onCellSelected, this, _1, _2, _3));
            scene->cellSetSelected.disconnect(
                boost::bind(&_Impl::_onCellSetSelected, this, _1));
        }

        _done = true;
        _receiveThread.join();
    }

    void trackScene(const ScenePtr& scene)
    {
        _scene = scene;
        scene->cellSelected.connect(
            boost::bind(&_Impl::_onCellSelected, this, _1, _2, _3));
        scene->cellSetSelected.connect(
            boost::bind(&_Impl::_onCellSetSelected, this, _1));
    }

    void setTrackState(const bool track) { _trackState = track; }
    bool getTrackState() const { return _trackState; }
    void sendToggleRequest(const brain::GIDSet& cells)
    {
        if (_trackState)
            _toggle(cells);
        else
            _sendToggleRequest(cells);
    }

    zeroeq::URI getURI() const { return _publisher.getURI(); }
private:
    SceneWeakPtr _scene;

    bool _done;
    std::thread _receiveThread;

    bool _trackState;
    GIDSet _selection;

    std::mutex _mutex;

    SceneEventBroker* _parent;

    zeroeq::Publisher _publisher;
    zeroeq::Subscriber _subscriber;

    void _init()
    {
        _subscriber.subscribe(
            lexis::data::SelectedIDs::ZEROBUF_TYPE_IDENTIFIER(),
            [&](const void* data, const size_t size) {
                _onSelectionUpdated(
                    lexis::data::SelectedIDs::create(data, size));
            });
        _subscriber.subscribe(
            lexis::data::ToggleIDRequest::ZEROBUF_TYPE_IDENTIFIER(),
            [&](const void* data, const size_t size) {
                _onToggleRequest(
                    lexis::data::ToggleIDRequest::create(data, size));
            });
        _subscriber.subscribe(
            lexis::data::CellSetBinaryOp::ZEROBUF_TYPE_IDENTIFIER(),
            [&](const void* data, const size_t size) {
                _onCellSetBinaryOp(
                    lexis::data::CellSetBinaryOp::create(data, size));
            });

        _receiveThread = std::thread(&_Impl::_poll, this);
    }

    void _poll()
    {
        while (!_done)
            _subscriber.receive(250);
    }

    void _onSelectionUpdated(lexis::data::ConstSelectedIDsPtr event)
    {
        GIDSet target;
        for (const uint32_t gid : event->getIdsVector())
            target.insert(gid);

        if (_trackState)
            _selection = target;

        _parent->cellsSelected(target);
    }

    void _onToggleRequest(lexis::data::ConstToggleIDRequestPtr event)
    {
        if (!_trackState)
            return;

        _toggle(event->getIdsVector());
    }

    void _onCellSetBinaryOp(lexis::data::ConstCellSetBinaryOpPtr event)
    {
        GIDSet first;
        for (const uint32_t gid : event->getFirstVector())
            first.insert(gid);
        GIDSet second;
        for (const uint32_t gid : event->getSecondVector())
            second.insert(gid);

        _parent->cellSetBinaryOp(first, second, event->getOperation());
    }

    void _onCellSelected(uint32_t gid, uint16_t, uint16_t)
    {
        _onCellSetSelected({gid});
    }

    void _onCellSetSelected(const brain::GIDSet& target)
    {
        sendToggleRequest(target);
    }

    void _sendToggleRequest(const brain::GIDSet& cells)
    {
        const uint32_ts ids(cells.begin(), cells.end());
        std::unique_lock<std::mutex> lock(_mutex);
        _publisher.publish(lexis::data::ToggleIDRequest(ids));
    }

    template <typename T>
    void _toggle(const T& gids)
    {
        for (const uint32_t gid : gids)
        {
            if (_selection.find(gid) != _selection.end())
                _selection.erase(gid);
            else
                _selection.insert(gid);
        }

        _parent->cellsSelected(_selection);

        const uint32_ts ids(_selection.begin(), _selection.end());
        std::unique_lock<std::mutex> lock(_mutex);
        _publisher.publish(lexis::data::SelectedIDs(ids));
    }
};

SceneEventBroker::SceneEventBroker()
    : _impl(new _Impl(this))
{
}

SceneEventBroker::SceneEventBroker(const std::string& session)
    : _impl(new _Impl(session, this))
{
}

SceneEventBroker::SceneEventBroker(const zeroeq::URI& publisher,
                                   const zeroeq::URI& subscriber)
    : _impl(new _Impl(publisher, subscriber, this))
{
}

SceneEventBroker::~SceneEventBroker()
{
    delete _impl;
}

void SceneEventBroker::setTrackState(const bool track)
{
    _impl->setTrackState(track);
}

bool SceneEventBroker::getTrackState() const
{
    return _impl->getTrackState();
}

void SceneEventBroker::sendToggleRequest(const GIDSet& cells)
{
    return _impl->sendToggleRequest(cells);
}

void SceneEventBroker::trackScene(const ScenePtr& scene)
{
    _impl->trackScene(scene);
}

zeroeq::URI SceneEventBroker::getURI() const
{
    return _impl->getURI();
}
}
}
}
