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

#include "MultiFragmentCompositor.h"
#include "ChannelCompositor.h"

#include "cuda/CUDAContext.h"

#include <co/buffer.h>
#include <co/connection.h>

#include <boost/thread/barrier.hpp>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

#include <future>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
namespace
{
class ProfilerTrace
{
public:
    struct Colors
    {
        const static uint32_t pink = 0x00ff007f;
        const static uint32_t red = 0x00ff0000;
        const static uint32_t orange = 0x00ff7f00;
        const static uint32_t yellow = 0x00ffff00;
        const static uint32_t green = 0x003fdf00;
        const static uint32_t gray = 0x007f7f7f;
        const static uint32_t lightGray = 0x00cfcfcf;
    };
#ifdef USE_NVTX
    ProfilerTrace(const char* name, uint32_t color = Colors::red)
    {
        nvtxEventAttributes_t event;
        memset(&event, 0, sizeof(nvtxEventAttributes_t));
        event.version = NVTX_VERSION;
        event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.colorType = NVTX_COLOR_ARGB;
        event.color = color;
        event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event.message.ascii = name;
        nvtxRangePushEx(&event);
    }
    ~ProfilerTrace() { nvtxRangePop(); }
#else
    ProfilerTrace(const char*, uint32_t = 0) {}
#endif
};

void _noDelete(void*)
{
}

/**
   @return the corrected position of ith data piece for a given thread index
   out of total. This result accounts for the fact the each thread processes
   the part that stays in its device the last one.
*/
size_t _correctedRegionIndex(const size_t i, const size_t index,
                             const size_t total)
{
    if (i == index)
        return total - 1;
    if (i > index)
        return i - 1;
    return i;
}

unsigned int _getDeviceID(osg::GraphicsContext* context)
{
    /* Ideally we could use the CUDA device ID, however there's not guarantee
       that the device numbers used by the display server and the CUDA device
       IDs match. The compositing logic requires the device ID to be same
       as the logical number used by Equalizer in the autoconfiguration, this
       can only be guaranteed using the port and device numbers. */
    const auto& traits = *context->getTraits();
    const auto display = traits.displayNum;
    const auto screen = traits.screenNum;
    /* We expect the display configuration to be either a separate server per
       GPU or a single server with a screen per GPU, but not hybrids. */
    if (display != 0 && screen != 0)
        LBTHROW(
            std::runtime_error("Cannot figure out device number from "
                               "display and screen numbers"));
    /* For SSH displays we assume that there's only one and it's the first
       one. */
    if (display >= 10 && traits.hostName == "localhost")
        return 0;
    return display + screen;
}
}

ChannelCompositor::ChannelCompositor(osg::GraphicsContext* context,
                                     MultiFragmentCompositor* parent)
    : ChannelCompositor(context, _getDeviceID(context), parent)
{
}

ChannelCompositor::ChannelCompositor(core::CUDAContext* context,
                                     MultiFragmentCompositor* parent)
    : ChannelCompositor(context, context->getDeviceID(), parent)
{
}

template <typename Context>
ChannelCompositor::ChannelCompositor(Context* context,
                                     const unsigned int deviceID,
                                     MultiFragmentCompositor* parent)
    : _parent(parent)
    , _functors(context)
    /* If multiprocess configurations are used, the logical device ID must be 0
       instead of the actual one. We can detect this with _devicesPerNode. */
    , _deviceID(deviceID % parent->_devicesPerNode)
    , _regionID(_parent->_nodeIndex * _parent->_devicesPerNode + _deviceID)
{
}

void ChannelCompositor::setup(Scene& scene)
{
    _functors.setup(scene);
}

void ChannelCompositor::setup(cudaArray_t counts, cudaArray_t heads,
                              void* fragments, const size_t totalFragments)
{
    _functors.setup(counts, heads, fragments, totalFragments);
}

void ChannelCompositor::extractFrameParts(const PixelViewports& inputViewports)
{
    assert(inputViewports.size() == _parent->_connections.size() ||
           (inputViewports.size() == _parent->_devicesPerNode &&
            _parent->_connections.empty() && _parent->_nodeIndex == 0));

    /* Moving the viewport associated to this device to the end of the
       viewport list, so the CUDA kernels can be overlapped with the GPU-CPU
       data transfers. */
    PixelViewports viewports;
    viewports.reserve(inputViewports.size());
    for (size_t i = 0; i < inputViewports.size(); ++i)
    {
        if (i != _regionID)
            viewports.push_back(inputViewports[i]);
    }
    _viewport = inputViewports[_regionID];
    viewports.push_back(_viewport);

    ProfilerTrace trace("compact and sort", ProfilerTrace::Colors::green);
    std::vector<int> locations;
    typedef MultiFragmentFunctors::Location Location;
    for (size_t i = 0; i != viewports.size() - 1; ++i)
        locations.push_back((int)Location::host);
    locations.push_back((int)Location::currentDevice);
    _parts = _functors.extractFrameParts(viewports, locations);
}

charPtr ChannelCompositor::compositeFrameParts(const osg::Vec4& background)
{
    if (_parts.empty())
        return charPtr();

    std::vector<std::future<void>> pending;
    for (size_t regionID = 0; regionID < _parts.size(); ++regionID)
    {
        const size_t corrected =
            _correctedRegionIndex(regionID, _regionID, _parts.size());

        /* Ensure all threads have synchronized outbound frame parts before
           sending or receiving. */
        if (regionID != _regionID)
        {
            ProfilerTrace trace("CUDA sync theirs",
                                ProfilerTrace::Colors::gray);
            _parent->_outboundFrameParts[_deviceID][regionID] =
                _parts[corrected].get();
        }
        {
            ProfilerTrace trace("thread barrier", ProfilerTrace::Colors::gray);
            _parent->_barrier->wait();
        }

        if (regionID == _regionID)
        {
            auto receives = _receiveFrameParts(_viewport);
            std::move(receives.begin(), receives.end(),
                      std::back_inserter(pending));
        }
        else
        {
            auto send = _sendFrameParts(regionID);
            if (send.valid())
                pending.emplace_back(std::move(send));
        }
    }

    /* Finally, this thread's own part is synched. */
    {
        ProfilerTrace trace("CUDA sync mine", ProfilerTrace::Colors::gray);
        auto part = _parts.back().get();
        _parent->_outboundFrameParts[_deviceID][_regionID] = part;
        _parent->_inboundFrameParts[_deviceID][_regionID] = part;
    }

    {
        ProfilerTrace trace("end operations", ProfilerTrace::Colors::pink);
        for (auto& op : pending)
            op.get();
    }

    /* Merging and blending parts. */
    charPtr output;
    {
        ProfilerTrace trace("merge and blend", ProfilerTrace::Colors::yellow);
        output = _functors.mergeAndBlend(_parent->_inboundFrameParts[_deviceID],
                                         background);
    }

    /* Deallocating all frame resources now.
       This will avoid unnecessary device-host synchronizations later on
       for bufers that need to be deallocated with cudaFree. */
    {
        /* Ensuring no thread is using inbound or outbound from any other
           thread before proceeding. */
        ProfilerTrace trace("thread barrier", ProfilerTrace::Colors::lightGray);
        _parent->_barrier->wait();
    }
    for (auto& i : _parent->_inboundFrameParts[_deviceID])
        i.reset();
    for (auto& i : _parent->_outboundFrameParts[_deviceID])
        i.reset();
    _functors.cleanup();
    _parts.clear();

    return output;
}

size_t ChannelCompositor::getFrameRegion() const
{
    return _regionID;
}

std::vector<std::future<void>> ChannelCompositor::_receiveFrameParts(
    const eq::fabric::PixelViewport& viewport)
{
    const size_t deviceCount = _parent->_devicesPerNode;
    const size_t nodeCount =
        _parent->_connections.empty()
            ?
            /* If there're no connections we still have the local node */
            1
            : _parent->_connections.size() / deviceCount;
    const size_t offsetsBufferSize =
        sizeof(uint32_t) * (viewport.w * viewport.h + 1);

    std::vector<std::future<void>> futures;

    for (size_t node = 0; node != nodeCount; ++node)
    {
        ProfilerTrace trace("start receiving region",
                            ProfilerTrace::Colors::pink);

        if (node == _parent->_nodeIndex)
        {
            /* No need to receive from network in this case. */
            const size_t regionID = node * deviceCount + _deviceID;
            for (size_t i = 0; i < deviceCount; ++i)
            {
                if (i == _deviceID)
                    /* This thread own region is synched later. */
                    continue;
                _parent->_inboundFrameParts[_deviceID][node * deviceCount + i] =
                    _parent->_outboundFrameParts[i][regionID];
            }
            continue;
        }

        size_t connectionIdx = node * deviceCount + _deviceID;
        auto& connection = _parent->_connections[connectionIdx];

        auto buffer = _parent->_readBuffers[connectionIdx];
        size_t readSize = sizeof(size_t) * deviceCount;
        buffer->reserve(readSize);
        buffer->setSize(0);
        connection->recvNB(buffer, readSize);
        if (!connection->recvSync(buffer))
            throw std::runtime_error("Error reading region");

        readSize = 0;
        size_t* sizes = reinterpret_cast<size_t*>(buffer->getData());
        for (size_t i = 0; i < _parent->_devicesPerNode; ++i)
        {
            auto& part =
                _parent->_inboundFrameParts[_deviceID][node * deviceCount + i];
            part.bufferSize = sizes[i];
            part.viewport = viewport;
            /* Checking the size of the fragment buffer because for empty
               regions the offsets are not sent either. */
            if (sizes[i])
                readSize += sizes[i] + offsetsBufferSize;
        }

        buffer->reserve(readSize);
        buffer->setSize(0);
        if (readSize)
            connection->recvNB(buffer, readSize);

        auto completion = [this, connectionIdx, node, offsetsBufferSize]() {
            const size_t devices = _parent->_devicesPerNode;
            co::BufferPtr buff;

            /* Checking if all incoming regions are empty. This is a special
               that only happens in very sparse scenes. */
            bool allEmpty = true;
            for (size_t i = 0; i < devices && allEmpty; ++i)
            {
                auto& part =
                    _parent->_inboundFrameParts[_deviceID][node * devices + i];
                allEmpty &= part.bufferSize == 0;
            }
            if (allEmpty)
                return;

            if (!_parent->_connections[connectionIdx]->recvSync(buff))
                throw std::runtime_error("Error reading region");

            size_t offset = 0;
            char* data = reinterpret_cast<char*>(buff->getData());
            for (size_t i = 0; i < devices; ++i)
            {
                auto& part =
                    _parent->_inboundFrameParts[_deviceID][node * devices + i];
                if (part.bufferSize == 0)
                    continue;
                /* The frame region info is filled with shallow pointers
                   to the read buffer. */
                part.offsets.reset(data + offset, _noDelete);
                offset += offsetsBufferSize;
                part.fragmentBuffer.reset(data + offset, _noDelete);
                offset += part.bufferSize;
            }
        };

        futures.emplace_back(std::async(std::launch::async, completion));
    }
    return futures;
}

std::future<void> ChannelCompositor::_sendFrameParts(const size_t regionID)
{
    ProfilerTrace trace("start sending region");

    const size_t deviceCount = _parent->_devicesPerNode;
    const size_t owner = regionID / deviceCount;

    /* IF the destination is another thread in this node there's nothing
       to do. */
    if (owner == _parent->_nodeIndex)
        return std::future<void>();

    /* Only the device thread with the same device ID as the destination sends
       all the information together. The caller must have ensured that the
       oubound frame parts are ready for seding. */
    if (_deviceID != regionID % deviceCount)
        return std::future<void>();

    return std::async(std::launch::async, [this, regionID, deviceCount]() {
        auto& connection = _parent->_connections[regionID];

        std::vector<size_t> sizes;
        sizes.reserve(deviceCount);
        for (size_t i = 0; i < deviceCount; ++i)
            sizes.push_back(
                _parent->_outboundFrameParts[i][regionID].bufferSize);

        connection->lockSend();
        connection->send(sizes.data(), sizeof(size_t) * sizes.size(), true);

        const auto& viewport =
            _parent->_outboundFrameParts[_deviceID][regionID].viewport;
        const size_t offsetsBufferSize =
            sizeof(uint32_t) * (viewport.w * viewport.h + 1);
        for (size_t i = 0; i < deviceCount; ++i)
        {
            const auto& part = _parent->_outboundFrameParts[i][regionID];
            if (!part.bufferSize)
                /* Skipping empty parts. */
                continue;

            connection->send(part.offsets.get(), offsetsBufferSize, true);
            connection->send(part.fragmentBuffer.get(), part.bufferSize, true);
        }
        connection->unlockSend();
    });
}

MultiFragmentCompositor::MultiFragmentCompositor(
    const core::ConnectionDescriptions& descriptions, const size_t nodeIndex,
    const size_t devicesPerNode)
    : MultiFragmentCompositor(core::connectPeers(descriptions, nodeIndex,
                                                 devicesPerNode),
                              nodeIndex, devicesPerNode)
{
}

MultiFragmentCompositor::MultiFragmentCompositor(
    const core::ConnectionDescriptions& descriptions, co::Connection& listener,
    const size_t nodeIndex, const size_t devicesPerNode)
    : MultiFragmentCompositor(core::connectPeers(descriptions, listener,
                                                 nodeIndex, devicesPerNode),
                              nodeIndex, devicesPerNode)
{
}

MultiFragmentCompositor::MultiFragmentCompositor(const size_t devicesPerNode)
    : MultiFragmentCompositor(co::Connections(), 0, devicesPerNode)
{
    for (auto& i : _outboundFrameParts)
        i.resize(devicesPerNode);
    for (auto& i : _inboundFrameParts)
        i.resize(devicesPerNode);
}

MultiFragmentCompositor::MultiFragmentCompositor(co::Connections&& connections,
                                                 const size_t nodeIndex,
                                                 const size_t devicesPerNode)
    : _connections(std::move(connections))
    , _devicesPerNode(devicesPerNode)
    , _nodeIndex(nodeIndex)
    , _outboundFrameParts(devicesPerNode)
    , _inboundFrameParts(devicesPerNode)
    , _barrier(new boost::barrier(devicesPerNode))
{
    for (auto& i : _outboundFrameParts)
        i.resize(_connections.size());
    for (auto& i : _inboundFrameParts)
        i.resize(_connections.size());

    /* Creating the reading buffers. */
    _readBuffers.reserve(_connections.size());
    for (size_t i = 0; i < _connections.size(); ++i)
        _readBuffers.push_back(co::BufferPtr(new co::Buffer()));
}

MultiFragmentCompositor::~MultiFragmentCompositor()
{
}

ChannelCompositorPtr MultiFragmentCompositor::createChannelCompositor(
    osg::GraphicsContext* context)
{
    return ChannelCompositorPtr(new ChannelCompositor(context, this));
}

ChannelCompositorPtr MultiFragmentCompositor::createChannelCompositor(
    core::CUDAContext* context)
{
    return ChannelCompositorPtr(new ChannelCompositor(context, this));
}

bool MultiFragmentCompositor::isSingleChannel() const
{
    return _connections.size() == 1 && _devicesPerNode == 1;
}
}
}
}
