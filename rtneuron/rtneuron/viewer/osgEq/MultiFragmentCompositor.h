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

#ifndef RTNEURON_OSGEQ_MULTIFRAGMENTCOMPOSITOR_H
#define RTNEURON_OSGEQ_MULTIFRAGMENTCOMPOSITOR_H

#include "MultiFragmentFunctors.h"

#include "coreTypes.h"
#include "util/net.h"

#include <future>
#include <memory>
#include <vector>

namespace osg
{
class GraphicsContext;
}

namespace boost
{
class barrier;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Scene;
class ChannelCompositor;
typedef std::unique_ptr<ChannelCompositor> ChannelCompositorPtr;

class MultiFragmentCompositor
{
public:
    friend class ChannelCompositor;

    /** Initialize the compositing network for all fragments sort-last.

        This class is supposed to be created at the node level. From the input
        connection descriptions (which must include the local node), each node
        opens devicesPerNode connections to every other node.
    */
    MultiFragmentCompositor(const core::ConnectionDescriptions& descriptions,
                            size_t nodeIndex, size_t devicesPerNode);

    /** The same as above with providing an already listening connection. */
    MultiFragmentCompositor(const core::ConnectionDescriptions& descriptions,
                            co::Connection& listener, size_t nodeIndex,
                            size_t devicesPerNode);

    /** Compositor to use in the single node case. */
    MultiFragmentCompositor(size_t devices);

    ~MultiFragmentCompositor();

    ChannelCompositorPtr createChannelCompositor(osg::GraphicsContext* context);

    ChannelCompositorPtr createChannelCompositor(core::CUDAContext* context);

    /** Return true is only one channel is doing rendering.
        This is meant for debugging purposes, to ensure that something is
        displayed when a single channel configuration includes a DB range in
        the compound.
    */
    bool isSingleChannel() const;

private:
    MultiFragmentCompositor(co::Connections&& _connections, size_t nodeIndex,
                            size_t devicesPerNode);

    /* The available connections will be used in the following way by the
       channel compositors.
       - For sending: Being d the device index of the target channel, the thred
         of the channel with the same device index collects the information of
         all local devices and sends it to the target using the connection
         devicesPerNode * remote_node + d (which equals the frame region to be
         composed by the target device).
       - For receiving: Each channel thread for device d receives in the
         connection devicesPerNode * remote_node + d the information of all
         devices of the remote node. */
    co::Connections _connections;

    size_t _devicesPerNode;
    size_t _nodeIndex;

    /* One frame part per channel participating in the compositing, that is
       _connections.size() * devicesPerNode. */
    typedef std::vector<MultiFragmentFunctors::FramePart> Parts;
    /* One list of regions per local device */
    std::vector<Parts> _outboundFrameParts;
    std::vector<Parts> _inboundFrameParts;

    typedef std::vector<co::BufferPtr> Buffers;
    Buffers _readBuffers;

    std::unique_ptr<boost::barrier> _barrier;
};
}
}
}
#endif
