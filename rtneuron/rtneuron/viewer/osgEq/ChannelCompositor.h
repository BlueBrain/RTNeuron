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

#ifndef RTNEURON_OSGEQ_CHANNELCOMPOSITOR_H
#define RTNEURON_OSGEQ_CHANNELCOMPOSITOR_H

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
class MultiFragmentCompositor;

class ChannelCompositor
{
public:
    void setup(Scene& scene);

    /* @internal Used in performance tests */
    void setup(cudaArray_t counts, cudaArray_t heads, void* fragments,
               size_t totalFragments);

    /** Start the processing of the frame parts to exchange with other peers.

        @param viewports The viewport assigned to each participating channel.
               This list must be the same in all channels and must have the same
               length as the descriptions.size() * devicesPerNode from the
               parameters used to create the parent MultiFragmentCompositor
    */
    void extractFrameParts(const PixelViewports& viewports);

    /** Exchange fragment parts between all channels participating in the
        compositing and return the pixel data of the fragment region this
        channel is responsible of compositing.

        This function must be called from all channel, otherwise it may lead
        to a deadlock. extractFrameParts has to be called before calling this
        function

        @param background The RGBA background color to use during compositing.

        @return The RGBA pixel data for the frame region associated to this
                channel.
    */
    charPtr compositeFrameParts(const osg::Vec4& background);

    /** Return the frame region index this compositor will merge and blend. */
    size_t getFrameRegion() const;

private:
    MultiFragmentCompositor* const _parent;
    MultiFragmentFunctors _functors;
    const unsigned int _deviceID;
    const unsigned int _regionID;

    eq::fabric::PixelViewport _viewport;
    Futures<MultiFragmentFunctors::FramePart> _parts;

    friend class MultiFragmentCompositor;

    ChannelCompositor(osg::GraphicsContext* context,
                      MultiFragmentCompositor* parent);

    ChannelCompositor(core::CUDAContext* context,
                      MultiFragmentCompositor* parent);

    /* Delegated constructor. */
    template <typename Context>
    ChannelCompositor(Context* context, unsigned int deviceID,
                      MultiFragmentCompositor* parent);

    std::vector<std::future<void>> _receiveFrameParts(
        const eq::fabric::PixelViewport& viewport);

    /* Send frame data for the given regionId collected in this node to the
       destination node using the appropriate connection of the parent
       compositor. */
    std::future<void> _sendFrameParts(const size_t regionId);
};
}
}
}
#endif
