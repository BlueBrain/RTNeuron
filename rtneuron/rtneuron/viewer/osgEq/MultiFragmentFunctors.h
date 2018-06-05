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

#ifndef RTNEURON_OSGEQ_MULTIFRAGMENTFUNCTORS_H
#define RTNEURON_OSGEQ_MULTIFRAGMENTFUNCTORS_H

#include <osg/Observer>
#include <osg/Vec4>
#include <osg/Version>
#include <osg/ref_ptr>

#include <eq/fabric/pixelViewport.h>

#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>

#include <future>
#include <memory>
#include <vector>

namespace osg
{
class GraphicsContext;
class State;
class Texture;
class TextureRectangle;
}

namespace bbp
{
namespace osgTransparency
{
class BaseRenderBin;
class TextureBuffer;
class FragmentData;
}

namespace rtneuron
{
namespace core
{
class CUDAContext;
}

namespace osgEq
{
class Scene;

typedef std::vector<eq::fabric::PixelViewport> PixelViewports;
typedef std::shared_ptr<char> charPtr;
typedef std::shared_ptr<void> voidPtr;

template <typename T>
using Futures = std::vector<std::future<T>>;

class MultiFragmentFunctors : public osg::Observer
{
public:
    /*--- Public declarations ---*/

    /** A frame part if the partial information of a frame region defined by
        a viewport.

       @sa extractFrameRegions for memory management considerations.
    */
    struct FramePart
    {
        FramePart()
            : location(Location::host)
            , bufferSize(0)
        {
        }

        eq::fabric::PixelViewport viewport;
        enum class Location
        {
            host = 0,
            device
        };
        /* The location of the buffers. For device locations, the device is that
           one that was requested in extractFrameParts. */
        Location location;
        /** This buffer contains the offsets which result from the prefix sum
            of the per pixel fragment counts. Its size is width * height + 1. */
        voidPtr offsets;
        /** Sorted and compacted per pixel fragment counts. */
        voidPtr fragmentBuffer;
        size_t bufferSize; //!< Fragment buffer size in bytes

        void reset()
        {
            offsets.reset();
            fragmentBuffer.reset();
            bufferSize = 0;
            location = Location::host;
        };
    };

    /** @internal */
    struct Texture
    {
        enum class Type
        {
            image,
            buffer
        };
        Type type;
        cudaGraphicsResource_t resource;
        osg::Texture* texture;
        union {
            cudaArray_t array;
            void* buffer;
        };
        const char* name;
    };

    /*--- Public constructors/destructor ---*/

    MultiFragmentFunctors(osg::GraphicsContext* state);

    /** @internal Used in performance tests */
    MultiFragmentFunctors(core::CUDAContext* context);

    ~MultiFragmentFunctors();

    /*--- Public member functions ---*/

    void setup(Scene& scene);

    /* @internal Used in performance tests */
    void setup(cudaArray_t counts, cudaArray_t heads, void* fragments,
               size_t totalFragments);

    /**
       Extract and issue the device->host copy of fragment data for the given
       viewports.

       The host memory pointers returned in the FrameParts point to a single
       buffer controlled internally by this class. Calling this function may
       deallocate and/or overwrite the buffer returned in a previous call.
       This single buffer is used to avoid expensive pinned memory
       de/allocations in the host.

       @param viewports
       @param locations The desired locations of the output data.
              The possible values are:
              * -2: Host memory
              * -1: The current device
              * a number >= 0: a explicit device.
              If this vector is empty all data is downloaded to the host.
              @note At the moment the implementation doesn't not support copies
                    between devices and will copy to the host instead.
    */
    Futures<FramePart> extractFrameParts(
        const PixelViewports& viewports,
        const std::vector<int>& locations = {});

    class Location
    {
    public:
        static const int host = -2;
        static const int currentDevice = -1;
    };

    /** Merges and blends the given parts of a frame region into a single RGBA
        buffer.

        @param parts The list of frame parts. All viewports must be the same,
               otherwise the behaviour is undefined.
        @param background Background color to blend with.

        @return An in-memory buffer with RGBA the merged and blended pixels.
    */
    charPtr mergeAndBlend(const std::vector<FramePart>& parts,
                          const osg::Vec4& background);

    /** Perform cleanup of GPU buffers.

       Buffers allocated by extractFrameParts are not deallocated immediately
       once they are not needed any more. This is done because cudaFree is
       device synchronous, and deallocating the buffers as soon as possible
       prevents from overlapping device-host transfers with network
       operations.

       This operation does not cleanup the pinned memory buffers allocated
       in the host side.
    */
    void cleanup();

private:
    /*--- Private declarations ---*/
    class Helpers;
    typedef std::vector<cudaEvent_t> CUDAEvents;

    /*--- Private member variables ---*/

    osg::ref_ptr<core::CUDAContext> _cudaContext;
    cudaStream_t _dToHcopyStream;
    unsigned int _contextID;

    CUDAEvents _availableEvents;

    Texture _counts;
    Texture _heads;
    Texture _fragments;

    struct PinnedMemBuffer
    {
        PinnedMemBuffer()
            : size(0)
        {
        }
        charPtr ptr;
        size_t size;
    };
    PinnedMemBuffer _fragmentBuffer;
    PinnedMemBuffer _offsetBuffer;
    size_t _totalNumFragments;

    std::vector<voidPtr> _disposedBuffers;

    osgTransparency::BaseRenderBin* _renderBin;

    /*--- Private member functions ---*/

    MultiFragmentFunctors(const MultiFragmentFunctors& state) = delete;

    MultiFragmentFunctors(core::CUDAContext* context, unsigned int contextID);

    bool _postDraw(const osgTransparency::FragmentData& data);

    bool _registerTextures(osg::State& state, osg::TextureRectangle* counts,
                           osg::TextureRectangle* heads,
                           osgTransparency::TextureBuffer* fragments);
    bool _mapTextures(const cudaStream_t stream);
    void _unmapTextures(const cudaStream_t stream);

    void _allocHostBuffers(const PixelViewports& viewports,
                           const std::vector<int>& locations);

    /**
       Issue the copies of portions of the counts texture to host memory.

       @param viewports List of viewports with the regions to copy.
       @param counts Count texture coming from osgTransparency.
       @return A list of futures that return a charPtr buffer per
       viewport. Each output buffer will have as many uint32_t values as
       pixels in its region. Each futures.get() runs a synchronization on a CUDA
       event recorded in the default stream.
    */
    Futures<charPtr> _copyCountsToHostAsync(const PixelViewports& viewports,
                                            const std::vector<int>& locations,
                                            const cudaArray_t counts);

    void objectDeleted(void* object) final;
};
}
}
}
#endif
