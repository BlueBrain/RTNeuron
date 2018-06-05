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

#include "MultiFragmentFunctors.h"
#include "Scene.h"

#include "cuda/CUDAContext.h"
#include "cuda/fragments.h"

#include <osg/State>
#include <osg/TextureRectangle>
#include <osgTransparency/FragmentListOITBin.h>
#include <osgTransparency/TextureBuffer.h>

#include <boost/bind.hpp>

#include <utility>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
namespace
{
void _noDelete(void*)
{
}

/*
  Helper types and functions
*/
typedef std::shared_ptr<uint32_t> uint32_tPtr;
typedef std::vector<uint32_tPtr> uint32_tPtrs;
typedef std::vector<charPtr> charPtrs;

template <typename T>
using Promises = std::vector<std::promise<T>>;

/**
   Unregister the given texture from CUDA.

   @param data All the GL, CUDA and metada for the texture.
   @param functors If not null, this object is removed as an observer
          of the OSG texture object. This parameter must be null when
          invoked from the objectDeleted callback as it will cause a
          trivial deadlock otherwise.
*/
void _unregisterTexture(MultiFragmentFunctors::Texture& data,
                        MultiFragmentFunctors* functors = 0)
{
    if (!data.resource)
        return;

    if (cudaGraphicsUnregisterResource(data.resource) != cudaSuccess)
        LBERROR << "CUDA error unregistering " << data.name
                << "buffer: " << cudaGetErrorString(cudaGetLastError())
                << std::endl;
    data.resource = 0;

    if (data.texture && functors)
        data.texture->removeObserver(functors);
    data.texture = 0;
}

bool _registerTextureImpl(osg::TextureRectangle& texture,
                          MultiFragmentFunctors::Texture& data,
                          const unsigned int contextID)
{
    if (data.type != MultiFragmentFunctors::Texture::Type::image)
    {
        throw std::runtime_error(
            "Invalid type of texture for image registration in CUDA");
    }

    const osg::Texture::TextureObject* to = texture.getTextureObject(contextID);
    const auto flags = cudaGraphicsRegisterFlagsReadOnly;
    return cudaGraphicsGLRegisterImage(&data.resource, to->id(), to->target(),
                                       flags) == cudaSuccess;
}

bool _registerTextureImpl(osgTransparency::TextureBuffer& texture,
                          MultiFragmentFunctors::Texture& data,
                          const unsigned int contextID)
{
    if (data.type != MultiFragmentFunctors::Texture::Type::buffer)
    {
        throw std::runtime_error(
            "Invalid type of texture for image registration in CUDA");
    }

    const osg::GLBufferObject* buffer = texture.getGLBufferObject(contextID);
    assert(buffer);
    const auto flags = cudaGraphicsRegisterFlagsReadOnly;
    return cudaGraphicsGLRegisterBuffer(&data.resource, buffer->getGLObjectID(),
                                        flags) == cudaSuccess;
}

template <typename OsgTexture>
bool _registerTexture(OsgTexture* texture, MultiFragmentFunctors::Texture& data,
                      MultiFragmentFunctors* functors,
                      const unsigned int contextID)
{
    if (data.texture == texture)
        /* Already registered */
        return true;

    if (data.resource)
        _unregisterTexture(data, functors);

    if (!_registerTextureImpl(*texture, data, contextID))
    {
        LBERROR << "CUDA error registering texture " << data.name << ": "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return false;
    }

    data.texture = texture;
    texture->addObserver(functors);
    return true;
}

bool _mapTexture(MultiFragmentFunctors::Texture& texture)
{
    if (!texture.resource)
        return true;

    cudaError_t error;
    if (texture.type == MultiFragmentFunctors::Texture::Type::image)
    {
        error = cudaGraphicsSubResourceGetMappedArray(&texture.array,
                                                      texture.resource, 0, 0);
    }
    else
    {
        size_t size; /* In principle we don't need to store the size */
        error = cudaGraphicsResourceGetMappedPointer(&texture.buffer, &size,
                                                     texture.resource);
    }
    if (error != cudaSuccess)
    {
        LBERROR << "CUDA mapping texture " << texture.name << ": "
                << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}
}

class MultiFragmentFunctors::Helpers
{
public:
    class CUDAEvent
    {
    public:
        CUDAEvent(cudaEvent_t event, MultiFragmentFunctors* parent)
            : _parent(parent)
            , _event(event)
        {
        }
        CUDAEvent(CUDAEvent&& event)
            : _parent(event._parent)
            , _event(event._event)
        {
            event._parent = 0;
        }
        CUDAEvent& operator=(CUDAEvent&& event)
        {
            _parent = event._parent;
            _event = event._event;
            event._parent = 0;
            return *this;
        }

        ~CUDAEvent()
        {
            if (_parent)
                _parent->_availableEvents.push_back(_event);
        }

        operator cudaEvent_t() const { return _event; }
        CUDAEvent(const CUDAEvent&) = delete;
        CUDAEvent& operator=(const CUDAEvent&) = delete;

    private:
        MultiFragmentFunctors* _parent;
        cudaEvent_t _event;
    };

    struct ExtendedFramePart : public MultiFragmentFunctors::FramePart
    {
        ExtendedFramePart(MultiFragmentFunctors* functors)
            : numFragments(0)
            , kernelSyncEvent(getEvent(functors))
            , offsetCopyEvent(getEvent(functors))
            , sizeCopyEvent(getEvent(functors))
            , fragmentCopyEvent(getEvent(functors))
        {
        }
        ExtendedFramePart(ExtendedFramePart&& part)
            : FramePart(std::move(part))
            , offsetsDev(std::move(part.offsetsDev))
            , fragmentsDev(std::move(part.fragmentsDev))
            , numFragments(part.numFragments)
            , kernelSyncEvent(std::move(part.kernelSyncEvent))
            , offsetCopyEvent(std::move(part.offsetCopyEvent))
            , sizeCopyEvent(std::move(part.sizeCopyEvent))
            , fragmentCopyEvent(std::move(part.fragmentCopyEvent))
        {
        }
        uint32_tPtr offsetsDev;
        voidPtr fragmentsDev;
        uint32_t numFragments;
        CUDAEvent kernelSyncEvent;
        CUDAEvent offsetCopyEvent;
        CUDAEvent sizeCopyEvent;
        CUDAEvent fragmentCopyEvent;
    };

    /**
        Takes an event which is not being used from the list of available ones
        or creates a new one.
        @return An event handler.
    */
    static CUDAEvent getEvent(MultiFragmentFunctors* functors)
    {
        if (!functors->_availableEvents.empty())
        {
            cudaEvent_t event = functors->_availableEvents.back();
            functors->_availableEvents.pop_back();
            return CUDAEvent(event, functors);
        }
        cudaEvent_t event;
        const auto error =
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (error != cudaSuccess)
        {
            LBERROR << "Error creating CUDA event: "
                    << cudaGetErrorString(error) << std::endl;
            abort();
        }
        return CUDAEvent(event, functors);
    }
};

/*
  Constructors/destructor
*/

MultiFragmentFunctors::MultiFragmentFunctors(osg::GraphicsContext* context)
    : MultiFragmentFunctors(core::CUDAContext::getOrCreateContext(context),
                            context->getState()->getContextID())
{
}

MultiFragmentFunctors::MultiFragmentFunctors(core::CUDAContext* context)
    : MultiFragmentFunctors(context, 0)
{
}

MultiFragmentFunctors::MultiFragmentFunctors(core::CUDAContext* context,
                                             const unsigned int contextID)
    : _cudaContext(context)
    , _dToHcopyStream(context->createStream())
    , _contextID(contextID)
    , _counts{Texture::Type::image, nullptr, nullptr, nullptr, "counts"}
    , _heads{Texture::Type::image, nullptr, nullptr, nullptr, "heads"}
    , _fragments{Texture::Type::buffer, nullptr, nullptr, nullptr, "fragments"}
    , _totalNumFragments(0)
{
}

MultiFragmentFunctors::~MultiFragmentFunctors()
{
    core::ScopedCUDAContext context(_cudaContext);

    _unregisterTexture(_counts, this);
    _unregisterTexture(_heads, this);
    _unregisterTexture(_fragments, this);

    /* Destroying synchronization objects */
    context->destroyStream(_dToHcopyStream);
    for (auto event : _availableEvents)
    {
        const auto error = cudaEventDestroy(event);
        if (error != cudaSuccess)
            LBERROR << "Error destroying CUDA event: "
                    << cudaGetErrorString(error) << std::endl;
    }
}

/*
  Member functions
*/

void MultiFragmentFunctors::setup(Scene& scene)
{
    using namespace osgTransparency;
    _renderBin = scene.getAlphaBlendedRenderBin();
    auto& parameters = static_cast<FragmentListOITBin::Parameters&>(
        _renderBin->getParameters());
    auto callback = boost::bind(&MultiFragmentFunctors::_postDraw, this, _1);
    parameters.setCaptureCallback(_contextID, callback);
}

void MultiFragmentFunctors::setup(const cudaArray_t counts,
                                  const cudaArray_t heads, void* fragments,
                                  const size_t totalFragments)
{
    _counts = Texture{Texture::Type::image, nullptr, nullptr, counts, "counts"};
    _heads = Texture{Texture::Type::image, nullptr, nullptr, heads, "heads"};
    _fragments = Texture{Texture::Type::buffer,
                         nullptr,
                         nullptr,
                         {.buffer = fragments},
                         "fragments"};
    _totalNumFragments = totalFragments;
}

Futures<MultiFragmentFunctors::FramePart>
    MultiFragmentFunctors::extractFrameParts(const PixelViewports& viewports,
                                             const std::vector<int>& locations)
{
    if (!_counts.texture && !_counts.array)
    {
        /* No frame rendered yet and not in testing mode. */
        return Futures<FramePart>();
    }

    using namespace core;
    ScopedCUDAContext context(_cudaContext);
    auto freeFunctor = context->cudaFreeFunctor(cudaFree);

    if (!_mapTextures(0))
        abort();

    typedef Helpers::ExtendedFramePart ExtFramePart;

    std::vector<ExtFramePart> parts;

    _allocHostBuffers(viewports, locations);

    size_t totalPixels = 0;
    for (const auto& viewport : viewports)
        totalPixels += viewport.w * viewport.h;

    /* Computing the fragment offsets from the counts and issue the copy
       operations to get in the host the number of fragments inside each
       viewport. */
    void* tmp = 0;
    size_t tmpSize = 0;
    /* Allocating the single buffer for all pixels offsets. */
    uint32_t* offsets = 0;
    {
        const size_t size = (totalPixels + viewports.size()) * sizeof(uint32_t);
        const auto error = cudaMalloc(&offsets, size);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
    }
    /* This pointer will be shared by all parts through shared pointer
       aliasing. */
    uint32_tPtr offsetBuffer(offsets, freeFunctor);

    for (size_t i = 0; i != viewports.size(); ++i)
    {
        parts.emplace_back(ExtFramePart(this));
        auto& part = parts.back();

        const auto& vp = viewports[i];
        part.viewport = vp;

        /* Deciding whether the output goes to the host memory or stays in this
           device. */
        if (!locations.empty() && locations[i] == Location::currentDevice)
            part.location = FramePart::Location::device;
        else
            part.location = FramePart::Location::host;

        /* The offset buffer pointer uses shared_ptr pointer aliasing to the
           single buffer. */
        part.offsetsDev = uint32_tPtr(offsetBuffer, offsets);

        /* Computing the fragments offsets from the counts. */
        void* currentTmp = tmp;
        const auto error =
            core::cuda::fragmentCountsToOffsets(vp.w, vp.h, vp.x, vp.y,
                                                _counts.array, offsets,
                                                currentTmp, tmpSize);

        if (error != cudaSuccess)
        {
            LBERROR << "CUDA error computing fragment offsets: "
                    << cudaGetErrorString(error) << std::endl;
            abort();
        }
        if (currentTmp != tmp)
        {
            tmp = currentTmp;
            _disposedBuffers.push_back(voidPtr(tmp, freeFunctor));
        }

        /* Issuing the copy of only the fragment count on this viewport. The
           offset buffer will be copied back later. */
        const size_t pixels = vp.w * vp.h;
        cudaEventRecord(part.kernelSyncEvent);
        cudaStreamWaitEvent(_dToHcopyStream, part.kernelSyncEvent, 0);
        cudaMemcpyAsync(&part.numFragments, offsets + pixels, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, _dToHcopyStream);
        cudaEventRecord(part.sizeCopyEvent, _dToHcopyStream);

        /* Making offsets point to the position to be used for the next part. */
        offsets += pixels + 1;
    }

    /* Now, issuing the device->host copy operations for the offset buffers.
       These copies will be overlapping the execution of the compact/sort
       kernels. */
    size_t totalSize = 0;
    for (size_t i = 0; i != viewports.size(); ++i)
    {
        auto& part = parts[i];
        if (part.location == FramePart::Location::host)
        {
            const auto& vp = part.viewport;
            const size_t pixels = vp.w * vp.h;
            const size_t size = (pixels + 1) * sizeof(uint32_t);
            char* hostBuffer = _offsetBuffer.ptr.get() + totalSize;
            part.offsets.reset(hostBuffer, _noDelete);
            cudaMemcpyAsync(hostBuffer, part.offsetsDev.get(), size,
                            cudaMemcpyDeviceToHost, _dToHcopyStream);
            cudaEventRecord(part.sizeCopyEvent, _dToHcopyStream);
            totalSize += size;
        }
        else
        {
            part.offsets = part.offsetsDev;
        }
    }

    totalSize = 0;
    for (size_t i = 0; i != viewports.size(); ++i)
    {
        auto& part = parts[i];
        const auto& vp = part.viewport;
        offsets = part.offsetsDev.get();

        cudaError_t error;
#ifndef NDEBUG
        error = cudaEventSynchronize(part.kernelSyncEvent);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error in counts to offsets kernel: ") +
                cudaGetErrorString(error));
        }
#endif

        /* Allocating the device fragment buffer. */
        error = cudaEventSynchronize(part.sizeCopyEvent);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error finishing output size copy: ") +
                cudaGetErrorString(error));
        }
        size_t bufferSize = part.numFragments * 2 * sizeof(uint32_t);
        part.bufferSize = bufferSize;

        void* fragments;
        /* Apparently cudaMalloc is now asynchronous, so this allocation
           shouldn't make the CPU wait for the previous viewport computation
           to be done (http://stackoverflow.com/questions/36001364) */
        error = cudaMalloc(&fragments, bufferSize);
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RTNeuron: error allocating device memory: ") +
                cudaGetErrorString(error));
        }
        part.fragmentsDev = voidPtr(fragments, freeFunctor);

        /* Issuing the compact/sort kernel. */
        error = core::cuda::compactAndSortFragmentLists(
            vp.w, vp.h, vp.x, vp.y, _heads.array, (uint32_t*)_fragments.buffer,
            offsets, fragments, bufferSize);
        if (error != cudaSuccess)
        {
            LBERROR << "CUDA error compacting and sorting fragment lists: "
                    << cudaGetErrorString(error) << std::endl;
            abort();
        }
        cudaEventRecord(part.kernelSyncEvent);

        if (part.location == FramePart::Location::host)
        {
            /* Queueing the aync operation to copy the buffer to the host. */
            char* hostBuffer = _fragmentBuffer.ptr.get() + totalSize;
            part.fragmentBuffer.reset(hostBuffer, _noDelete);
            totalSize += bufferSize;

            cudaStreamWaitEvent(_dToHcopyStream, part.kernelSyncEvent, 0);
            cudaMemcpyAsync(hostBuffer, fragments, bufferSize,
                            cudaMemcpyDeviceToHost, _dToHcopyStream);
            cudaEventRecord(part.fragmentCopyEvent, _dToHcopyStream);
        }
        else
        {
            part.fragmentBuffer = part.fragmentsDev;
        }
    }

    /* And finally we queue all the futures with the final synchronization. */
    Futures<FramePart> futures;
    for (size_t i = 0; i != viewports.size(); ++i)
    {
        futures.emplace_back(std::async(
            std::launch::deferred,
            [this](ExtFramePart&& part, const bool unmap) {
                if (part.location == FramePart::Location::host)
                {
                    auto error = cudaEventSynchronize(part.offsetCopyEvent);
                    if (error != cudaSuccess)
                        LBERROR << "Error finishing offset copy: "
                                << cudaGetErrorString(error) << std::endl;
                    /* Here we can proceed to the compression of the offsets
                       while the other copy operation is being finished. */
                    _disposedBuffers.push_back(part.offsetsDev);

                    error = cudaEventSynchronize(part.fragmentCopyEvent);
                    if (error != cudaSuccess)
                        LBERROR << "Error finishing fragment copy: "
                                << cudaGetErrorString(error) << std::endl;
                    _disposedBuffers.push_back(part.fragmentsDev);
                }
                else
                    cudaEventSynchronize(part.kernelSyncEvent);

                if (unmap)
                    _unmapTextures(0);
                return FramePart(std::move(part));
            },
            std::move(parts[i]), i == viewports.size() - 1));
    }
    return futures;
}

void MultiFragmentFunctors::cleanup()
{
    _disposedBuffers.clear();
}

charPtr MultiFragmentFunctors::mergeAndBlend(
    const std::vector<FramePart>& parts, const osg::Vec4& background)
{
    std::vector<const float*> depths;
    std::vector<const uint32_t*> colors;
    std::vector<const uint32_t*> offsets;
    std::vector<const void*> fragments;
    std::vector<const void*> toFree;
    depths.reserve(parts.size());
    colors.reserve(parts.size());
    offsets.reserve(parts.size());
    fragments.reserve(parts.size());
    toFree.reserve(parts.size() * 2);

    eq::fabric::PixelViewport vp = parts[0].viewport;
    size_t pixels = vp.w * vp.h;

    for (const auto& part : parts)
    {
        assert(vp == part.viewport);
        uint32_t* partOffsets;
        void* fragmentBuffer;
        size_t size = part.bufferSize;

        if (size == 0)
            continue;

        if (part.location == FramePart::Location::host)
        {
            auto error =
                cudaMalloc(&partOffsets, (pixels + 1) * sizeof(uint32_t));
            if (error != cudaSuccess)
            {
                LBERROR << "Error allocating device memory: "
                        << cudaGetErrorString(error) << std::endl;
                abort();
            }
            cudaMemcpy(partOffsets, part.offsets.get(),
                       (pixels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
            toFree.push_back(partOffsets);

            error = cudaMalloc(&fragmentBuffer, size);
            if (error != cudaSuccess)
            {
                LBERROR << "Error allocating device memory: "
                        << cudaGetErrorString(error) << std::endl;
                abort();
            }
            fragments.push_back(fragmentBuffer);
            cudaMemcpy(fragmentBuffer, part.fragmentBuffer.get(), size,
                       cudaMemcpyHostToDevice);
            toFree.push_back(fragmentBuffer);
        }
        else
        {
            partOffsets = (uint32_t*)part.offsets.get();
            fragmentBuffer = part.fragmentBuffer.get();
        }

        offsets.push_back(partOffsets);
        depths.push_back((float*)fragmentBuffer);
        colors.push_back((uint32_t*)fragmentBuffer +
                         size / (sizeof(float) + sizeof(uint32_t)));
    }
    const uint32_t rgba = (unsigned char)(background[0] * 255) +
                          ((unsigned char)(background[1] * 255) << 8) +
                          ((unsigned char)(background[2] * 255) << 16) +
                          ((unsigned char)(background[3] * 255) << 24);
    void* output = 0;
    auto error = core::cuda::mergeAndBlendFragments(depths, colors, offsets,
                                                    rgba, pixels, output);
    if (error != cudaSuccess)
    {
        LBERROR << "Error during fragment merge/blend: "
                << cudaGetErrorString(error) << std::endl;
        abort();
    }
    const charPtr buffer(new char[4 * vp.w * vp.h],
                         std::default_delete<char[]>());
    error = cudaMemcpy(buffer.get(), output, 4 * vp.w * vp.h,
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        LBERROR << "Error after merge/blend, detected during copy: "
                << cudaGetErrorString(error) << std::endl;
        abort();
    }

    for (auto p : toFree)
        cudaFree((void*)p);
    cudaFree(output);

    return buffer;
}

bool MultiFragmentFunctors::_postDraw(const osgTransparency::FragmentData& data)
{
    osg::State& state = data.getState();
    osg::TextureRectangle* counts = data.getCounts();
    osg::TextureRectangle* heads = data.getHeads();
    osgTransparency::TextureBuffer* fragments = data.getFragments();

    if (!_registerTextures(state, counts, heads, fragments))
    {
        /* No multifragment compositing will proceed, let fragment sorting
           from osgTransparency proceed normally. */
        return true;
    }

    _totalNumFragments = data.getNumFragments();

    /* Replacing the callback with an empty one immediately. The callback
       will be reset the next frame. */
    using namespace osgTransparency;
    auto& parameters = static_cast<FragmentListOITBin::Parameters&>(
        _renderBin->getParameters());
    parameters.setCaptureCallback(state.getContextID(), CaptureCallback());

    return false;
}

bool MultiFragmentFunctors::_registerTextures(
    osg::State& state, osg::TextureRectangle* counts,
    osg::TextureRectangle* heads, osgTransparency::TextureBuffer* fragments)
{
    const unsigned int contextID = state.getContextID();
    assert(_contextID == contextID);
    if (!_registerTexture(counts, _counts, this, contextID) ||
        !_registerTexture(heads, _heads, this, contextID) ||
        !_registerTexture(fragments, _fragments, this, contextID))
    {
        _unregisterTexture(_counts, this);
        _unregisterTexture(_heads, this);
        _unregisterTexture(_fragments, this);
        return false;
    }
    return true;
}

bool MultiFragmentFunctors::_mapTextures(const cudaStream_t stream)
{
    if (!_counts.resource)
        return true; /* Testing codepath, textures don't come from OpenGL */

    cudaGraphicsResource_t resources[] = {_counts.resource, _heads.resource,
                                          _fragments.resource};
    if (cudaGraphicsMapResources(3, resources, stream) != cudaSuccess)
    {
        LBERROR << "CUDA mapping resources: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return false;
    }

    return _mapTexture(_counts) && _mapTexture(_heads) &&
           _mapTexture(_fragments);
}

void MultiFragmentFunctors::_unmapTextures(const cudaStream_t stream)
{
    if (!_counts.resource)
        return; /* Testing codepath, textures don't come from OpenGL */

    cudaGraphicsResource_t resources[] = {_counts.resource, _heads.resource,
                                          _fragments.resource};
    if (cudaGraphicsUnmapResources(3, resources, stream) != cudaSuccess)
    {
        LBERROR << "CUDA error during texture unmapping: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
}

void MultiFragmentFunctors::_allocHostBuffers(const PixelViewports& viewports,
                                              const std::vector<int>& locations)
{
    /* Allocating the buffers to hold all offset and fragment data. This
       allocations are kept bewteen frames because cudaMallocHost is expensive,
       being only changed if they need to grow. */

    size_t size = 0;

    /* Offset buffer. */
    for (size_t i = 0; i != viewports.size(); ++i)
    {
        const auto& vp = viewports[i];
        if (locations.empty() || locations[i] != Location::currentDevice)
            size += (vp.w * vp.h + 1) * sizeof(uint32_t);
    }
    if (_offsetBuffer.size < size)
    {
        char* buffer;
        const auto error = cudaMallocHost(&buffer, size);
        if (error != cudaSuccess)
        {
            LBERROR << "Error allocating device memory: "
                    << cudaGetErrorString(error) << std::endl;
            abort();
        }
        _offsetBuffer.size = size;
        _offsetBuffer.ptr.reset(buffer,
                                _cudaContext->cudaFreeFunctor(cudaFreeHost));
    }

    /* Fragment buffer.
       In this case we are allocating more than strictly needed because we
       don't need to allocate space for the viewports that stay in device
       memory. However, the only way to subtract the fragments counts for these
       viewports is to solve the counts to offsets and then retrieve the total
       fragment counts. */
    size = _totalNumFragments * sizeof(uint32_t) * 2;
    if (_fragmentBuffer.size < size)
    {
        char* buffer;
        if (cudaMallocHost(&buffer, size) != cudaSuccess)
        {
            LBERROR << "Error allocating device memory: "
                    << cudaGetErrorString(cudaGetLastError()) << std::endl;
            abort();
        }
        _fragmentBuffer.size = size;
        _fragmentBuffer.ptr.reset(buffer,
                                  _cudaContext->cudaFreeFunctor(cudaFreeHost));
    }
}

void MultiFragmentFunctors::objectDeleted(void* object)
{
    auto texture = static_cast<osg::TextureRectangle*>(object);
    if (texture == _counts.texture)
        _unregisterTexture(_counts);
    if (texture == _heads.texture)
        _unregisterTexture(_heads);
    if (texture == _fragments.texture)
        _unregisterTexture(_fragments);
}
}
}
}
