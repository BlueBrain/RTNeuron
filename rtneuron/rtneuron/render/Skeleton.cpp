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

#include <list>

#include "Skeleton.h"
#include "timing.h"

#include "util/cameraToID.h"
#include "util/shapes.h"

#ifdef USE_CUDA
#include "cuda/CUDAContext.h"
#include "scene/CollectSkeletonsVisitor.h"
#endif

#include <osg/BlendEquation>
#include <osg/Geode>
#include <osg/PolygonMode>
#include <osg/ShapeDrawable>
#include <osg/ValueObject>
#include <osg/Version>
#include <osgUtil/CullVisitor>
#include <osgViewer/Renderer>

#include <vmmlib/frustum.hpp>

#include <boost/bind.hpp>
#include <boost/function/function0.hpp>
#include <boost/function/function1.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/weak_ptr.hpp>

#include <mutex>
#include <unordered_set>

// This seems to be buggy if multiple contexts are available. Also the
// stream is not cleaned up properly.
//#define CREATE_OWN_STREAMS

#ifdef WIN32
#undef near
#undef far
#endif

namespace bbp
{
namespace rtneuron
{
namespace core
{
using vmml::Frustumd;

/*
  Static declarations
*/

const bool Skeleton::s_useSharedMemory =
    ::getenv("RTNEURON_SHARED_MEM_CULL_KERNEL") != 0;

namespace
{
#ifdef USE_CUDA
const bool s_globalVisibilityReadback =
    ::getenv("RTNEURON_PER_SKELETON_READBACK") == 0;
#endif

std::mutex s_globalSkeletonDataMutex;

/*
   Helper functions and declarations
*/

enum MaskingTraversalResult
{
    NEXT_CAPSULE,
    NEXT_RANGE
};

/**
   Special smart pointer for arrays to store both strong and weak references.
 */
template <typename T>
class ArrayPtr
{
public:
    ArrayPtr() {}
    ArrayPtr(const boost::shared_array<T>& array)
        : _array(array)
    {
    }

    ArrayPtr(T* array)
        : _array(array)
    {
    }

    ArrayPtr(const boost::shared_ptr<T>& ref)
        : _weakRef(ref)
    {
    }

    operator T*()
    {
        T* pointer = _array.get();
        if (!pointer)
            pointer = _weakRef.lock().get();
        return pointer;
    }

    operator const T*() const
    {
        const T* pointer = _array.get();
        if (!pointer)
            pointer = _weakRef.lock().get();
        return pointer;
    }

    T* get() { return operator T*(); }
    const T* get() const { return operator const T*(); }
private:
    boost::weak_ptr<T> _weakRef;
    boost::shared_array<T> _array;
};

#ifdef USE_CUDA

class FrustumCache
{
    /* Declarations */
protected:
    struct Data
    {
        Data()
            : init(false)
            , frustum()
            , rows(0)
            , columns(0)
            , frameNumber(0)
            , changed(true)
        {
        }

        bool init;
        cuda::CullFrustum frustum;
        Frustumd rltbnf;

        unsigned int rows;
        unsigned int columns;

        unsigned int frameNumber;

        bool changed;
    };

    /* Member functions */
public:
    bool getFrustum(osgUtil::CullVisitor* culler,
                    const cuda::CullFrustum*& frustum)
    {
        osg::RenderInfo& renderInfo = culler->getRenderInfo();
        const osg::State* state = renderInfo.getState();

        Data& data = _cache[getCameraAndEyeID(renderInfo)];
        frustum = &data.frustum;

        if (data.frameNumber == state->getFrameStamp()->getFrameNumber())
            return data.changed;

        data.frameNumber = state->getFrameStamp()->getFrameNumber();
        data.changed = updateFrustum(culler, data);
        data.init = true;

        return data.changed;
    }

protected:
    bool updateFrustum(osgUtil::CullVisitor* culler, Data& data)
    {
        Frustumd frustum;
        culler->getProjectionMatrix()->getFrustum(
            frustum.left(), frustum.right(), frustum.bottom(), frustum.top(),
            frustum.nearPlane(), frustum.farPlane());

        if (data.init && frustum.equals(data.rltbnf))
            return false;

        /* Computing Top, bottom, right and left plane normals. */
        /* Right plane normal */
        osg::Vec2d rn(frustum.nearPlane(), frustum.right());
        rn.normalize();
        /* Left plane normal */
        osg::Vec2d ln(frustum.nearPlane(), frustum.left());
        ln.normalize();
        /* Top plane normal */
        osg::Vec2d tn(frustum.nearPlane(), frustum.top());
        tn.normalize();
        /* Bottom plane normal */
        osg::Vec2d bn(frustum.nearPlane(), frustum.bottom());
        bn.normalize();
        data.frustum.near = -frustum.nearPlane();
        data.frustum.far = -frustum.farPlane();
        data.frustum.rn.x = rn.x();
        data.frustum.rn.y = rn.y();
        data.frustum.ln.x = ln.x();
        data.frustum.ln.y = ln.y();
        data.frustum.tn.x = tn.x();
        data.frustum.tn.y = tn.y();
        data.frustum.bn.x = bn.x();
        data.frustum.bn.y = bn.y();
        data.rltbnf = frustum;
        return true;
    }

    /* Member attributes */
protected:
    /* Here the culler is used as the map key instead of the graphics context
       because otherwise culling with quad buffered stereo won't work right. */
    typedef AutoExpandLFVector<Data> Cache;
    Cache _cache;
};
FrustumCache s_frustumCache;

#endif // USE_CUDA

} // namespace anonymous

/*
  Nested classes
*/

/*
  class Skeleton::PerCameraAndEyeState
*/

#ifdef USE_CUDA
class Skeleton::PerCameraAndEyeState
{
public:
    /* These device and host arrays may be globally shared among all the
       skeletons to be culled.
       MasterCullCallback holds the array and initializes these pointers, which
       are
       weak references. */
    ArrayPtr<uint32_t> devVisibilities;
    ArrayPtr<uint32_t> visibilities;
    /* This is the offset of this particular skeleton in the
       _devVisibilities array */
    size_t offset{std::numeric_limits<size_t>::max()};
    /* Visibilities array size. Only defined if an array per skeleton
       is used. */
    size_t size{0};

    unsigned int visibilityVersion{0};
    CullingState cullingState{Skeleton::FULLY_VISIBLE};

    // Replace with a camera/eye indexed AutoExpandLFVector?
    typedef std::map<osgUtil::CullVisitor*, osg::Matrixf> ModelViewCache;
    ModelViewCache modelViewCache;

    /* Used with per skeleton readback */
    cudaEvent_t ready{0};
    bool pending{false};

    static void dirty(PerCameraAndEyeStates& states)
    {
        /* The input vector is supposed to be indexed by values returned by
           getCameraAndEyeID().
           This code increments all version numbers of the input state vector,
           even for states which are not really used by the application
           (e.g. right/left eyes). */
        for (PerCameraAndEyeState& state : states)
            ++state.visibilityVersion;
    }
};
#endif

/*
  class Skeleton::MasterCullCallbackImpl
*/

#ifdef USE_CUDA

class MasterCallbackSharedData
{
public:
    typedef std::list<boost::function0<void>> CullCallbacks;
    CullCallbacks cullPendingFunctors;
    typedef std::list<SkeletonPtr> Skeletons;
    Skeletons visibilityAllocPendingSkeletons;
    Skeletons readbackPendingSkeletons;
    Skeletons cullPendingSkeletons;

    static MasterCallbackSharedData* getActive(unsigned int cameraEyeID)
    {
        return _activeSharedData[cameraEyeID];
    }

    static void setActive(unsigned int cameraEyeID,
                          MasterCallbackSharedData* data)
    {
        _activeSharedData[cameraEyeID] = data;
    }

    static AutoExpandLFVector<MasterCallbackSharedData*> _activeSharedData;
};
AutoExpandLFVector<MasterCallbackSharedData*>
    MasterCallbackSharedData::_activeSharedData;

class Skeleton::MasterCullCallbackImpl : public Skeleton::MasterCullCallback
{
public:
    void operator()(osg::Node* node, osg::NodeVisitor* nv);

    void allocVisibilities(osg::RenderInfo& renderInfo);

    void synchCulledSkeletons(osg::RenderInfo& renderInfo);

    /**
       Traverses the subgraphs to find new skeletons and uploads then
       to the device associated to the graphics context of the render info.
    */
    void compileSkeletons(osg::RenderInfo& renderInfo, osg::Node* subgraph);

private:
    typedef std::pair<size_t, boost::shared_ptr<uint32_t>> SizePointer;
    typedef AutoExpandLFVector<SizePointer> PerCameraAndEyeVisibilities;

    PerCameraAndEyeVisibilities _deviceVisibilities;
    PerCameraAndEyeVisibilities _hostVisibilities;

    typedef AutoExpandLFVector<bool> PerCameraAndEyeNewSkeletonsFlag;
    PerCameraAndEyeNewSkeletonsFlag _allSkeletonDataUploaded;

    /* These data types refer to the visibility arrays */
    typedef std::unordered_set<Skeleton*> AllocatedSkeletons;
    typedef AutoExpandLFVector<AllocatedSkeletons> PerDeviceAllocatedSkeletons;
    PerDeviceAllocatedSkeletons _allocatedSkeletons;

    AutoExpandLFVector<MasterCallbackSharedData> _sharedCullData;
};

void Skeleton::MasterCullCallbackImpl::operator()(osg::Node* node,
                                                  osg::NodeVisitor* nv)
{
    osgUtil::CullVisitor* culler = dynamic_cast<osgUtil::CullVisitor*>(nv);
    osg::RenderInfo& renderInfo = culler->getRenderInfo();
    osg::Camera* camera = culler->getCurrentCamera();

    assert(camera->getGraphicsContext() ==
           culler->getState()->getGraphicsContext());

    unsigned int cameraID = getCameraAndEyeID(camera);
    MasterCallbackSharedData::setActive(cameraID, &_sharedCullData[cameraID]);

    /* No camera is current in renderInfo until this call */
    renderInfo.pushCamera(camera);
    /** \todo This solution won't work for pipelined cull and draw */
    osg::GraphicsContext* context = camera->getGraphicsContext();
    context->setUserValue("cam_eye_id", cameraID);

    /* Creating a CUDA context attached to this GraphicsContext if needed
       The context can then be retrieved during the draw traversal (which
       should be done in the same thread). */
    ScopedCUDAContext scopedContext(CUDAContext::getOrCreateContext(context));

    /* Clearing the device visibility array if already allocated. */
    const SizePointer& visibilities = _deviceVisibilities[cameraID];
    if (visibilities.second.get() != 0)
    {
        cudaMemset(visibilities.second.get(), 0,
                   sizeof(uint32_t) * visibilities.first);
    }

    traverse(node, nv);

    MasterCallbackSharedData& sharedCullData = _sharedCullData[cameraID];

    if (!sharedCullData.visibilityAllocPendingSkeletons.empty())
    {
        if (s_globalVisibilityReadback)
        {
            /* Synching all skeletons that were not deferred, otherwise, its
               culling results will be lost when the visibility array is
               changed. */
            synchCulledSkeletons(renderInfo);
        }
        allocVisibilities(renderInfo);
    }

    /* Resolving the culling of skeletons which have been just allocated and
       are also visible. */
    typedef std::list<boost::function0<void>> CullCallbacks;
    CullCallbacks callbacks;
    callbacks.splice(callbacks.begin(), sharedCullData.cullPendingFunctors);
    for (CullCallbacks::iterator i = callbacks.begin(); i != callbacks.end();
         ++i)
        (*i)();

    if (s_globalVisibilityReadback)
    {
        sharedCullData.readbackPendingSkeletons.splice(
            sharedCullData.readbackPendingSkeletons.end(),
            sharedCullData.cullPendingSkeletons);

        /* Synching kernels and reading back visibility info */
        synchCulledSkeletons(renderInfo);
    }

    renderInfo.popCamera();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cerr << "RTNeuron: CUDA error at the end of cull traversal: "
                  << cudaGetErrorString(error) << std::endl;

    MasterCallbackSharedData::setActive(cameraID, 0);
}

void Skeleton::MasterCullCallbackImpl::allocVisibilities(
    osg::RenderInfo& renderInfo)
{
    unsigned int cameraID = getCameraAndEyeID(renderInfo);

    osg::GraphicsContext* context = renderInfo.getState()->getGraphicsContext();
    CUDAContext* cudaContext =
        dynamic_cast<CUDAContext*>(context->getUserData());

    MasterCallbackSharedData& sharedCullData = _sharedCullData[cameraID];
    typedef MasterCallbackSharedData::Skeletons Skeletons;
    Skeletons& skeletons = sharedCullData.visibilityAllocPendingSkeletons;
    AllocatedSkeletons& allocated = _allocatedSkeletons[cameraID];

    /* Checking if we need to allocate any new space. */
    bool needRealloc = false;
    for (auto i : skeletons)
    {
        Skeleton& skeleton = *i;

        if (allocated.insert(&skeleton).second)
            needRealloc = true;
    }
    skeletons.clear();

    if (!needRealloc)
        return;

    /* Allocating device and host memory for the current skeletons. */
    uint32_t totalSections = 0;
    for (auto skeleton : allocated)
        totalSections += skeleton->_sectionCount;

    /* Allocating visibilities array. */
    void* devVisibilitiesPtr;
    if (cudaMalloc(&devVisibilitiesPtr, sizeof(uint32_t) * totalSections) !=
        cudaSuccess)
    {
#ifndef NDEBUG
        std::cerr << "Couldn't allocate " << sizeof(uint32_t) * totalSections
                  << " bytes on device memory " << std::endl;
#endif
        throw std::bad_alloc();
    }

    /* First memory clear for visibility array */
    cudaMemset(devVisibilitiesPtr, 0, sizeof(uint32_t) * totalSections);

    boost::shared_ptr<uint32_t> devVisibilities((uint32_t*)devVisibilitiesPtr,
                                                cudaContext->cudaFreeFunctor(
                                                    cudaFree));

    void* visibilitiesPtr;
    if (cudaMallocHost(&visibilitiesPtr, totalSections * sizeof(uint32_t)) !=
        cudaSuccess)
    {
#ifndef NDEBUG
        std::cerr << "Couldn't allocate " << totalSections * sizeof(uint32_t)
                  << " bytes of page-locked memory " << std::endl;
#endif
        throw std::bad_alloc();
    }
    boost::shared_ptr<uint32_t> visibilities((uint32_t*)visibilitiesPtr,
                                             cudaContext->cudaFreeFunctor(
                                                 cudaFreeHost));

    size_t offset = 0;
    for (auto skeleton : allocated)
    {
        Skeleton::PerCameraAndEyeState& state = skeleton->_state[cameraID];
        state.devVisibilities = devVisibilities;
        if (state.visibilities)
        {
            /* Copying the current visibility masks from the old host vector to
               the new one because they may have been just computed. */
            memcpy(visibilities.get() + offset,
                   state.visibilities.get() + state.offset,
                   sizeof(uint32_t) * skeleton->_sectionCount);
        }
        state.offset = offset;
        state.visibilities = visibilities;
        offset += skeleton->_sectionCount;
    }

    /* Skeletons store weak references, so replacing the pointers is the
       last thing done. */
    _deviceVisibilities[cameraID] =
        std::make_pair(totalSections, devVisibilities);
    _hostVisibilities[cameraID] = std::make_pair(totalSections, visibilities);
}

void Skeleton::MasterCullCallbackImpl::synchCulledSkeletons(
    osg::RenderInfo& renderInfo)
{
    unsigned int cameraID = getCameraAndEyeID(renderInfo);

    typedef MasterCallbackSharedData::Skeletons Skeletons;
    Skeletons skeletons;
    skeletons.splice(skeletons.begin(),
                     _sharedCullData[cameraID].readbackPendingSkeletons);
    if (skeletons.empty())
        return;

    size_t start = std::numeric_limits<size_t>::max();
    size_t end = 0;
    for (auto s : skeletons)
    {
        Skeleton& skeleton = *s;
        Skeleton::PerCameraAndEyeState& state = skeleton._state[cameraID];
        start = std::min(start, state.offset);
        end = std::max(end, state.offset + skeleton._sectionCount);
#ifdef CREATE_OWN_STREAMS
#ifndef NDEBUG
        cudaError_t error = cudaStreamSynchronize(skeleton._stream);
        if (error != cudaSuccess)
        {
            std::cerr << cudaGetErrorString(error) << skeleton._stream
                      << std::endl;
            abort();
        }
#else
        cudaStreamSynchronize(skeleton._stream);
#endif
#endif
    }

#ifndef CREATE_OWN_STREAMS
    osg::GraphicsContext* context = renderInfo.getState()->getGraphicsContext();
    CUDAContext* cudaContext =
        dynamic_cast<CUDAContext*>(context->getUserData());
    cudaStreamSynchronize(cudaContext->defaultStream());
#endif

    uint32_t* devVisibilities = _deviceVisibilities[cameraID].second.get();
    uint32_t* visibilities = _hostVisibilities[cameraID].second.get();

    /** Need to think about how to avoid this intermediate copy
        without screwing up the skeletons visibilities for those
        skeletons that are included in the output array, have been
        cleared, but haven't been processed in this batch. */
    assert(start < end);
    boost::shared_array<uint32_t> tmp(new uint32_t[end - start]);
#ifndef NDEBUG
    cudaError_t error;
    error =
        cudaMemcpy(tmp.get(), devVisibilities + start,
                   sizeof(uint32_t) * (end - start), cudaMemcpyDeviceToHost);
    assert(error == cudaSuccess);
#else
    cudaMemcpy(tmp.get(), devVisibilities + start,
               sizeof(uint32_t) * (end - start), cudaMemcpyDeviceToHost);
#endif
    /* Copying to the final visibility array only the chunks of the
       temporary array that belong to the skeletons culled in this
       frame.
       This is a solution to the artifacts caused by:
       - Changing the LOD threshold with a still camera
       - Replacing the visibility array when new skeletons appear inside
       the frustum and no precopilation step is used. */
    for (auto skeleton : skeletons)
    {
        Skeleton::PerCameraAndEyeState& state = skeleton->_state[cameraID];

        memcpy(visibilities + state.offset, &tmp[state.offset - start],
               skeleton->_sectionCount * sizeof(uint32_t));
    }
}

void Skeleton::MasterCullCallbackImpl::compileSkeletons(
    osg::RenderInfo& renderInfo, osg::Node* subgraph)
{
    osg::State* state = renderInfo.getState();
    osg::GraphicsContext* context = state->getGraphicsContext();
    unsigned int cameraID = getCameraAndEyeID(renderInfo);

    ScopedCUDAContext scopedContext(CUDAContext::getOrCreateContext(context));

    CollectSkeletonsVisitor visitor(renderInfo);
    subgraph->accept(visitor);

    if (visitor.visibilityAllocPending.size() == 0)
        return;

    /* Uploading the skeleton data to the current device. */
    for (auto skeleton : visitor.deviceAllocPending)
        skeleton->uploadToCurrentDevice();

    /* Allocating the visiblity arrays. */
    for (auto skeleton : visitor.visibilityAllocPending)
        _sharedCullData[cameraID].visibilityAllocPendingSkeletons.push_back(
            skeleton);

    allocVisibilities(renderInfo);
}

/*
  class Skeleton::TrivialMasterCullCallback
*/

class Skeleton::TrivialMasterCullCallback : public Skeleton::MasterCullCallback
{
public:
    void operator()(osg::Node* node, osg::NodeVisitor* nv);

    void compileSkeletons(osg::RenderInfo&, osg::Node*) {}
    void issueCulledSkeletonsReadback(osg::RenderInfo& renderInfo);

protected:
    AutoExpandLFVector<MasterCallbackSharedData> _sharedCullData;
};

void Skeleton::TrivialMasterCullCallback::operator()(osg::Node* node,
                                                     osg::NodeVisitor* nv)
{
    osgUtil::CullVisitor* culler = dynamic_cast<osgUtil::CullVisitor*>(nv);
    osg::RenderInfo& renderInfo = culler->getRenderInfo();
    osg::Camera* camera = culler->getCurrentCamera();

    assert(camera->getGraphicsContext() ==
           culler->getState()->getGraphicsContext());

    unsigned int cameraID = getCameraAndEyeID(camera);
    MasterCallbackSharedData::setActive(cameraID, &_sharedCullData[cameraID]);

    /* Creating a CUDA context attached to this GraphicsContext if needed
       The context can then be retrieved during the draw traversal (which
       should be done in the same thread). */
    osg::GraphicsContext* context = camera->getGraphicsContext();
    ScopedCUDAContext scopedContext(CUDAContext::getOrCreateContext(context));

    renderInfo.pushCamera(camera);

    traverse(node, nv);

    /* Only issues the read back in the stream, the final synchronization
       is done when Skeleton::visibilities is called from the drawing
       code. */
    issueCulledSkeletonsReadback(renderInfo);

    renderInfo.popCamera();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cerr << "RTNeuron: CUDA error at the end of cull traversal: "
                  << cudaGetErrorString(error) << std::endl;

    MasterCallbackSharedData::setActive(cameraID, 0);
}

void Skeleton::TrivialMasterCullCallback::issueCulledSkeletonsReadback(
    osg::RenderInfo& renderInfo)
{
    unsigned int cameraEyeID = getCameraAndEyeID(renderInfo);
    CUDAContext* cudaContext = CUDAContext::getCurrentContext();
    cudaStream_t stream = cudaContext->defaultStream();

    typedef MasterCallbackSharedData::Skeletons Skeletons;
    Skeletons skeletons;
    skeletons.splice(skeletons.begin(),
                     _sharedCullData[cameraEyeID].readbackPendingSkeletons);
    for (Skeletons::iterator s = skeletons.begin(); s != skeletons.end(); ++s)
    {
        Skeleton* skeleton = s->get();
        Skeleton::PerCameraAndEyeState& state = skeleton->_state[cameraEyeID];

        cudaMemcpyAsync(state.visibilities.get(), state.devVisibilities.get(),
                        state.size, cudaMemcpyDeviceToHost, stream);
        cudaError_t error = cudaEventRecord(state.ready);
        if (error != cudaSuccess)
        {
            std::cerr << "RTNeuron: Error in visibility result readback: "
                      << cudaGetErrorString(error) << std::endl;
            abort();
        }
    }
}

/*
  class Skeleton::CullCallback
*/

void Skeleton::CullCallback::accept(CollectSkeletonsVisitor& visitor) const
{
    osg::RenderInfo& renderInfo = visitor.getRenderInfo();

    const Skeleton::PerCameraAndEyeState& state =
        _skeleton->_state[getCameraAndEyeID(renderInfo)];
    if (!state.devVisibilities)
        visitor.visibilityAllocPending.insert(_skeleton);

    unsigned int device =
        CUDAContext::getDeviceID(renderInfo.getState()->getGraphicsContext());
    if ((*_skeleton->_perDeviceArray)[device].get() == 0)
        visitor.deviceAllocPending.insert(_skeleton);
}

class Skeleton::CullCallbackImpl : public Skeleton::CullCallback
{
public:
    CullCallbackImpl(const SkeletonPtr& skeleton)
        : CullCallback(skeleton)
    {
    }

    virtual bool cull(osg::NodeVisitor* nv, osg::Drawable* drawable,
                      osg::RenderInfo* renderInfo) const;

private:
    void _cull(osg::NodeVisitor* nv, osg::Drawable* drawable,
               const unsigned int cameraEyeID) const;

    template <typename Function>
    void _launchOrQueueCullKernel(const Function& function,
                                  const unsigned int cameraEyeID,
                                  PerCameraAndEyeState& state) const;

    bool _getUpdatedModelView(const osg::Matrixf*& modelView,
                              osgUtil::CullVisitor* culler,
                              PerCameraAndEyeState& state) const;
};

bool Skeleton::CullCallbackImpl::cull(osg::NodeVisitor* nv,
                                      osg::Drawable* drawable,
                                      osg::RenderInfo* renderInfo) const
{
    unsigned int cameraEyeID = getCameraAndEyeID(*renderInfo);
    Skeleton::PerCameraAndEyeState& state = _skeleton->_state[cameraEyeID];

    osgUtil::CullVisitor* culler = dynamic_cast<osgUtil::CullVisitor*>(nv);

    /* Uploading the skeleton info to the current device if needed */
    _skeleton->uploadToCurrentDevice();

    if (!state.devVisibilities)
    {
        /* Allocating the visibility output array or annotating the skeleton
           to do it later */
        if (s_globalVisibilityReadback)
            MasterCallbackSharedData::getActive(cameraEyeID)
                ->visibilityAllocPendingSkeletons.push_back(_skeleton);
        else
            _skeleton->_allocVisibilities(state);
    }

    bool culled;
    culled = culler->isCulled(drawable->getBound());
    if (!culled)
        _cull(culler, drawable, cameraEyeID);
    else
        state.cullingState = NOT_VISIBLE;
    return culled;
}

void Skeleton::CullCallbackImpl::_cull(osg::NodeVisitor* nv,
                                       osg::Drawable* drawable,
                                       const unsigned int cameraEyeID) const
{
    osgUtil::CullVisitor* culler = dynamic_cast<osgUtil::CullVisitor*>(nv);
    PerCameraAndEyeState& state = _skeleton->_state[cameraEyeID];

    if (culler->getCurrentCullingSet().getFrustum().containsAllOf(
            drawable->getBound()))
    {
        if (state.cullingState != FULLY_VISIBLE)
            ++state.visibilityVersion;
        state.cullingState = FULLY_VISIBLE;
        return;
    }

    const cuda::CullFrustum* frustum;
    const osg::Matrixf* modelView;

    bool changed = s_frustumCache.getFrustum(culler, frustum);
    changed |= _getUpdatedModelView(modelView, culler, state);

    if (state.cullingState != VISIBLE || changed)
        ++state.visibilityVersion;
    state.cullingState = VISIBLE;

    if (!changed)
        return;

    boost::function0<void> cullFunction =
        boost::bind(&Skeleton::_cull, _skeleton, boost::cref(*modelView),
                    boost::cref(*frustum), boost::ref(state));
    _launchOrQueueCullKernel(cullFunction, cameraEyeID, state);
}

template <typename Function>
void Skeleton::CullCallbackImpl::_launchOrQueueCullKernel(
    const Function& cullFunction, const unsigned int cameraEyeID,
    PerCameraAndEyeState& state) const
{
    MasterCallbackSharedData& sharedCullData =
        *MasterCallbackSharedData::getActive(cameraEyeID);

    if (state.devVisibilities)
    {
        cullFunction();
        sharedCullData.readbackPendingSkeletons.push_back(_skeleton);
    }
    else
    {
        assert(s_globalVisibilityReadback);
        /* Leaving this operation pending until the device memory has
           been allocated. */
        sharedCullData.cullPendingFunctors.push_back(cullFunction);
        sharedCullData.cullPendingSkeletons.push_back(_skeleton);
    }

    if (!s_globalVisibilityReadback)
        state.pending = true;
}

bool Skeleton::CullCallbackImpl::_getUpdatedModelView(
    const osg::Matrixf*& modelView, osgUtil::CullVisitor* culler,
    PerCameraAndEyeState& state) const
{
    const osg::Matrix& matrix = *culler->getModelViewMatrix();
    osg::Matrixf& cached = state.modelViewCache[culler];
    bool changed = false;
    for (int i = 0; i < 16; ++i)
    {
        if (cached.ptr()[i] != (float)matrix.ptr()[i])
        {
            changed = true;
            cached.ptr()[i] = matrix.ptr()[i];
        }
    }
    modelView = &cached;

#ifndef NDEBUG
    if (changed)
    {
        const osg::Vec3 scale = cached.getScale();
        if (std::abs(scale.x() - scale.y()) > 5e-6 ||
            std::abs(scale.y() - scale.z()) > 5e-6 ||
            std::abs(scale.z() - scale.x()) > 5e-6)
        {
            std::stringstream msg;
            msg << "RTNeuron: Asymmetric scaling factors in model-view "
                   "matrices are unsupported by CUDA-based view frustum "
                   "culling. Scaling is "
                << scale.x() << ' ' << scale.y() << ' ' << scale.z();
            LBWARN << msg.str() << std::endl;
        }
    }
#endif

    return changed;
}

#endif // USE_CUDA

struct Skeleton::SoftClippingInfo
{
    uint32_ts masks;
    std::vector<uint32_ts> protectedMasks;
};

/*
  Constructors/destructor
*/
Skeleton::Skeleton()
    : _clippingInfo(new SoftClippingInfo)
{
}

Skeleton::Skeleton(const Skeleton& skeleton)
    : _length(skeleton._length)
    , _sectionCount(skeleton._sectionCount)
    , _starts(skeleton._starts)
    , _ends(skeleton._ends)
    , _widths(skeleton._widths)
    , _portions(skeleton._portions)
    , _sections(skeleton._sections)
    , _portionCounts(skeleton._portionCounts)
#ifdef USE_CUDA
    , _blockCount(skeleton._blockCount)
    , _sectionStartsLength(skeleton._sectionStartsLength)
    , _firstBlockSection(skeleton._firstBlockSection)
    , _perBlockSectionsStarts(skeleton._perBlockSectionsStarts)
    , _accumSectionsPerBlock(skeleton._accumSectionsPerBlock)
#endif
    , _startPositions(skeleton._startPositions)
    , _endPositions(skeleton._endPositions)
    , _branchOrders(skeleton._branchOrders)
#ifdef USE_CUDA
    , _perDeviceArray(skeleton._perDeviceArray)
    , _deviceArraySize(skeleton._deviceArraySize)
    , _stream(0)
#endif
    , _clippingInfo(new SoftClippingInfo())
    , _maxVisibleOrder(skeleton._maxVisibleOrder)
{
}

Skeleton::~Skeleton()
{
}

/*
  Member functions
*/
#if USE_CUDA
osg::Drawable::CullCallback* Skeleton::createCullCallback(
    const SkeletonPtr& skeleton)
{
    return new CullCallbackImpl(skeleton);
}
#endif // USE_CUDA

const uint32_t* Skeleton::getClipMasks() const
{
    if (_clippingInfo->masks.empty())
        return 0;
    return &_clippingInfo->masks[0];
}

void Skeleton::softClip(const osg::Vec4d& plane)
{
    if (_clippingInfo->masks.empty())
        _clippingInfo->masks.resize(_sectionCount);

    for (size_t i = 0; i < _length; ++i)
    {
        const osg::Vec4 start(_starts[i][0], _starts[i][1], _starts[i][2], 1.0);
        const osg::Vec4 end(_ends[i][0], _ends[i][1], _ends[i][2], 1.0);
        const float width = _widths[i];
        if (start * plane < -width && end * plane < -width)
        {
            /* The capsule is in the negative hemispace so it is marked
               invisible. */
            _clippingInfo->masks[_sections[i]] |= 1 << _portions[i];
        }
    }
    _dirty();
}

void Skeleton::softClip(const uint16_ts& sections)
{
    uint32_ts& masks = _clippingInfo->masks;
    if (masks.empty())
        masks.resize(_sectionCount);
    for (uint16_ts::const_iterator i = sections.begin(); i != sections.end();
         ++i)
    {
        assert(*i < _sectionCount);
        masks[*i] = 0xFFFFFFFF;
    }
    _dirty();
}

void Skeleton::softUnclip(const uint16_ts& sections)
{
    if (_clippingInfo->masks.empty())
        return;

    uint32_ts& masks = _clippingInfo->masks;
    for (const uint16_t section : sections)
    {
        assert(section < _sectionCount);
        if (_clippingInfo->protectedMasks.empty())
        {
            masks[section] = 0;
        }
        else
        {
            masks[section] = _clippingInfo->protectedMasks.back()[section];
        }
    }
    _dirty();
}

void Skeleton::softClip(const uint16_ts& sections, const floats& starts,
                        const floats& ends)
{
    if (sections.empty())
        return;

    if (_clippingInfo->masks.empty())
        _clippingInfo->masks.resize(_sectionCount);

    _applyCapsuleMaskingOperation(
        sections, starts, ends,
        [](uint32_ts& masks, const uint16_t section, const uint8_t portion,
           const float capsuleStart, const float capsuleEnd,
           const float rangeStart, const float rangeEnd) {
            if (rangeEnd < capsuleEnd)
            {
                /* The capsule end is past the current range. It's not clipped
                   and we have to advance to the next range.
                   Capsules ranges are allowed to overlap slighty, this code
                   works as long as the values in _endPositions are
                   monotonically increasing. */
                return NEXT_RANGE;
            }
            else
            {
                if (rangeStart <= capsuleStart)
                {
                    /* The capsule range is within the current range. Marking it
                       as invisible */
                    masks[section] |= 1 << portion;
                }
                return NEXT_CAPSULE;
            }
        });
}

void Skeleton::softUnclip(const uint16_ts& sections, const floats& starts,
                          const floats& ends)
{
    if (_clippingInfo->masks.empty())
        return;

    const auto& protectedMasks = _clippingInfo->protectedMasks;
    _applyCapsuleMaskingOperation(
        sections, starts, ends,
        [&protectedMasks](uint32_ts& masks, const uint16_t section,
                          const uint8_t portion, const float capsuleStart,
                          const float capsuleEnd, const float rangeStart,
                          const float rangeEnd) {
            if (rangeEnd < capsuleStart)
            {
                /* The capsule start is past the current range. It can't be
                   unclipped and we have to advance to the next range.
                   Capsules range can overlap slighty, this code works as long
                   as the values in _endPositions are monotonically
                   increasing. */
                return NEXT_RANGE;
            }
            else
            {
                if (rangeStart < capsuleEnd)
                {
                    /* The capsule range is within the current range. Marking it
                       as visible */
                    masks[section] &= ~(1 << portion);

                    /* Merging the protected masks if needed */
                    if (!protectedMasks.empty())
                        masks[section] |= protectedMasks.back()[section];
                }
                return NEXT_CAPSULE;
            }
        });
    _dirty();
}

void Skeleton::softClipAll()
{
    uint32_ts& masks = _clippingInfo->masks;
    if (masks.empty())
        masks.resize(_sectionCount);
    for (size_t i = 0; i != _sectionCount; ++i)
        masks[i] = 0xFFFFFFFF;
    _dirty();
}

void Skeleton::softUnclipAll()
{
    uint32_ts& masks = _clippingInfo->masks;
    if (_clippingInfo->protectedMasks.empty())
        uint32_ts().swap(masks);
    else
    {
        /* Protecting an empty set of masks is an error, therefore this
           operation can't have cleared the masks. */
        assert(!masks.empty());
        std::copy(_clippingInfo->protectedMasks.back().begin(),
                  _clippingInfo->protectedMasks.back().end(), masks.begin());
    }
    _dirty();
}

void Skeleton::protectClipMasks()
{
    uint32_ts& masks = _clippingInfo->masks;
    assert(!masks.empty());
    _clippingInfo->protectedMasks.push_back(
        uint32_ts(masks.begin(), masks.end()));
}

void Skeleton::popClipMasks()
{
    assert(!_clippingInfo->protectedMasks.empty());
    _clippingInfo->protectedMasks.pop_back();
}

void Skeleton::hardClip(const osg::Vec4d& plane)
{
#ifdef USE_CUDA
    assert(_perDeviceArray.get() == 0 || _perDeviceArray->size() == 0);
    assert(!culledOnce());
    if (s_useSharedMemory)
        LBUNIMPLEMENTED;
#endif

    /* Counting visible capsules to know the final size of the arrays and
       avoid rellocations while filling them up. */
    size_t length = 0;
    for (size_t i = 0; i < _length; ++i)
    {
        const osg::Vec4 start(_starts[i][0], _starts[i][1], _starts[i][2], 1.0);
        const osg::Vec4 end(_ends[i][0], _ends[i][1], _ends[i][2], 1.0);
        const float width = _widths[i];
        if (start * plane >= -width || end * plane >= -width)
            ++length;
    }

    /* Filling the new arrays, correcting portions ids and counts.
       The array with the branch orders is kept as is, because it's indexed
       by section ID, and since section IDs are not remapped, we can't remove
       the clipped sections from it. */
    boost::shared_array<osg::Vec3f> starts(new osg::Vec3f[length]);
    boost::shared_array<osg::Vec3f> ends(new osg::Vec3f[length]);
    boost::shared_array<float> widths(new float[length]);
    boost::shared_array<boost::uint8_t> portions(new boost::uint8_t[length]);
    boost::shared_array<uint16_t> sections(new uint16_t[length]);
    boost::shared_array<float> startPositions(new float[length]);
    boost::shared_array<float> endPositions(new float[length]);
    uint16_t currentSection = 0;
    size_t nextPortion = 0;
    for (size_t i = 0, j = 0; i < _length; ++i)
    {
        const osg::Vec4 start(_starts[i][0], _starts[i][1], _starts[i][2], 1.0);
        const osg::Vec4 end(_ends[i][0], _ends[i][1], _ends[i][2], 1.0);
        const float width = _widths[i];
        if (start * plane < -width && end * plane < -width)
        {
            --_portionCounts[_sections[i]];
        }
        else
        {
            if (currentSection != _sections[i])
            {
                nextPortion = 0;
                currentSection = _sections[i];
            }
            starts[j] = _starts[i];
            ends[j] = _ends[i];
            widths[j] = _widths[i];
            portions[j] = nextPortion;
            sections[j] = _sections[i];
            startPositions[j] = _startPositions[i];
            endPositions[j] = _endPositions[i];
            ++j;
            ++nextPortion;
        }
    }
    /* Discard the sections that have become empty at the end of the ID list */
    size_t trailingEmptySections = 0;
    for (size_t i = _sectionCount; i != 0 && _portionCounts[i - 1] == 0; --i)
        ++trailingEmptySections;
    if (trailingEmptySections != 0)
    {
        _sectionCount -= trailingEmptySections;
        boost::shared_array<unsigned short> counts(
            new unsigned short[_sectionCount]);
        memcpy(&counts[0], &_portionCounts[0],
               sizeof(unsigned short) * _sectionCount);
        _portionCounts = counts;
    }

    _length = length;
    _starts = starts;
    _ends = ends;
    _widths = widths;
    _portions = portions;
    _sections = sections;
    _startPositions = startPositions;
    _endPositions = endPositions;
}

void Skeleton::setMaximumVisibleBranchOrder(const unsigned int order)
{
    _maxVisibleOrder = order;
    _dirty();
}

bool Skeleton::culledOnce() const
{
    return true;
}

unsigned int Skeleton::getVisibilityVersion(
    const unsigned int cameraEyeID) const
{
#ifdef USE_CUDA
    return _state[cameraEyeID].visibilityVersion;
#else
    (void)cameraEyeID;
    return _version;
#endif
}

unsigned int Skeleton::getVisibilityVersion(osg::RenderInfo& renderInfo) const
{
#ifdef USE_CUDA
    return _state[getCameraAndEyeID(renderInfo)].visibilityVersion;
#else
    (void)renderInfo;
    return _version;
#endif
}

const uint32_t* Skeleton::getVisibilities(const unsigned int cameraEyeID)
{
#ifdef USE_CUDA
    if (_state.empty())
        return 0;

    PerCameraAndEyeState& cullState = _state[cameraEyeID];
    /* In CUDA 5 maybe it's possible to simplify this code with a callback
       that changes the monitor state. That way this piece of code code
       be single line regardless of the synchronization style used. */
    if (!s_globalVisibilityReadback && cullState.pending)
    {
        CUDAContext* context = CUDAContext::getCurrentContext();
        assert(context != 0);

        ScopedCUDAContext cudaContext(context);
        cudaEventSynchronize(cullState.ready);
        cullState.pending = false;
    }
    assert(cullState.visibilities.get());
    return cullState.visibilities.get() + cullState.offset;
#else
    (void)cameraEyeID;
    return 0;
#endif
}

const uint32_t* Skeleton::getVisibilities(osg::RenderInfo& renderInfo)
{
    return getVisibilities(getCameraAndEyeID(renderInfo));
}

Skeleton::CullingState Skeleton::getCullingState(
    const unsigned int cameraEyeID) const
{
#ifdef USE_CUDA
    if (_state.empty())
        return FULLY_VISIBLE;
    return _state[cameraEyeID].cullingState;
#else
    (void)cameraEyeID;
    return FULLY_VISIBLE;
#endif
}

osg::Geode* Skeleton::createSkeletonModel(const double detailRatio,
                                          const bool wireframe) const
{
    osg::Geode* geode = new osg::Geode();
    osg::ShapeDrawable* drawable =
        sphereDrawable(_starts[0], _widths[0],
                       osg::Vec4(1, 1, 1, wireframe ? 1.0 : 0.2), detailRatio);
    geode->addDrawable(drawable);

    for (size_t i = 1; i < _length; ++i)
    {
        osg::Vec3 center = (_ends[i] + _starts[i]) * 0.5;
        double radius = _widths[i];
        osg::Vec3 axis = _ends[i] - _starts[i];
        osg::ShapeDrawable* capsule =
            capsuleDrawable(center, radius, axis,
                            osg::Vec4(1, 1, 1, wireframe ? 1.0 : 0.2),
                            detailRatio);
        geode->addDrawable(capsule);
    }
    if (wireframe)
    {
        geode->getOrCreateStateSet()->setAttributeAndModes(
            new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,
                                 osg::PolygonMode::LINE));
    }
    else
    {
        osg::StateSet* stateSet = geode->getOrCreateStateSet();
        stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateSet->setAttributeAndModes(new osg::BlendEquation());
    }
    return geode;
}

size_t Skeleton::getEstimatedHostDataSizeForCUDACull() const
{
    size_t size = (/* sections */
                   sizeof(uint16_t) +
                   /* portion ranges */
                   sizeof(float) * 2) *
                  _length;
    return size;
}

size_t Skeleton::getDeviceSkeletonDataSize() const
{
#ifdef USE_CUDA
    size_t size = (sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) +
                   sizeof(boost::uint8_t)) *
                  _length;
    if (s_useSharedMemory)
    {
        size += sizeof(boost::uint16_t) * _blockCount;
        size += sizeof(boost::uint16_t) * (_blockCount + 1);
        size += sizeof(boost::uint16_t) * _sectionStartsLength;
    }
    else
    {
        size += sizeof(boost::uint16_t) * _length;
    }

    size += sizeof(cuda::SkeletonInfo);
    return size;
#else
    return 0;
#endif
}

#ifdef USE_CUDA
Skeleton::MasterCullCallback* Skeleton::createMasterCullCallback()
{
    if (s_globalVisibilityReadback)
        return new MasterCullCallbackImpl();
    else
        return new TrivialMasterCullCallback();
}
#endif

void Skeleton::allocHostArrays(const Portions* portions)
{
    size_t size = 0;
    _length = portions->size() + 1;

    _starts.reset(new osg::Vec3f[_length]);
    size += sizeof(float) * 3 * _length;
    _ends.reset(new osg::Vec3f[_length]);
    size += sizeof(float) * 3 * _length;
    _widths.reset(new float[_length]);
    size += sizeof(float) * _length;
    _portions.reset(new boost::uint8_t[_length]);
    // cppcheck-suppress unreadVariable
    size += sizeof(boost::uint8_t) * _length;
    _sections.reset(new uint16_t[_length]);

    _startPositions.reset(new float[_length]);
    _endPositions.reset(new float[_length]);

#ifdef USE_CUDA
    if (s_useSharedMemory)
    {
        _blockCount =
            (_length + cuda::CULL_BLOCK_SIZE - 1) / cuda::CULL_BLOCK_SIZE;
        _firstBlockSection.reset(new boost::uint16_t[_blockCount]);
        size += sizeof(boost::uint16_t) * _blockCount;

        /* We add an extra element to ensure that the last block can access
           the total accumulation. */
        _accumSectionsPerBlock.reset(
            new boost::uint16_t[sizeof(boost::uint16_t) * (_blockCount + 1)]);
        size += sizeof(boost::uint16_t) * (_blockCount + 1);

        /* This is the count of the actual sections contained in the portions.
           Soma included by default. */
        size_t sectionCount = 1;
        uint16_t currentSection = -1;
        size_t blockIndex = 1;
        size_t unterminatedSections = 0;
        for (Portions::const_iterator i = portions->begin();
             i != portions->end(); ++i, ++blockIndex)
        {
            bool newSection = currentSection != i->first;
            if (newSection)
            {
                currentSection = i->first;
                ++sectionCount;
            }
            if (blockIndex == cuda::CULL_BLOCK_SIZE)
            {
                blockIndex = 0;
                if (!newSection)
                    ++unterminatedSections; /* Last section from last block
                                               continues in this block */
            }
        }

        /* Here we also add an extra element to reduce the number of conditional
           statements inside the kernel. */
        _sectionStartsLength = sectionCount + unterminatedSections + 1;
        _perBlockSectionsStarts.reset(
            new boost::uint16_t[sizeof(boost::uint16_t) *
                                _sectionStartsLength]);
        size += sizeof(boost::uint16_t) * _sectionStartsLength;
    }
    else
    {
        size += sizeof(boost::uint16_t) * _length;
    }

    size += sizeof(cuda::SkeletonInfo);

    _deviceArraySize = size;
    _perDeviceArray.reset(new DeviceArrays());
#endif // USE_CUDA
}

#ifdef USE_CUDA
void Skeleton::uploadToCurrentDevice()
{
    CUDAContext* context = CUDAContext::getCurrentContext();
    unsigned int deviceID = context->getDeviceID();

    /* This code doesn't need a mutex in general, except for multi-pipe
       configurations on the same graphics device (X server screen). This
       special case is only needed in testing setups (because it doesn't
       make sense performance-wise), but it's a sufficiently strong use
       case to be taken into account */
    std::unique_lock<std::mutex> lock(s_globalSkeletonDataMutex);

    /* Checking if the skeleton needs to be uploaded to this device */
    boost::shared_ptr<uint8_t>& deviceArrayPtr = (*_perDeviceArray)[deviceID];

    if (deviceArrayPtr.get() != 0)
        return; /* Already uploaded */

    void* array;
    if (cudaMalloc(&array, _deviceArraySize) != cudaSuccess)
    {
        std::cerr << "RTNeuron: error allocating device memory: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        abort();
    }
    /* In principle no deallocation of a previous array should occur.
       Anyways the helper pieces are prepared to deal safely with such a
       case. */
    deviceArrayPtr.reset((boost::uint8_t*)array,
                         context->cudaFreeFunctor(cudaFree));

    uint8_t* deviceArray = deviceArrayPtr.get();

    cuda::SkeletonInfo info;
    info.length = _length;
    if (s_useSharedMemory)
    {
        info.blockCount = _blockCount;
        info.sectionsStartsLength = _sectionStartsLength;
    }
    cudaMemcpy(deviceArray, &info, sizeof(cuda::SkeletonInfo),
               cudaMemcpyHostToDevice);
    deviceArray += sizeof(cuda::SkeletonInfo);

    /*! \bug Asynchronous calls don't work as expected. Need to read the
      documention better to understand possible race conditions */
    cudaMemcpy(deviceArray, &_starts[0], sizeof(float3) * _length,
               cudaMemcpyHostToDevice);
    deviceArray += _length * sizeof(float3);

    cudaMemcpy(deviceArray, &_ends[0], sizeof(float3) * _length,
               cudaMemcpyHostToDevice);
    deviceArray += _length * sizeof(float3);

    cudaMemcpy(deviceArray, &_widths[0], sizeof(float) * _length,
               cudaMemcpyHostToDevice);
    deviceArray += _length * sizeof(float);

    if (s_useSharedMemory)
    {
        cudaMemcpy(deviceArray, &_firstBlockSection[0],
                   sizeof(boost::uint16_t) * _blockCount,
                   cudaMemcpyHostToDevice);
        deviceArray += _blockCount * sizeof(boost::uint16_t);

        cudaMemcpy(deviceArray, &_accumSectionsPerBlock[0],
                   sizeof(boost::uint16_t) * (_blockCount + 1),
                   cudaMemcpyHostToDevice);
        deviceArray += (_blockCount + 1) * sizeof(boost::uint16_t);

        cudaMemcpy(deviceArray, &_perBlockSectionsStarts[0],
                   sizeof(boost::uint16_t) * _sectionStartsLength,
                   cudaMemcpyHostToDevice);
        deviceArray += _sectionStartsLength * sizeof(boost::uint16_t);
    }
    else
    {
        cudaMemcpy(deviceArray, &_sections[0],
                   sizeof(boost::uint16_t) * _length, cudaMemcpyHostToDevice);
        deviceArray += _length * sizeof(boost::uint16_t);
    }

    /* We put this last to avoid alignment issues. */
    cudaMemcpy(deviceArray, &_portions[0], sizeof(boost::uint8_t) * _length,
               cudaMemcpyHostToDevice);
}
#endif

#ifdef USE_CUDA
void Skeleton::_cull(const osg::Matrixf& modelview,
                     const cuda::CullFrustum& frustum,
                     PerCameraAndEyeState& state)
{
    CUDAContext* context = CUDAContext::getCurrentContext();
    cudaStream_t stream;
#ifdef CREATE_OWN_STREAMS
    if (_stream == 0)
        _stream = context->createStream();
    stream = _stream;
#else
    stream = context->defaultStream();
#endif
    if (state.ready == 0)
        state.ready = context->createEvent();

    assert(_perDeviceArray.get());
    void* deviceArray = (*_perDeviceArray)[context->getDeviceID()].get();
    assert(state.devVisibilities.get() != 0);
    assert(deviceArray != 0);
    uint32_t* devVisibilities = state.devVisibilities.get() + state.offset;

    /* If using a readback per skeleton, clear the visibility array */
    cudaError_t error;
    if (!s_globalVisibilityReadback)
    {
        error = cudaMemsetAsync(devVisibilities, 0, state.size, stream);
        if (error != cudaSuccess)
        {
            std::cerr << "RTNeuron: Error clearing memory: "
                      << cudaGetErrorString(error) << std::endl;
            abort();
        }
    }

    /* Culling */
    error =
        cuda::cullSkeleton(modelview.ptr(), frustum,
                           reinterpret_cast<cuda::SkeletonInfo*>(deviceArray),
                           devVisibilities, _length, s_useSharedMemory, stream);

    if (error != cudaSuccess)
    {
        std::cerr << "RTNeuron: Error launching kernel: "
                  << cudaGetErrorString(error) << std::endl;
        abort();
    }
}
#endif

#ifdef USE_CUDA
void Skeleton::_allocVisibilities(Skeleton::PerCameraAndEyeState& state)
{
    assert(!s_globalVisibilityReadback);
    CUDAContext* context = CUDAContext::getCurrentContext();

    state.size = _sectionCount * sizeof(uint32_t);

    void* devVisibilitiesPtr;
    if (cudaMalloc(&devVisibilitiesPtr, state.size) != cudaSuccess)
    {
#ifndef NDEBUG
        std::cerr << "Couldn't allocate " << state.size
                  << " bytes on device memory " << std::endl;
#endif
        throw std::bad_alloc();
    }
    state.devVisibilities =
        boost::shared_array<uint32_t>((uint32_t*)devVisibilitiesPtr,
                                      context->cudaFreeFunctor(cudaFree));
    state.offset = 0;

    void* visibilitiesPtr;
    if (cudaMallocHost(&visibilitiesPtr, state.size) != cudaSuccess)
    {
#ifndef NDEBUG
        std::cerr << "Couldn't allocate " << state.size
                  << " bytes of page-locked memory " << std::endl;
#endif
        throw std::bad_alloc();
    }
    state.visibilities =
        boost::shared_array<uint32_t>((uint32_t*)visibilitiesPtr,
                                      context->cudaFreeFunctor(cudaFreeHost));
}
#endif // USE_CUDA

template <typename Operation>
void Skeleton::_applyCapsuleMaskingOperation(const uint16_ts& sections,
                                             const floats& starts,
                                             const floats& ends,
                                             const Operation& operation)
{
    size_t capsule = 0;
    size_t portion = 0;
    size_t range = 0;
    uint32_ts& masks = _clippingInfo->masks;

    _dirty();

    while (range != sections.size() && capsule != _length)
    {
        /* Advancing in the capsule and range arrays until the section id is
           the same in both. This code allows gaps in the section id
           enumeration in both arrays. */
        while (_sections[capsule] != sections[range])
        {
            portion = 0; /* Starting a new section in the capsule array. */
            while (_sections[capsule] < sections[range])
            {
                if (++capsule == _length)
                    return; /* Nothing left to do */
                /* Starting a new section in the capsule array. */
                portion = 0;
            }
            while (sections[range] < _sections[capsule])
            {
                ++range;
                if (range == sections.size())
                    return; /* Nothing left to do */
            }
        }

        if (operation(masks, sections[range], portion, _startPositions[capsule],
                      _endPositions[capsule], starts[range],
                      ends[range]) == NEXT_RANGE)
        {
            ++range;
        }
        else // NEXT_CAPSULE
        {
            ++capsule;
            ++portion;
        }
    }
}

void Skeleton::_dirty()
{
#ifdef USE_CUDA
    PerCameraAndEyeState::dirty(_state);
#else
    ++_version;
#endif
}
}
}
}
