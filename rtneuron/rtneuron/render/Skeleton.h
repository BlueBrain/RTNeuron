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

#ifndef RTNEURON_SKELETON_H
#define RTNEURON_SKELETON_H

#include "coreTypes.h"

#include "timing.h"
#include "util/AutoExpandLFVector.h"

#ifdef USE_CUDA
#include "../cuda/cull.h"
#endif

#include <eq/fabric/eye.h>

#include <osg/Drawable>
#include <osg/Matrix>
#include <osg/Matrixf>
#include <osg/NodeCallback>

#include <boost/shared_array.hpp>
#include <cstdint>

namespace bbp
{
namespace rtneuron
{
namespace core
{
static const unsigned int MAX_CAPSULES_PER_SECTION = 32;

class Skeleton
{
public:
    /*--- Public declarations ---*/

    typedef std::pair<uint16_t, uint16_t> Portion;
    typedef std::vector<Portion> SectionPortions;
    typedef std::multimap<uint16_t, Portion> Portions;

    enum CullingState
    {
        FULLY_VISIBLE,
        VISIBLE,
        NOT_VISIBLE
    };

    struct PortionRanges
    {
        PortionRanges()
            : length(0)
        {
        }

        /* Start and end index of the interval of a primitive contained in
           a skeleton capsule. */
        typedef std::pair<uint32_t, uint32_t> Range;
        typedef boost::shared_array<Range> Ranges; // not shared_ptr until C++17

        Ranges ranges;
        /* Length of the above array. */
        size_t length;
    };

#ifdef USE_CUDA
    class CullCallback : public osg::Drawable::CullCallback
    {
    public:
        void accept(CollectSkeletonsVisitor& visitor) const;

    protected:
        CullCallback(const SkeletonPtr& skeleton)
            : _skeleton(skeleton)
        {
        }

        SkeletonPtr _skeleton;
    };

    class MasterCullCallback : public osg::NodeCallback
    {
    public:
        virtual void operator()(osg::Node* node, osg::NodeVisitor* nv) = 0;

        /**
           Uploads skeleton data to the device associated to the graphics
           context of the render info if possible.
           The skeletons are harvested travering the given subgraph.
        */
        virtual void compileSkeletons(osg::RenderInfo& renderInfo,
                                      osg::Node* subgraph) = 0;
    };
#endif

    /*--- Public constructors/destructor ---*/

    virtual ~Skeleton();

/*--- Public member functions ---*/

#ifdef USE_CUDA
    static osg::Drawable::CullCallback* createCullCallback(
        const SkeletonPtr& skeleton);
#endif
    /**
       Return whether this skeleton has been culled at least once for any
       context.

       This is a workaround to prevent accessing the state during
       compileGLObjects.
    */
    bool culledOnce() const;
    /**
       Curent version number of the visibilities array.

       This number is updated at the end of the culling pass for the current
       frame. If the visiblity version number hasn't changed it means that
       the fine grained culling kernel hasn't been run but the skeleton
       might now be fully visible (call fullyVisible to check for this case).
     */
    unsigned int getVisibilityVersion(const unsigned int cameraEyeID) const;

    /**
       Overloaded version of the previous function which extracts the
       camera/eye identifier from a osg:RenderInfo object.
     */
    unsigned int getVisibilityVersion(osg::RenderInfo& renderInfo) const;

    /**
       Per section visibility array.

       Returns an arrays of integers, each one being a bitfield of the
       visibility of the portions of a single section. The contents of the
       array are only up to date when fullyVisible is false and the culling
       pass has finished completely.
       This function may block if the visibilites computed for this Skeleton
       haven't been copied from device memory yet. (This is needed when the
       cull and draw threads are distinct)

       \bug There is a potenial pitfall here with the current implementation
       and interface. If using CUDA, the MasterCullCallback holds the array,
       it's an error to take this pointer and operate with it after the
       callback is deleted.
     */
    const uint32_t* getVisibilities(const unsigned int cameraEyeID);

    /**
       Overloaded version of the previous function which extracts the
       camera/eye identifier from a osg:RenderInfo object.
     */
    const uint32_t* getVisibilities(osg::RenderInfo& renderInfo);

    /**
       Returns the result of the coarse visibility test on the skeleton.
       Tiling subdivision is considered when appropriate.
    */
    CullingState getCullingState(const unsigned int cameraEyeID) const;

    /**
       Static per section clip masks.

       Returns an array of per section visilibity mask to be applied in addition
       to view frustum culling. Those portions whose mask bit is 1 should
       be considered invisible.

       The clipping mask is produced by applying clipping planes to the
       skeleton or provided by the user. If no clipping has been applied
       a null pointer is returned.
    */
    const uint32_t* getClipMasks() const;

    /**
       Applies a clipping plane to this skeleton.

       The clipping results in an array of clipping masks, where
       the capsules in the negative hemispace are marked as invisible.
       The clipping mask array is created if it doesn't exist.
    */
    void softClip(const osg::Vec4d& plane);

    /**
       Applies a clipping mask to the capsules of the given sections.
    */
    void softClip(const uint16_ts& sections);

    /**
       Clears the clipping mask of the capsules belonging to the given sections.

       Capsules present in the protected masks are not unclipped, i.e. they
       remain clipped.
    */
    void softUnclip(const uint16_ts& sections);

    /**
       Stores the current set of masks in the stack of protected masks.

       Protected masks store the bits of capsules that cannot be unclipped
       by unclip operations. By definition, the set of bits cannot become
       smaller from one set of protected masks to the next set up in the
       stack.

       It is an error to try to protect and empty set of masks.
    */
    void protectClipMasks();

    /**
       Remove the masks at the top of the stack of protected masks.

       Masks are not updated, use softUnclipAll() to clear all bit masks
       excepts those from the new stack top.

       It is an error to pop from an empty stack.
     */
    void popClipMasks();

    /**
       Apply a clipping mask to the capsules of a set of section ranges.

       Creates a clipping mask array if needed. Capsules that are fully
       contained in a any of the input ranges are marked as invisible.
       @param sections A sorted list of section ids. It contains one per range
              and ids may be repeated.
       @param starts A list of range starts. The sublist belonging to each
              section must be sorted.
       @param ends A list of range ends. The sublist belonging to each
              section must be sorted.
    */
    void softClip(const uint16_ts& sections, const floats& starts,
                  const floats& ends);

    /**
       Unclip a set of section ranges.

       If a clipping array exists, the capsules that partially overlap any
       of the input ranges are marked as visible in their section masks.
       The intervals are considered open, i.e. overlapping just at an edge
       is not sufficient.

       Capsules present in the protected masks are not unclipped.

       @param sections A sorted list of section ids. It contains one per range
              and ids may be repeated.
       @param starts A list of range starts. The sublist belonging to each
              section must be sorted.
       @param ends A list of range ends. The sublist belonging to each
              section must be sorted.
    */
    void softUnclip(const uint16_ts& sections, const floats& starts,
                    const floats& ends);

    void softClipAll();

    /**
       Unclips all capsules except those in the protected masks.
     */
    void softUnclipAll();

    /**
       Applies a clipping plane to this skeleton removing clipped
       capsules.

       The capsules in the negative hemispace are removed. Arrays are
       relocated to reduce memory usage. The section count may be reduced
       to discard empty sections in the ID range between the ID of the last
       non empty section and the last section ID.

       This operation cannot be applied if the skeleton has already been
       culled for the first time, otherwise the results are undefined.
    */
    void hardClip(const osg::Vec4d& plane);

    unsigned int getMaximumVisibleBranchOrder() const
    {
        return _maxVisibleOrder;
    }

    void setMaximumVisibleBranchOrder(const unsigned int order);

    /**
       Returns the branch order of each section.
     */
    const uint8_t* getBranchOrders() const { return _branchOrders.get(); }
    /**
       Returns the total number of sections from this skeleton.
     */
    size_t getSectionCount() const { return _sectionCount; }
    /**
       Returns the total number of capsules of this skeleton.
    */
    size_t getSize() const { return _length; }
    /**
       Returns the array of section IDs of the section portion
       encompassed by each capsule.
     */
    const uint16_t* getSections() const { return _sections.get(); }
    /**
       Returns the number of portions each section has.
     */
    const unsigned short* getPortionCounts() const
    {
        return _portionCounts.get();
    }

#ifdef USE_CUDA
    static MasterCullCallback* createMasterCullCallback();
#endif

    /**
       Accessory function that creates a drawable geometry representing the
       skeleton capsules.
     */
    osg::Geode* createSkeletonModel(double detailRatio = 0.3,
                                    bool wireframe = true) const;

    size_t getEstimatedHostDataSizeForCUDACull() const;

    /**
       Without including visibility array
     */
    size_t getDeviceSkeletonDataSize() const;

    size_t getEstimatedVisibilityStateSize() const
    {
        return (/* state */
                sizeof(char) +
                /* visibility bits */
                sizeof(unsigned int) * _sectionCount);
    }

protected:
    /*--- Protected declarations ---*/

    class CullCallbackImpl;

    /*--- Protected member attributes ---*/

    static const bool s_useSharedMemory;

    size_t _length = 0;
    size_t _sectionCount = 0;

    /* These data are only needed by the device */
    boost::shared_array<osg::Vec3f> _starts;
    boost::shared_array<osg::Vec3f> _ends;
    boost::shared_array<float> _widths;
    boost::shared_array<uint8_t> _portions;

    boost::shared_array<uint16_t> _sections;
    boost::shared_array<unsigned short> _portionCounts;

#ifdef USE_CUDA
    size_t _blockCount = 0;
    size_t _sectionStartsLength = 0;

    /* Depending on the cull kernel implementation used either
       (_firstBlockSection, _perBlockSectionsStarts and _accumSectionsPerBlock)
       or _sections will be copied to GPU memory. */
    boost::shared_array<uint16_t> _firstBlockSection;
    boost::shared_array<uint16_t> _perBlockSectionsStarts;
    /* The 2 most significant bits of each element indicate of the first and
       last sections of that block are fully contained within the block
       or not. */
    boost::shared_array<uint16_t> _accumSectionsPerBlock;
#endif

    /* Data that is needed during processing of the morphology in the host */
    boost::shared_array<float> _startPositions;
    boost::shared_array<float> _endPositions;

    boost::shared_array<uint8_t> _branchOrders;

    /*--- Protected constructors/destructor ---*/

    Skeleton();

    /**
       All array data is shared between skeletons except for the clipping
       and final visibility info.
     */
    Skeleton(const Skeleton& skeleton);

    /*--- Protected member functions ---*/

    void allocHostArrays(const Portions* portions);

#ifdef USE_CUDA
    /**
       uploads the skeleton data to the device of the current
       CUDAContext.

       Since the device arrays are shared between common skeletons, this
       means that uploading only needs to be done once per skeleton.
       The device array is deallocated when the last skeleton referencing the
       data is disposed.
       If no CUDA Context is current the behaviour is undefined.
     */
    void uploadToCurrentDevice();
#endif

private:
/*--- Private declarations ---*/

#ifdef USE_CUDA
    class MasterCullCallbackImpl;
    class TrivialMasterCullCallback;

    class PerCameraAndEyeState;
    using PerCameraAndEyeStates = AutoExpandLFVector<PerCameraAndEyeState>;
#else
    unsigned int _version = 0;
#endif
    struct SoftClippingInfo;

/*--- Private member attributes */

#ifdef USE_CUDA
    /*
      Per camera/eye cull state.
      Thread-safety is guaranteed as long as each camera/eye pair is accessed
      by a single thread at a time.
    */
    PerCameraAndEyeStates _state;

    /* A single array per device with all the data above.
       Device IDs must be taken from CUDAContext objects. */
    using DeviceArrays = AutoExpandLFVector<boost::shared_ptr<uint8_t>>;
    /* The device array vector is shared between all (shallow) copies of
       this skeleton. */
    std::shared_ptr<DeviceArrays> _perDeviceArray;
    size_t _deviceArraySize = 0;

    /* Only used if the skeleton is creating its own stream */
    cudaStream_t _stream = 0;
#endif

    std::shared_ptr<SoftClippingInfo> _clippingInfo;
    unsigned int _maxVisibleOrder = std::numeric_limits<unsigned int>::max();

#if defined USE_CUDA && !defined NDEBUG
    LB_TS_VAR(ThreadID);
    AutoExpandLFVector<ThreadIDStruct> _deviceThreads;
#endif

/*--- Private member functions ---*/
#ifdef USE_CUDA
    void _cull(const osg::Matrixf& modelview, const cuda::CullFrustum& frustum,
               PerCameraAndEyeState& state);

    /**
       Since skeletons are immutable, this function uploads the data to the
       device only if needed. */
    void _allocVisibilities(PerCameraAndEyeState& state);
#endif

    template <typename Operation>
    void _applyCapsuleMaskingOperation(const uint16_ts& sections,
                                       const floats& starts, const floats& ends,
                                       const Operation& operation);

    void _dirty();
};
}
}
}
#endif
