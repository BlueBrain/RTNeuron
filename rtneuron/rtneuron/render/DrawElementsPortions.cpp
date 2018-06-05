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

#define GL_GLEXT_PROTOTYPES
#include <osg/GL> /* Must go before GL/glext.h */

#include "timing.h"

#include "DrawElementsPortions.h"
#include "config/Globals.h"
#include "util/cameraToID.h"

#include <osg/State>
#include <osg/Stats>
#include <osg/ValueObject>
#include <osg/Version>
#ifdef CULL_TIMING
#include <osgUtil/SceneView>
#include <osgViewer/Renderer>
#endif
#include <boost/timer.hpp>

#include <GL/glu.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Data types
*/

class DrawElementsPortions::CullData
{
    /*
      Declarations
    */
public:
    typedef osg::buffered_value<GLuint> GLObjectList;

    struct Ranges
    {
        bool valid() const { return counts.get() != 0; }
        uint32_tsPtr counts;
        uint32_tsPtr starts;
    };

    /*
      Per camera and eye visibility metadata of a skeleton
    */
    struct VisibilityData
    {
        VisibilityData(const Skeleton& skeleton, const unsigned int cameraEyeID)
            : version(skeleton.getVisibilityVersion(cameraEyeID))
            , state(skeleton.getCullingState(cameraEyeID))
            , maxVisibleOrder(skeleton.getMaximumVisibleBranchOrder())
        {
        }

        const unsigned int version;
        const Skeleton::CullingState state;
        const unsigned int maxVisibleOrder;
    };

    /*
       Constructors
    */
public:
    CullData()
        : _visible(true)
        , _fullyVisible(true)
        , _version(0)
        , _maxVisibleOrder(std::numeric_limits<unsigned int>::max())
    {
    }

    /*
      Member functions
    */
public:
    bool fastVisibilityCheck(const Skeleton* skeleton)
    {
        /* This is the case for detailed soma meshes. */
        if (skeleton == 0)
        {
            _fullyVisible = true;
            return true;
        }
        if (!skeleton->culledOnce())
        {
            /* Since we don't know, we assume the skeleton is visible */
            _fullyVisible = true;
            _visible = true;
            return true;
        }
        return false;
    }

    /**
       Returns true if the pritimive ranges are up to date.
       The version stamp of the cull data is also updated.
    */
    bool reusePrimitiveRanges(const VisibilityData& data);

    /**
       Allocates the Ranges arrays if needed and clears them.
    */
    void initRanges(size_t reserveSize = 0);

    /**
       Computes the primtive ranges for the given skeleton and visibility masks
     */
    void computeCountsAndStarts(const uint32_t* visibilities,
                                const Skeleton& skeleton,
                                const DrawElementsPortions& portions);

    /*
       Member attributes
    */
public:
    GLObjectList _vboList;

    Ranges _ranges;

    bool _visible;
    bool _fullyVisible;

    unsigned int _version;
    boost::shared_array<uint32_t> _visibility;
    unsigned int _maxVisibleOrder;
};

void DrawElementsPortions::CullData::initRanges(size_t reserveSize)
{
    if (!_ranges.valid())
    {
        _ranges.counts.reset(new uint32_ts());
        _ranges.starts.reset(new uint32_ts());
        if (reserveSize != 0)
        {
            _ranges.counts->reserve(reserveSize);
            _ranges.starts->reserve(reserveSize);
        }
    }
    else
    {
        _ranges.counts->clear();
        _ranges.starts->clear();
    }
}

bool DrawElementsPortions::CullData::reusePrimitiveRanges(
    const VisibilityData& data)
{
    assert(data.version != 0 || data.state == Skeleton::FULLY_VISIBLE);
    if (_maxVisibleOrder == data.maxVisibleOrder && _version == data.version &&
        (data.version != 0 || data.state != Skeleton::FULLY_VISIBLE))
    {
        assert(!_visible || _ranges.valid());
        return true;
    }
    return false;
}

void DrawElementsPortions::CullData::computeCountsAndStarts(
    const uint32_t* visibilities, const Skeleton& skeleton,
    const DrawElementsPortions& portions)
{
    _maxVisibleOrder = skeleton.getMaximumVisibleBranchOrder();
    initRanges(skeleton.getSize());

    /* Updating intervals */
    unsigned int portionIndex = 0;
    const uint32_t* clipMasks = skeleton.getClipMasks();
    const uint16_t* sections = skeleton.getSections();
    const uint8_t* branchOrders = skeleton.getBranchOrders();

    assert(skeleton.getSize() == portions._portionRanges.length);

    for (size_t i = 0; i < skeleton.getSectionCount(); ++i)
    {
        if (skeleton.getPortionCounts()[i] == 0)
            /* This can happen only after the skeleton has been hard
               clipped. */
            continue;

        assert(portionIndex < skeleton.getSize());
        if (branchOrders[sections[portionIndex]] > _maxVisibleOrder)
        {
            portionIndex += skeleton.getPortionCounts()[i];
            continue;
        }
        uint32_t mask;
        if (visibilities)
            mask = visibilities[i];
        else
            mask = skeleton.getPortionCounts()[i] == MAX_CAPSULES_PER_SECTION
                       ? 0xffffffffu
                       : (1u << skeleton.getPortionCounts()[i]) - 1u;

        assert(visibilities == 0 ||
               mask <=
                   (skeleton.getPortionCounts()[i] == MAX_CAPSULES_PER_SECTION
                        ? 0xffffffffu
                        : (1u << skeleton.getPortionCounts()[i]) - 1u));
        if (clipMasks)
            mask &= ~clipMasks[i];
        unsigned int firstIndex = portionIndex;
        for (; mask != 0; ++portionIndex, mask >>= 1)
        {
            assert(portionIndex < skeleton.getSize());
            if (mask & 1)
            {
                const std::pair<uint32_t, uint32_t>& range =
                    portions._portionRanges.ranges[portionIndex];
                if (range.first > range.second)
                    continue;

                if (_ranges.counts->size() != 0)
                {
                    uint32_t start = _ranges.starts->back();
                    uint32_t count = _ranges.counts->back();
                    if (start + count >= range.first)
                    {
                        assert(range.second > start);
                        _ranges.counts->back() =
                            std::max(count, range.second - start + 1);
                    }
                    else
                    {
                        if (portions._mode == GL_TRIANGLE_STRIP)
                        {
                            uint32_t end = start + count - 3;
                            /* Removing degenerated triangles from
                               previous strip */
                            const IndexArray& indices = portions._indices;
                            for (; (indices[end] == indices[end + 1] ||
                                    indices[end + 1] == indices[end + 2] ||
                                    indices[end] == indices[end + 2]);
                                 --end)
                                assert(end != 0);
                            _ranges.counts->back() = end - start + 3;
                        }
                        /* Pushing the new interval */
                        _ranges.starts->push_back(range.first);
                        assert(range.second > range.first);
                        _ranges.counts->push_back(range.second - range.first +
                                                  1);
                    }
                }
                else
                {
                    _ranges.starts->push_back(range.first);
                    assert(range.second > range.first);
                    _ranges.counts->push_back(range.second - range.first + 1);
                }
            }
        }
        portionIndex = firstIndex + skeleton.getPortionCounts()[i];
    }

#ifdef PRINT_CULLED_RANGES
    for (size_t i = 0; i < _ranges.starts->size(); ++i)
    {
        std::cout << '(' << (*_ranges.starts)[i] << ','
                  << (*_ranges.counts)[i] + (*_ranges.starts)[i] << ')';
    }
    std::cout << std::endl;
#endif
}

/*
  Helper functions
*/
inline void shared_array_non_deleter(const void*)
{
}

/*
  Constructors
*/
DrawElementsPortions::DrawElementsPortions()
{
}

DrawElementsPortions::DrawElementsPortions(osg::DrawElementsUInt* elements)
    : osg::DrawElements(*elements)
    , _elements(elements)
    , _count(elements->getNumIndices())
    , _indices(&*elements->begin(), shared_array_non_deleter)
{
}

DrawElementsPortions::DrawElementsPortions(GLenum mode,
                                           const IndexArray& indices,
                                           unsigned int count)
    : _count(count)
    , _indices(indices)
{
    _mode = mode;
    _primitiveType = osg::PrimitiveSet::PrimitiveType;
}

DrawElementsPortions::DrawElementsPortions(const DrawElementsPortions& prim,
                                           osg::CopyOp copyOp)
    : osg::DrawElements(prim, copyOp)
    , _eboOwner(prim._eboOwner)
    , _count(prim._count)
    , _indices(prim._indices)
    , _skeleton(prim._skeleton)
    , _portionRanges(prim._portionRanges)
{
}

/*
  Member functions
*/

void DrawElementsPortions::draw(osg::State& state,
                                bool useVertexBufferObjects) const
{
    GLuint* indices = _indices.get();
    assert(!_eboOwner.valid() || _eboOwner->_indices == _indices);

    if (!indices)
        return;

    const CullData& cd = _applyVisibility(state);

    if (!cd._visible)
        return;

    /* Code taken from OSG */
    if (useVertexBufferObjects)
    {
        const DrawElementsPortions* owner =
            _eboOwner.valid() ? _eboOwner.get() : this;
        osg::GLBufferObject* ebo =
            owner->getOrCreateGLBufferObject(state.getContextID());
        assert(ebo);
        state.bindElementBufferObject(ebo);
        _drawPortions(cd, (GLuint*)ebo->getOffset(owner->getBufferIndex()));
    }
    else
    {
        _drawPortions(cd, indices);
    }
#ifdef CULL_TIMING
    if (collectStats)
    {
        /* No race condictions here because these stats should be accesed only
           by one thread. */
        double& dispatchTime =
            stats->getAttributeMap(frameNumber)["Draw_dispatch_CPU_time"];
        dispatchTime += timer.tick() / 1000.0;
    }
#endif
}

void DrawElementsPortions::accept(osg::PrimitiveFunctor& functor) const
{
    if (_indices.get() != 0)
    {
        functor.drawElements(_mode, _count, _indices.get());
    }
}

void DrawElementsPortions::accept(osg::PrimitiveIndexFunctor& functor) const
{
    if (_indices.get() != 0)
    {
        functor.drawElements(_mode, _count, _indices.get());
    }
}

void DrawElementsPortions::_drawPortions(const CullData& cullData,
                                         const GLuint* indices) const
{
    if (!Globals::areGLDrawCallsEnabled())
        return;

    if (cullData._fullyVisible)
    {
        glDrawElements(_mode, _count, GL_UNSIGNED_INT, indices);
    }
    else if (cullData._visible)
    {
        assert(cullData._ranges.valid());
        for (size_t i = 0; i < cullData._ranges.starts->size(); ++i)
        {
            assert((*cullData._ranges.counts)[i] !=
                   std::numeric_limits<unsigned int>::max());
            glDrawElements(_mode, (*cullData._ranges.counts)[i],
                           GL_UNSIGNED_INT,
                           indices + (*cullData._ranges.starts)[i]);
        }
    }
}

const DrawElementsPortions::CullData& DrawElementsPortions::_applyVisibility(
    osg::State& state) const
{
    unsigned int cameraEyeID = 0;
    /* There is no other way to get the current camera from here */
    if (!state.getGraphicsContext()->getUserValue("cam_eye_id", cameraEyeID))
    {
        osg::Camera* camera = state.getGraphicsContext()->getCameras().front();
        cameraEyeID = getCameraAndEyeID(camera);
    }

    CullDataPtr& cullDataPtr = _cullData[cameraEyeID];
    if (!cullDataPtr)
        cullDataPtr.reset(new CullData());
    CullData& cd = *cullDataPtr;

    if (cd.fastVisibilityCheck(_skeleton.get()))
        return cd;

    const uint32_t* visibilities;

    CullData::VisibilityData visibilityData(*_skeleton, cameraEyeID);

    if (visibilityData.state == Skeleton::NOT_VISIBLE)
    {
        cd._fullyVisible = false;
        cd._visible = false;
        return cd;
    }

    if (cd.reusePrimitiveRanges(visibilityData))
        /* The state of cd._fullyVisible and cd._visible is also reused */
        return cd;

    if (visibilityData.state == Skeleton::VISIBLE)
    {
        cd._visible = false;
        cd._fullyVisible = false;

        visibilities = _skeleton->getVisibilities(cameraEyeID);
        for (unsigned int i = 0; i < _skeleton->getSectionCount(); ++i)
            cd._visible |= visibilities[i] != 0;
    }
    else /* visibilityData.state == Skeleton::FULLY_VISIBLE */
    {
        cd._fullyVisible = (visibilityData.maxVisibleOrder ==
                                std::numeric_limits<unsigned int>::max() &&
                            _skeleton->getClipMasks() == 0);
        cd._visible = true; /* This assumes that masking doesn't clip the
                               full skeleton. */

        if (cd._fullyVisible)
            /* Fully visible without masks applied */
            return cd;

        cd._visible = true;
        /* Fully visible with masks */
        visibilities = 0;
    }

    if (!cd._visible)
        return cd;

    cd.computeCountsAndStarts(visibilities, *_skeleton, *this);

    cd._version = visibilityData.version;

    return cd;
}
}
}
}
