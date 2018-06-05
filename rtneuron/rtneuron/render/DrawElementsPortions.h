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

#ifndef RTNEURON_DRAWELEMENTSPORTIONS_H
#define RTNEURON_DRAWELEMENTSPORTIONS_H

#include "Skeleton.h"
#include "util/AutoExpandLFVector.h"

#include <osg/PrimitiveSet>
#include <osg/Version>

#include <lunchbox/debug.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class DrawElementsPortions : public osg::DrawElements
{
public:
    /*--- Public declarations ---*/

    typedef boost::shared_array<GLuint> IndexArray;

    /*--- Public constructors/destructors ---*/

    DrawElementsPortions();
    DrawElementsPortions(osg::DrawElementsUInt* elements);
    DrawElementsPortions(GLenum mode, const IndexArray& indices,
                         unsigned int count);
    DrawElementsPortions(const DrawElementsPortions& prim, osg::CopyOp copyOp);

    /*--- Public member functions ---*/

    META_Object(osg, DrawElementsPortions);

    void set(GLenum mode, const IndexArray& indices, unsigned int count)
    {
        _mode = mode;
        _count = count;
        _indices = indices;
    }

    virtual const GLvoid* getDataPointer() const { return _indices.get(); }
    virtual GLenum getDataType() { return GL_UNSIGNED_INT; }
    virtual unsigned int getTotalDataSize() const
    {
        /* NOTE: Not very sure about this result. */
        return 4 * _count;
    }

    virtual bool supportsBufferObject() const { return false; }
    virtual void setEBOOwner(DrawElementsPortions* owner) { _eboOwner = owner; }
    virtual osg::DrawElements* getDrawElements()
    {
        return _eboOwner.valid() ? _eboOwner.get() : this;
    }
    virtual const DrawElements* getDrawElements() const
    {
        return _eboOwner.valid() ? _eboOwner.get() : this;
    }

    virtual void draw(osg::State& state, bool useVertexBufferObjects) const;

    virtual void accept(osg::PrimitiveFunctor& functor) const;

    virtual void accept(osg::PrimitiveIndexFunctor& functor) const;

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
    virtual void resizeElements(unsigned int) { LBUNIMPLEMENTED; }
#endif

    virtual void reserveElements(unsigned int) { LBUNIMPLEMENTED; }
    virtual void setElement(unsigned int i, unsigned int v) { _indices[i] = v; }
    virtual unsigned int getElement(unsigned int i) { return _indices[i]; }
    virtual void addElement(unsigned int) { LBUNIMPLEMENTED; }
    virtual unsigned int index(unsigned int pos) const { return _indices[pos]; }
    virtual unsigned int getNumIndices() const { return _count; }
    /**
       Sets the skeleton to be tested for branch level culling.
     */
    void setSkeleton(const SkeletonPtr& skeleton) { _skeleton = skeleton; }
    void setPortionRanges(const Skeleton::PortionRanges& ranges)
    {
        _portionRanges = ranges;
    }

    void setIndexArray(IndexArray& indices) { _indices = indices; }
    IndexArray& getIndexArray() { return _indices; }
    virtual void offsetIndices(int) { abort(); }
private:
    /*--- Private declarations ---*/

    /* Cull cache for cull domains. */
    class CullData;

    /*--- Private member attributes ---*/

    /* Helper attribute provided to nest a DrawElements inside and avoid the
       need of copying the indices array */
    osg::ref_ptr<osg::DrawElements> _elements;

    osg::ref_ptr<DrawElementsPortions> _eboOwner;
    unsigned int _count;
    IndexArray _indices;

    using CullDataPtr = std::shared_ptr<CullData>;
    using PerCameraAndEyeCullData = AutoExpandLFVector<CullDataPtr>;
    mutable PerCameraAndEyeCullData _cullData;

    SkeletonPtr _skeleton;
    Skeleton::PortionRanges _portionRanges;

    /*--- Private member functions ---*/

    void _drawPortions(const DrawElementsPortions::CullData& cullData,
                       const GLuint* indices) const;

    const CullData& _applyVisibility(osg::State& state) const;
};
}
}
}
#endif
