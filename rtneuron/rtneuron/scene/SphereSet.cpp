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

#include "SphereSet.h"

#include "config/constants.h"
#include "render/SceneStyle.h"
#include "util/vec_to_vec.h"

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osg/Version>

#include <lunchbox/lfVector.h>
#include <lunchbox/scopedMutex.h>
#include <lunchbox/spinLock.h>

#include <stdexcept>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper classes
*/
template <typename T>
struct ArrayType;

template <typename T>
struct ArraySetter
{
    ArraySetter(osg::Array* array)
        : _array(*static_cast<typename ArrayType<T>::type*>(array))
    {
    }

    static osg::Array* createArray()
    {
        return new typename ArrayType<T>::type();
    }

    void operator()(const size_t index, const T& value) const
    {
        _array[index] = value;
    }

    void erase(const size_t start, const size_t end)
    {
        _array.erase(_array.begin() + start, _array.begin() + end);
    }

    typename ArrayType<T>::type& _array;
};

/* ArrayType will be only specialized for the types actually required. It
   may be possible to do something more general, but for the few types
   that we need it's no worth the effort. */
template <>
struct ArrayType<char>
{
    typedef osg::ByteArray type;
};
template <>
struct ArrayType<float>
{
    typedef osg::FloatArray type;
};
template <>
struct ArrayType<osg::Vec3>
{
    typedef osg::Vec3Array type;
};
template <>
struct ArrayType<osg::Vec4>
{
    typedef osg::Vec4Array type;
};

class SphereSubset;
typedef lunchbox::LFVector<SphereSubset> SphereSubsets;
typedef lunchbox::LFVector<SphereSet::SubSetID> SubSetIDs;

/*
  SphereSet::SubSetID_
*/

class SphereSet::SubSetID_
{
public:
    SubSetID_()
        : _subset(0)
        , _first(0)
        , _count(0)
        , _listIndex(0)
    {
    }

    SphereSubset* _subset;
    size_t _first;
    size_t _count;
    size_t _listIndex;
};

class SphereSubset
{
public:
    osg::ref_ptr<osg::Geometry> _geometry;
    osg::ref_ptr<osg::Vec4Array> _spheres;
    osg::ref_ptr<osg::Vec4Array> _colors;
    typedef std::vector<osg::ref_ptr<osg::Array>> Attributes;
    Attributes _attributes;

    osg::ref_ptr<osg::DrawArrays> _primitive;

    /* This is required by some LFVector operations, but in practice we
       are not going to use a SphereSubset created with this constructor */
    SphereSubset() {}
    SphereSubset(const size_t reservedSize)
        : _geometry(new osg::Geometry())
        , _spheres(new osg::Vec4Array())
        , _colors(new osg::Vec4Array())
        , _primitive(new osg::DrawArrays(osg::PrimitiveSet::POINTS))
        , _temptativeCount(0)
        , _sphereCount(0)
        , _resizeMutex(new lunchbox::SpinLock())
        , _geometryMutex(new lunchbox::SpinLock())
        , _createIDMutex(new lunchbox::SpinLock())
    {
        _spheres->setDataVariance(osg::Object::DYNAMIC);
        _colors->setDataVariance(osg::Object::DYNAMIC);
        _geometry->setVertexArray(_spheres);
        _geometry->setColorArray(_colors);
        _geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        _geometry->addPrimitiveSet(_primitive);
        _geometry->setUseDisplayList(false);
        _geometry->setUseVertexBufferObjects(true);

        /* Reserving the space on advance, so it's safe to update sphere
           parameters from one thread while another thread is adding a new
           sphere (because the underlying vectors won't be rellocated. */
        _spheres->reserve(reservedSize);
        _colors->reserve(reservedSize);

        /* This trick makes it possible to call
           osg::Geometry::getVertexAttribArray and
           osg::Geometry::setVertexAttribArray concurrently from two different
           threads. */
        _geometry->getVertexAttribArrayList().reserve(MAX_VERTEX_ATTRIBUTE_NUM +
                                                      1);

        _primitive->setFirst(0);
        _primitive->setCount(0);
    }

    ~SphereSubset()
    {
        for (SubSetIDs::iterator i = _subsetIds.begin(); i != _subsetIds.end();
             ++i)
        {
            delete *i;
#ifndef NDEBUG
            *i = 0;
#endif
        }
    }

    /** Return a SphereSubset which is guaranteed to never grow larger
        than maxSpheresPerSubset once the requested number of spheres
        is added to it.

        This postcondition is guaranteed even if multiple threads call this
        function at the same time.

        The SphereSubset is pulled from the input lock-free vector and a
        new one is added if necessary.
    */
    static SphereSubset* getOrCreateGeometrySubset(
        SphereSubsets& subsets, const size_t size,
        const size_t maxSpheresPerSubset)
    {
        if (size > maxSpheresPerSubset)
        {
            /* This case is special because no SphereSubset fulfills the
               size requirement. Creating a new sphere set and requesting
               all the space right away */
            SphereSubset subset(size);
            subset._temptativeCount += size;
            /* Nobody will be able to use the subset added at the moment
               it can be accessed, but we still need the lock to ensure
               that back() returns the element just added. */
            const auto&& lock = subsets.getWriteLock();
            subsets.push_back(subset, false);
            return &subsets.back();
        }

        /* Ensuring the subset list has at least one element. */
        if (subsets.empty())
        {
            /* If LFVector provided a template parameter to take an
               allocator we could simply configure the allocator to call
               the SphereSubset constructor and then call expand. */
            const auto&& lock = subsets.getWriteLock();
            if (subsets.empty())
                subsets.push_back(SphereSubset(maxSpheresPerSubset), false);
        }

        while (true)
        {
            /* Trying with the last one. This will cause memory
               fragmentation if the subsets requested have very disparate
               sizes and are close to maxSpheresPerSubset. However that's
               not the case for spherical somas, which are added one by one. */
            const SphereSubsets::iterator subset = --subsets.end();
            /* Let's see if we can use this one */
            const size_t total = subset->_temptativeCount += size;
            if (total < maxSpheresPerSubset)
            {
                /* We can use this one. The atomic update guarantees that
                   the requested size is reserved and will prevent the subset
                   from growing larger than maxSpheresPerSubset. */
                return &*subset;
            }

            /* This subset doesn't have enough free space, we have to
               ensure that at least a new one is created and try again. */
            {
                const auto&& lock = subsets.getWriteLock();
                if (subsets.end() - subset == 1)
                    subsets.push_back(SphereSubset(maxSpheresPerSubset), false);
            }
            /* But before looping again we have to give up the requested
               space. */
            subset->_temptativeCount -= size;
        }
    }

    SphereSet::SubSetID addSphere(const osg::Vec3& position, const float radius,
                                  const osg::Vec4& color)
    {
        SphereSet::SubSetID id = _createSubSetID(1);

        {
            lunchbox::ScopedFastWrite lock(*_resizeMutex);
            /* Using push_back could lead to an inconsistency between id->_first
               and the final position. */
            _spheres->resize(std::max(_spheres->size(), id->_first + 1));
            (*_spheres)[id->_first] = osg::Vec4(position, radius);
            _colors->resize(std::max(_spheres->size(), id->_first + 1));
            (*_colors)[id->_first] = color;
        }
        /* Updating the bounding box and primitive. */
        {
            lunchbox::ScopedFastWrite lock(*_geometryMutex);
            osg::BoundingBox boundingBox = _geometry->getInitialBound();
            boundingBox.expandBy(osg::BoundingSphere(position, radius));
            _geometry->setInitialBound(boundingBox);
            _primitive->setCount(_primitive->getCount() + 1);
        }

        _spheres->dirty();
        _colors->dirty();

        return id;
    }

    SphereSet::SubSetID addSpheres(const std::vector<osg::Vec3>& positions,
                                   const float radius, const osg::Vec4& color)
    {
        SphereSet::SubSetID id = _createSubSetID(positions.size());

        {
            const size_t size = id->_first + id->_count;
            lunchbox::ScopedFastWrite lock(*_resizeMutex);
            /* First resizing, this should be fast, updating later without
               the lock */
            _colors->resize(std::max(_colors->size(), size));
            _spheres->resize(std::max(_colors->size(), size));
        }

        osg::BoundingBox boundingBox;
        for (size_t i = 0; i != id->_count; ++i)
        {
            const osg::Vec3& position = positions[i];
            const size_t index = id->_first + i;
            (*_spheres)[index] = osg::Vec4(position, radius);
            boundingBox.expandBy(osg::BoundingSphere(position, radius));
            (*_colors)[index] = color;
        }

        {
            lunchbox::ScopedFastWrite lock(*_geometryMutex);
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
            osg::BoundingBox current = _geometry->getBoundingBox();
#else
            osg::BoundingBox current = _geometry->getBound();
#endif
            current.expandBy(boundingBox);
            _geometry->setInitialBound(boundingBox);
            _primitive->setCount(_primitive->getCount() + positions.size());
        }

        _spheres->dirty();
        _colors->dirty();

        return id;
    }

    /**
       Remove an id from the subset.
       The ID becomes invalid for the caller after return.
    */
    void remove(const SphereSet::SubSetID id)
    {
        assert(id->_subset == this);

        /* Remove the index set from the OSG arrays */
        const size_t end = id->_first + id->_count;
        _spheres->erase(_spheres->begin() + id->_first,
                        _spheres->begin() + end);
        _colors->erase(_colors->begin() + id->_first, _colors->begin() + end);
        for (Attributes::iterator i = _attributes.begin();
             i != _attributes.end(); ++i)
        {
            osg::Array* array = i->get();
            switch (array->getType())
            {
            case osg::Array::ByteArrayType:
            {
                ArraySetter<char> setter(array);
                setter.erase(id->_first, end);
                break;
            }
            case osg::Array::FloatArrayType:
            {
                ArraySetter<float> setter(array);
                setter.erase(id->_first, end);
                break;
            }
            case osg::Array::Vec3ArrayType:
            {
                ArraySetter<osg::Vec3> setter(array);
                setter.erase(id->_first, end);
                break;
            }
            case osg::Array::Vec4ArrayType:
            {
                ArraySetter<osg::Vec4> setter(array);
                setter.erase(id->_first, end);
                break;
            }
            default:
                /* This can't happen because the code that could trigger it
                   won't link due to missing template instantiations. */
                abort();
            }
        }

        _primitive->setCount(_primitive->getCount() - id->_count);

        /* Adjusting the index of the first sphere for those SubSetIDs that
           need it. */
        for (size_t i = id->_listIndex; i != _subsetIds.size() - 1; ++i)
        {
            SphereSet::SubSetID next = _subsetIds[i + 1];
            next->_first -= id->_count;
            --next->_listIndex;
            _subsetIds[i] = next;
        }
        delete id;
        _subsetIds.pop_back();
    }

    bool empty() const { return _primitive->getCount() == 0; }
    template <typename T>
    osg::Array* getOrCreateAttributeArray(const unsigned int index)
    {
        osg::Array* array = _geometry->getVertexAttribArray(index);
        if (array)
            return array;

        lunchbox::ScopedFastWrite lock(*_geometryMutex);
        array = _geometry->getVertexAttribArray(index);
        if (array)
            return array;

        array = ArraySetter<T>::createArray();
        ArraySetter<T> setter(array);
        /* The array will have a reserved size that guarantees that it will
           never be rellocated. */
        setter._array.reserve(_spheres->capacity());
        setter._array.resize(_spheres->size());
        array->setDataVariance(osg::Object::DYNAMIC);
        _geometry->setVertexAttribArray(index, array);
        _geometry->setVertexAttribBinding(index,
                                          osg::Geometry::BIND_PER_VERTEX);
        _attributes.push_back(array);

        return array;
    }

    template <typename T>
    void resizeAttributeArray(osg::Array* array) const
    {
        lunchbox::ScopedFastWrite lock(*_resizeMutex);
        ArraySetter<T> setter(array);
        setter._array.resize(_spheres->size());
    }

private:
    lunchbox::Atomic<uint32_t> _temptativeCount;
    uint32_t _sphereCount;
    uint32_t _nextSubsetIdIndex;
    SubSetIDs _subsetIds;

    /* We need this class to be movable. With C++11 the pointer wouldn't
       be needed */
    std::shared_ptr<lunchbox::SpinLock> _resizeMutex;
    std::shared_ptr<lunchbox::SpinLock> _geometryMutex;
    std::shared_ptr<lunchbox::SpinLock> _createIDMutex;

    /** Return a fully initialized SubSetID for a sphere set of the
        given size. */
    SphereSet::SubSetID _createSubSetID(const size_t size)
    {
        SphereSet::SubSetID id = new SphereSet::SubSetID_();
        id->_subset = this;
        id->_count = size;

        {
            lunchbox::ScopedFastWrite lock(*_createIDMutex);
            id->_first = _sphereCount;
            _sphereCount += size;
            id->_listIndex = _subsetIds.size();
        }
        _subsetIds.expand(id->_listIndex + 1);
        _subsetIds[id->_listIndex] = id;
        return id;
    }
};

/*
  SphereSet::Impl
*/

class SphereSet::Impl
{
public:
    friend class SphereSubset;
    friend class SubSetID_;

    /*--- Public constructors/destructor ---*/
    Impl(SceneStyle::StateType styleType, const size_t maxSpheresPerSubset)
        : _maxSpheresPerSubset(maxSpheresPerSubset)
        , _geode(new osg::Geode())
        , _styleTag(styleType)
    {
    }

    ~Impl() {}
    /*--- Public member functions ---*/
    SubSetID addSphere(const osg::Vec3& position, const float radius,
                       const osg::Vec4& color)
    {
        SphereSubset* subset = _getOrCreateGeometrySubset(1);
        return subset->addSphere(position, radius, color);
    }

    SubSetID addSpheres(const std::vector<osg::Vec3>& positions,
                        const float radius, const osg::Vec4& color)
    {
        SphereSubset* subset = _getOrCreateGeometrySubset(positions.size());
        return subset->addSpheres(positions, radius, color);
    }

    void updateRadius(SubSetID id, const float radius, const bool dirty)
    {
        assert(id->_subset);
        osg::Vec4Array& spheres = *id->_subset->_spheres;
        for (size_t i = id->_first; i < id->_first + id->_count; ++i)
            spheres[i][3] = radius;
        if (dirty)
            spheres.dirty();
    }

    void updateColor(SubSetID id, const osg::Vec4& color, const bool dirty)
    {
        assert(id->_subset);
        osg::Vec4Array& colors = *id->_subset->_colors;
        for (size_t i = id->_first; i < id->_first + id->_count; ++i)
            colors[i] = color;
        if (dirty)
            colors.dirty();
    }

    template <typename T>
    void updateAttribute(SubSetID id, const unsigned int index, const T& value,
                         const bool dirty)
    {
        SphereSubset* subset = id->_subset;
        assert(subset);
        osg::Array* array = subset->getOrCreateAttributeArray<T>(index);

        ArraySetter<T> setter(array);
        assert(id->_first + id->_count <= setter._array.capacity());
        if (array->getNumElements() < subset->_spheres->getNumElements())
            subset->resizeAttributeArray<T>(array);

        for (size_t i = id->_first; i < id->_first + id->_count; ++i)
            setter(i, value);

        if (dirty)
            array->dirty();
    }

    void remove(SubSetID id)
    {
        /* This operation is not guaranteed to be thread safe on the API.
           We wil only worry about sequential correctness. */
        SphereSubset* subset = id->_subset;
        assert(subset);
        subset->remove(id);

        if (subset->empty())
        {
            /* Removing geometry object */
            for (lunchbox::LFVector<SphereSubset>::iterator i =
                     _subsets.begin();
                 i != _subsets.end(); ++i)
            {
                if (&*i == subset)
                {
                    _geode->removeDrawable(id->_subset->_geometry);
                    _subsets.erase(i);
                    break;
                }
            }
        }
    }

    void clear()
    {
        _subsets.clear();
        _geode->removeDrawables(0, _geode->getNumDrawables());
    }

    osg::Node* getNode() { return _geode; }
    void applyStyle(const SceneStylePtr& style)
    {
        _sceneStyle = style;
        _geode->setStateSet(style->getStateSet(_styleTag));
    }

    /*--- Private member variables ---*/
private:
    const size_t _maxSpheresPerSubset;

    SphereSubsets _subsets;

    lunchbox::SpinLock _geodeMutex;
    osg::ref_ptr<osg::Geode> _geode;

    SceneStylePtr _sceneStyle;
    SceneStyle::StateType _styleTag;

    /*--- Private functions ---*/

    SphereSubset* _getOrCreateGeometrySubset(size_t size)
    {
        SphereSubset* subset =
            SphereSubset::getOrCreateGeometrySubset(_subsets, size,
                                                    _maxSpheresPerSubset);
        /* Checking if the size of the subsets differs from the number
           of children in the geode. If that's the case then add the missing
           geometry objects.
           This is safe for concurrent additions, but not for additions
           and removals. */
        const size_t numSubsets = _subsets.size();
        if (_geode->getNumDrawables() < numSubsets)
        {
            lunchbox::ScopedFastWrite lock(_geodeMutex);
            /* This will iterate at least to include any new subset returned
               by this function. If a new one is added after _subsets.size()
               has been queried, the thread that added it will also run this
               code. */
            for (size_t i = _geode->getNumDrawables(); i < numSubsets; ++i)
            {
                osg::Geometry* geometry = _subsets[i]._geometry;
                /* This is the contention point when the maximum number of
                   spheres per subset is too low because containsDrawable
                   runs in linear time on the number of children drawables. */
                if (!_geode->containsDrawable(geometry))
                    _geode->addDrawable(geometry);
            }
        }

        return subset;
    }
};

/*
  Constructors/destructor
*/

SphereSet::SphereSet(const SceneStyle::StateType styleType,
                     const size_t maxSpheresPerSubset)
    : _impl(new Impl(styleType, maxSpheresPerSubset))
{
}

SphereSet::~SphereSet()
{
    delete _impl;
}

/*
  Member functions
*/
SphereSet::SubSetID SphereSet::addSphere(const osg::Vec3& position,
                                         const float radius,
                                         const osg::Vec4& color)
{
    return _impl->addSphere(position, radius, color);
}

SphereSet::SubSetID SphereSet::addSpheres(
    const std::vector<osg::Vec3>& positions, const float radius,
    const osg::Vec4& color)
{
    return _impl->addSpheres(positions, radius, color);
}

void SphereSet::updateRadius(SubSetID id, float radius, const bool dirty)
{
    _impl->updateRadius(id, radius, dirty);
}

void SphereSet::updateColor(SubSetID id, const osg::Vec4& color,
                            const bool dirty)
{
    _impl->updateColor(id, color, dirty);
}

template <typename T>
void SphereSet::updateAttribute(SubSetID id, unsigned int index, const T& value,
                                const bool dirty)
{
    _impl->updateAttribute(id, index, value, dirty);
}

void SphereSet::remove(SubSetID id)
{
    _impl->remove(id);
}

void SphereSet::clear()
{
    _impl->clear();
}

osg::Node* SphereSet::getNode()
{
    return _impl->getNode();
}

void SphereSet::applyStyle(const SceneStylePtr& style)
{
    _impl->applyStyle(style);
}

/* Specializations needed by updateAttribute */
template void SphereSet::updateAttribute<char>(SubSetID, unsigned int,
                                               const char&, bool);
template void SphereSet::updateAttribute<float>(SubSetID, unsigned int,
                                                const float&, bool);
template void SphereSet::updateAttribute<osg::Vec3>(SubSetID, unsigned int,
                                                    const osg::Vec3&, bool);
template void SphereSet::updateAttribute<osg::Vec4>(SubSetID, unsigned int,
                                                    const osg::Vec4&, bool);
}
}
}
