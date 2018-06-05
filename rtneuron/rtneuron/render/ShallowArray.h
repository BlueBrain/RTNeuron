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

#ifndef RTNEURON_SHALLOWARRAY_H
#define RTNEURON_SHALLOWARRAY_H

#include <lunchbox/debug.h>

#include <osg/Array>
#include <osg/Version>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   \brief This class is an osg::Array adaptor to wrap plain T[] pointers.

   The purpose of this class is to provide an osg::Array interface for
   non-OSG arrays without having to perform any copy.
 */
template <typename T, osg::Array::Type ARRAYTYPE, int DataSize, int DataType>
class ShallowArray : public osg::Array
{
public:
    /* Public constructors/destructor */

    ShallowArray()
        : osg::Array(ARRAYTYPE, DataSize, DataType)
        , _array(0)
        , _length(0)
    {
    }

    ShallowArray(const ShallowArray& array,
                 const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY)
        : osg::Array(array, copyop)
        , _array(array._array)
        , _length(array._length)
    {
    }

    ShallowArray(T* array, unsigned int length)
        : osg::Array(ARRAYTYPE, DataSize, DataType)
        , _array(array)
        , _length(length)
    {
    }

    ShallowArray& operator=(const ShallowArray& array)
    {
        if (this == &array)
            return *this;
        assign(array.begin(), array.end());

        return *this;
    }

    virtual ~ShallowArray() {}
    /* Public Member functions */
    virtual osg::Object* cloneType() const { return new ShallowArray(); }
    virtual osg::Object* clone(const osg::CopyOp& copyop) const
    {
        return new ShallowArray(*this, copyop);
    }

    T& operator[](unsigned int i) { return _array[i]; }
    const T& operator[](unsigned int i) const { return _array[i]; }
    inline virtual void accept(osg::ArrayVisitor& av);
    inline virtual void accept(osg::ConstArrayVisitor& av) const;
    inline virtual void accept(unsigned int index, osg::ValueVisitor& vv);
    inline virtual void accept(unsigned int index,
                               osg::ConstValueVisitor& vv) const;

    virtual int compare(unsigned int lhs, unsigned int rhs) const
    {
        const T& elem_lhs = (*this)[lhs];
        const T& elem_rhs = (*this)[rhs];
        if (elem_lhs < elem_rhs)
            return -1;
        if (elem_rhs < elem_lhs)
            return 1;
        return 0;
    }

    virtual unsigned int getElementSize() const { return sizeof(T); }
    virtual const GLvoid* getDataPointer() const
    {
        if (_length)
            return _array;
        else
            return 0;
    }

    virtual unsigned int getTotalDataSize() const
    {
        return _length * sizeof(T);
    }

    virtual unsigned int getNumElements() const { return _length; }
#if OSG_VERSION_GREATER_OR_EQUAL(3, 2, 0)
    virtual void reserveArray(unsigned int) { LBUNIMPLEMENTED; }
    virtual void resizeArray(unsigned int) { LBUNIMPLEMENTED; }
#endif

    /* Member attibutes */
protected:
    T* _array;
    unsigned int _length;
};

template <typename T, osg::Array::Type ARRAYTYPE, int DataSize, int DataType>
inline void ShallowArray<T, ARRAYTYPE, DataSize, DataType>::accept(
    osg::ArrayVisitor& av)
{
    av.apply(*this);
}

template <typename T, osg::Array::Type ARRAYTYPE, int DataSize, int DataType>
inline void ShallowArray<T, ARRAYTYPE, DataSize, DataType>::accept(
    osg::ConstArrayVisitor& av) const
{
    av.apply(*this);
}

template <typename T, osg::Array::Type ARRAYTYPE, int DataSize, int DataType>
inline void ShallowArray<T, ARRAYTYPE, DataSize, DataType>::accept(
    unsigned int index, osg::ValueVisitor& vv)
{
    vv.apply((*this)[index]);
}

template <typename T, osg::Array::Type ARRAYTYPE, int DataSize, int DataType>
inline void ShallowArray<T, ARRAYTYPE, DataSize, DataType>::accept(
    unsigned int index, osg::ConstValueVisitor& vv) const
{
    vv.apply((*this)[index]);
}

typedef ShallowArray<osg::Vec3, osg::Array::Vec3ArrayType, 3, GL_FLOAT>
    Vec3ShallowArray;
typedef ShallowArray<unsigned, osg::Array::UIntArrayType, 1, GL_UNSIGNED_INT>
    UIntShallowArray;
}
}
}
#endif
