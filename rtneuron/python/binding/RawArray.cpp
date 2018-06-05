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

#include "RawArray.h"

#include <Python.h>
#include <boost/cstdint.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace
{
/* This is needed to import the PyArray_API symbol. Otherwise an undefined
   symbol error occurs at library loading time. */
class Init
{
public:
    Init()
    {
        if (!_import())
            boost::python::throw_error_already_set();
    }

private:
    bool _import()
    {
        import_array1(false);
        return true;
    }
} _init;
}

namespace bbp
{
namespace rtneuron
{
namespace
{
template <typename T, typename U>
void _copyArrayData(RawArray<T>& out, const boost::python::object& in)
{
    using namespace boost::python;
    auto* pyarray = (PyArrayObject*)in.ptr();

    size_t size = 1;
    const size_t ndim = PyArray_NDIM(pyarray);
    const npy_intp* shape = PyArray_SHAPE(pyarray);
    out.shape.clear();
    out.shape.reserve(ndim);
    for (size_t i = 0; i != ndim; ++i)
    {
        out.shape.push_back(shape[i]);
        size *= shape[i];
    }

    out.array.reset(new T[size]);

    const U* data = static_cast<U*>(PyArray_DATA(pyarray));

    if (ndim == 1)
    {
        for (size_t i = 0; i != size; ++i)
            out.array[i] = static_cast<T>(data[i]);
    }
    else if (ndim == 2)
    {
        const size_t n = shape[0];
        const size_t m = shape[1];
        auto flags = PyArray_FLAGS(pyarray);
        if (flags & NPY_ARRAY_F_CONTIGUOUS)
        {
            for (size_t j = 0; j != m; ++j)
            {
                for (size_t i = 0; i != n; ++i)
                    out.array[i * m + j] = static_cast<T>(data[j * n + i]);
            }
        }
        else if (flags & NPY_ARRAY_C_CONTIGUOUS)
        {
            for (size_t i = 0; i != n; ++i)
            {
                for (size_t j = 0; j != m; ++j)
                    out.array[i * m + j] = static_cast<T>(data[i * m + j]);
            }
        }
        else
        {
            PyErr_SetString(PyExc_ValueError,
                            "Unsupported array layout "
                            "(should be C or Fortran contiguous)");
            throw_error_already_set();
        }
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Unsupported shape dimension");
        throw_error_already_set();
    }
}

template <typename T>
RawArray<T> _extractRawArrayFromPyArray(const boost::python::object& array)
{
    using namespace boost::python;

    RawArray<T> out;

    object type = array.attr("dtype");
    bool error = false;
    size_t bytes = extract<size_t>(type.attr("itemsize"));
    char kind = extract<char>(type.attr("kind"));
    using namespace boost;
    switch (kind)
    {
    case 'f':
    {
        switch (bytes)
        {
        case 4:
            _copyArrayData<T, float>(out, array);
            break;
        case 8:
            _copyArrayData<T, double>(out, array);
            break;
        default:
            error = true;
        }
        break;
    }
    case 'i':
    {
        switch (bytes)
        {
        case 1:
            _copyArrayData<T, boost::int8_t>(out, array);
            break;
        case 2:
            _copyArrayData<T, boost::int16_t>(out, array);
            break;
        case 4:
            _copyArrayData<T, boost::int32_t>(out, array);
            break;
        case 8:
            _copyArrayData<T, boost::int64_t>(out, array);
            break;
        default:
            error = true;
        }
        break;
    }
    case 'u':
    {
        switch (bytes)
        {
        case 1:
            _copyArrayData<T, boost::uint8_t>(out, array);
            break;
        case 2:
            _copyArrayData<T, boost::uint16_t>(out, array);
            break;
        case 4:
            _copyArrayData<T, boost::uint32_t>(out, array);
            break;
        case 8:
            _copyArrayData<T, boost::uint64_t>(out, array);
            break;
        default:
            error = true;
        }
        break;
    }
    default:
        error = true;
    }

    if (error)
    {
        std::stringstream msg;
        msg << "Unsupported data type in array access. Don't know how to "
            << "convert " << kind << bytes << " into " << typeid(T).name()
            << sizeof(T);
        PyErr_SetString(PyExc_ValueError, msg.str().c_str());
        throw_error_already_set();
    }

    return out;
}

template <typename T>
RawArray<T> _extractRawArrayFromIterators(const boost::python::object& o)
{
    using namespace boost::python;

    size_t size = 0;
    std::vector<size_t> shape;
    shape.reserve(2);

    bool flatList = true;
    bool listOfLists = true;
    shape.push_back(len(o));
    for (int i = 0; i != len(o); ++i)
    {
        if (flatList)
            flatList = extract<T>(o[i]).check();
        if (listOfLists)
        {
            try
            {
                list l = extract<list>(o[i]);

                size_t length = len(l);
                if (shape.size() == 1)
                    shape.push_back(length);
                else if (shape[1] != length)
                {
                    PyErr_SetString(PyExc_ValueError,
                                    "Cannot extract flat nd-array from"
                                    " nested lists of irregular length");
                    throw_error_already_set();
                }
                size += length;
            }
            catch (...)
            {
                PyErr_Clear();
                listOfLists = false;
            }
        }
    }

    switch (shape.size())
    {
    case 1:
    {
        RawArray<T> out(new T[size], std::move(shape));
        stl_input_iterator<T> i(o), end;
        for (size_t index = 0; i != end; ++i, ++index)
            out.array[index] = *i;
        return out;
    }
    case 2:
    {
        RawArray<T> out(new T[size], std::move(shape));
        size_t index = 0;
        for (int i = 0; i != len(o); ++i)
        {
            list l = extract<list>(o[i]);
            stl_input_iterator<T> j(l), end;
            for (; j != end; ++j, ++index)
                out.array[index] = *j;
        }
        return out;
    }
    default:
    {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot extract flat array from object");
        throw_error_already_set();
        return RawArray<T>(); /* To suppress an spurious warning. */
    }
    }
}
}

template <typename T>
RawArray<T>::RawArray(const boost::python::object& o)
{
    if (o.ptr() == Py_None)
        return;

    using namespace boost::python;
    RawArray<T> out;

    if (PyArray_Check(o.ptr()))
    {
        /* Template function partial specialization is not allowed */
        out = _extractRawArrayFromPyArray<T>(o);
    }
    else
    {
        out = _extractRawArrayFromIterators<T>(o);
    }
    array = std::move(out.array);
    shape = std::move(out.shape);
}

template RawArray<float>::RawArray(const boost::python::object& o);
template RawArray<unsigned int>::RawArray(const boost::python::object& o);
}
}
