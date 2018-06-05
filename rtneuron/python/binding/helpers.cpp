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

#include <vmmlib/vector.hpp> // Don't move after helper.h

#include "helpers.h"

#include <vmmlib/matrix.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/stl_iterator.hpp>

#include <numpy/_numpyconfig.h>
#if NPY_API_VERSION >= 0x00000007
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

using namespace boost::python;

namespace bbp
{
namespace rtneuron
{
namespace
{
#if PY_VERSION_HEX >= 0x03000000
void* _importArray()
{
    import_array();
    return PyArray_API;
}
#else
void _importArray()
{
    import_array();
}
#endif
}

void importArray()
{
    /* This function is in this file to keep the array import and the usage of
       the C-API all in the same compilation unit. Otherwise some magic is
       needed before including the numpy headers in all files. */
    _importArray();
}

Vector2f extract_Vector2(const object& o)
{
    extract<Vector2f> extractor(o);
    if (extractor.check())
    {
        return (Vector2f)extractor;
    }
    else if (PyObject_HasAttrString(o.ptr(), "__len__") && len(o) == 2)
    {
        stl_input_iterator<float> i(o);
        Vector2f v;
        v.x() = *(i++);
        v.y() = *(i++);
        return v;
    }
    else
    {
        std::string typeName(o.ptr()->ob_type->tp_name);
        std::stringstream msg;
        msg << "Implicit conversion from " + typeName;
        if (PyObject_HasAttrString(o.ptr(), "__len__"))
            msg << " of length " << len(o);
        msg << " to Vector2f failed";
        PyErr_SetString(PyExc_ValueError, msg.str().c_str());
        throw error_already_set();
    }
}

Vector3f extract_Vector3(const object& o)
{
    extract<Vector3f> extractor(o);
    if (extractor.check())
    {
        return (Vector3f)extractor;
    }
    else if (PyObject_HasAttrString(o.ptr(), "__len__") && len(o) == 3)
    {
        stl_input_iterator<float> i(o);
        Vector3f v;
        v.x() = *(i++);
        v.y() = *(i++);
        v.z() = *(i++);
        return v;
    }
    else
    {
        std::string typeName(o.ptr()->ob_type->tp_name);
        std::stringstream msg;
        msg << "Implicit conversion from " + typeName;
        if (PyObject_HasAttrString(o.ptr(), "__len__"))
            msg << " of length " << len(o);
        msg << " to Vector3f failed";
        PyErr_SetString(PyExc_ValueError, msg.str().c_str());
        throw error_already_set();
    }
}

Vector4f extract_Vector4(const object& o)
{
    extract<Vector4f> extractor(o);
    if (extractor.check())
    {
        return (Vector4f)extractor;
    }
    else if (PyObject_HasAttrString(o.ptr(), "__len__") && len(o) == 4)
    {
        stl_input_iterator<float> i(o);
        Vector4f v;
        v.x() = *(i++);
        v.y() = *(i++);
        v.z() = *(i++);
        v.w() = *(i++);
        return v;
    }
    else
    {
        std::string typeName(o.ptr()->ob_type->tp_name);
        std::stringstream msg;
        msg << "Implicit conversion from " + typeName;
        if (PyObject_HasAttrString(o.ptr(), "__len__"))
            msg << " of length " << len(o);
        msg << " to Vector4f failed";
        PyErr_SetString(PyExc_ValueError, msg.str().c_str());
        throw error_already_set();
    }
}

Orientation extract_Orientation(const object& o)
{
    extract<Orientation> extractor(o);
    std::stringstream msg;
    if (extractor.check())
    {
        return (Orientation)extractor;
    }
    else if (PyObject_HasAttrString(o.ptr(), "__len__") && len(o) == 2)
    {
        stl_input_iterator<object> i(o);
        Orientation orientation;
        try
        {
            orientation.set_sub_vector<3, 0>(extract_Vector3(*i++));
        }
        catch (error_already_set&)
        {
            std::string typeName(i->ptr()->ob_type->tp_name);
            msg << "Conversion from " + typeName;
            if (PyObject_HasAttrString(o.ptr(), "__len__"))
                msg << " of length " << len(o);
            msg << " to Orientation axis failed";
        }
        try
        {
            orientation.w() = extract<float>(*i);
            return orientation;
        }
        catch (error_already_set&)
        {
            std::string typeName(i->ptr()->ob_type->tp_name);
            msg << "Conversion from " + typeName
                << " to Orientation angle failed";
        }
    }
    else
    {
        std::string typeName(o.ptr()->ob_type->tp_name);
        msg << "Implicit conversion from " + typeName;
        if (PyObject_HasAttrString(o.ptr(), "__len__"))
            msg << " of length " << len(o);
        msg << " to Orientation failed";
    }
    /* This point is reached if the extaction wasn't successful */
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    throw error_already_set();
}
}
}
