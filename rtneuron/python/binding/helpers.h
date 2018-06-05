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

#ifndef RTNEURON_PYTHON_HELPERS
#define RTNEURON_PYTHON_HELPERS

#include "rtneuron/types.h"

#ifdef override
#undef override
#endif
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

namespace bbp
{
namespace rtneuron
{
void importArray();

vmml::Vector2f extract_Vector2(const boost::python::object& o);
Vector3f extract_Vector3(const boost::python::object& o);
Vector4f extract_Vector4(const boost::python::object& o);
Orientation extract_Orientation(const boost::python::object& o);

template <typename T>
inline void extract_pair(const boost::python::object& o, T& a, T& b)
{
    using namespace boost::python;
    if (PyObject_HasAttrString(o.ptr(), "__len__") && len(o) == 2)
    {
        stl_input_iterator<T> i(o);
        a = *(i++);
        b = *(i++);
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Object is not pair");
        throw error_already_set();
    }
}
}
}
#endif
