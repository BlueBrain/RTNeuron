/* Copyright (c) 2006-2018, Ecole Polytechnique Federale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politecnica de Madrid (UPM)
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

#include <boost/python.hpp>

#include "boost_signal_connect_wrapper.h"
#include "docstrings.h"

#include "rtneuron/AttributeMap.h"
#include "rtneuron/detail/attributeMapTypeRegistration.h"

#include <boost/python/extract.hpp>
#include <boost/python/stl_iterator.hpp>

#include <iostream>

using namespace boost::python;
using namespace bbp::rtneuron;

namespace
{
lunchbox::Any convertToParameter(const object& o)
{
    if (o.ptr() == Py_None)
    {
        /* None is treated separately because a null AttributeMapPtr can
           be extracted from it. */
        PyErr_SetString(PyExc_ValueError,
                        "None is not a valid attribute type for AttributeMap");
        throw error_already_set();
    }

    extract<AttributeMapPtr> mapExtractor(o);
    if (mapExtractor.check())
        return lunchbox::Any((AttributeMapPtr)mapExtractor);
#if PY_VERSION_HEX < 0x03000000
    if (PyString_Check(o.ptr()))
#else
    if (PyUnicode_Check(o.ptr()))
#endif
        return lunchbox::Any((std::string)extract<std::string>(o));
    if (o == true)
        return lunchbox::Any(true);
    if (o == false)
        return lunchbox::Any(false);
    if (PyFloat_Check(o.ptr()))
        return lunchbox::Any((double)extract<double>(o));
    if (PyIndex_Check(o.ptr()))
        return lunchbox::Any((int)extract<int>(o));

    std::string typeName(o.ptr()->ob_type->tp_name);
    AttributeMap::TypeRegistrationPtr registration =
        AttributeMap::getTypeRegistration(typeName);
    if (!registration)
    {
        PyErr_SetString(PyExc_ValueError,
                        ("Type " + typeName + " is not a valid attribute"
                                              " type for AttributeMap")
                            .c_str());
        throw error_already_set();
    }
    return registration->getFromPython(o.ptr());
}
}

void AttributeMap_setattr(AttributeMap& map, str name, object value)
{
    AttributeMap::Parameters params;
    bool scalar = (extract<std::string>(value).check() ||
                   !PyObject_HasAttrString(value.ptr(), "__len__"));

    if (scalar)
        params.push_back(convertToParameter(value));
    else
        for (stl_input_iterator<object> i(value), end; i != end; ++i)
            params.push_back(convertToParameter(*i));

    if (params.empty())
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Cannot assign an empty list of parameters to an attribute");
        throw_error_already_set();
    }

    const std::string cname = extract<std::string>(name);

    ReleaseGIL release;
    map.set(cname, params);
}

object AttributeMap_getattr(AttributeMap& map, str name)
{
    if (name == "__name__")
        /* Needed for interactive help because __getattr__ has been
           overriden. */
        return str("AttributeMap");

    try
    {
        list l;
        const AttributeMap::Parameters& params =
            map.getParameters(extract<std::string>(name));
        for (AttributeMap::Parameters::const_iterator j = params.begin();
             j != params.end(); ++j)
        {
            if (j->type() == typeid(bool))
                l.append(lunchbox::any_cast<bool>(*j));
            else if (j->type() == typeid(int))
                l.append(lunchbox::any_cast<int>(*j));
            else if (j->type() == typeid(unsigned int))
                l.append(lunchbox::any_cast<unsigned int>(*j));
            else if (j->type() == typeid(float))
                l.append(lunchbox::any_cast<float>(*j));
            else if (j->type() == typeid(double))
                l.append(lunchbox::any_cast<double>(*j));
            else if (j->type() == typeid(std::string))
                l.append(lunchbox::any_cast<const std::string&>(*j));
            else if (j->type() == typeid(AttributeMapPtr))
                l.append(lunchbox::any_cast<const AttributeMapPtr&>(*j));
            else
            {
                AttributeMap::TypeRegistrationPtr registration =
                    AttributeMap::getTypeRegistration(j->type());
                if (!registration)
                {
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Unknown attribute type for "
                                    "\"to python\" conversion");
                    throw error_already_set();
                }
                PyObject* pyobject =
                    static_cast<PyObject*>(registration->convertToPython(*j));
                l.append(handle<>(pyobject));
            }
        }
        if (len(l) == 1)
            return l[0];
        else
            return l;
    }
    catch (const std::exception&)
    {
        PyErr_SetString(PyExc_AttributeError,
                        ("Attribute " +
                         (std::string)extract<std::string>(name) + " not found")
                            .c_str());
        throw error_already_set();
    }
}

AttributeMapPtr AttributeMap_initFromDict(dict d)
{
    AttributeMapPtr map(new AttributeMap());
    list items = d.items();
    for (int i = 0; i < len(items); ++i)
    {
        tuple t = extract<tuple>(items[i]);
        AttributeMap_setattr(*map, extract<str>(t[0]), t[1]);
    }
    return map;
}

object AttributeMap_dir(const AttributeMap& map)
{
    list attrNames;
    for (AttributeMap::const_iterator i = map.begin(); i != map.end(); ++i)
        attrNames.append(i->first);
    return attrNames;
}

std::string AttributeMap_str(const AttributeMap& map)
{
    std::stringstream str;
    str << map;
    return str.str();
}

AttributeMapPtr AttributeMap_copy(const AttributeMap& map)
{
    AttributeMapPtr copy(new AttributeMap());
    for (AttributeMap::const_iterator i = map.begin(); i != map.end(); ++i)
        copy->set(i->first, i->second);
    return copy;
}

AttributeMapPtr AttributeMap_deepcopy(const AttributeMap& map, dict)
{
    /* The dictionary is not used to keep the implementation simple.
       A strict implementation would require a modification of the
       merge method using only public methods. */
    return AttributeMapPtr(new AttributeMap(map));
}

void AttributeMap_doc(const AttributeMap& attributeMap)
{
    object pydoc = import("pydoc");
    object pager = pydoc.attr("pager");

    if (!attributeMap.getExtraDocstring().empty())
        pager("AttributeMap instance documentation:\n\n" +
              attributeMap.getExtraDocstring());
}

// export_AttributeMap ---------------------------------------------------------

void export_AttributeMap()
// clang-format off
{

class_<AttributeMap, AttributeMapPtr, boost::noncopyable> attributeMapWrapper
("AttributeMap", DOXY_CLASS(bbp::rtneuron::AttributeMap), init<>());

scope attributeMapScope = attributeMapWrapper;

class_<AttributeMap::AttributeChangedSignal, boost::noncopyable>
("__AttributeChangedSignal__", no_init)
    .def("connect", signal_connector<
             AttributeMap::AttributeChangedSignature>::connect)
    .def("disconnect", signal_connector<
             AttributeMap::AttributeChangedSignature>::disconnect)
        ;

attributeMapWrapper
    .def("__init__", make_constructor(AttributeMap_initFromDict),
         "Create an AttributeMap from a dictionary")
    .def("__setattr__", AttributeMap_setattr)
    .def("__getattr__", AttributeMap_getattr)
    .def("__dir__", AttributeMap_dir)
    .def("__str__", AttributeMap_str)
    .def("__copy__", AttributeMap_copy)
    .def("__deepcopy__", AttributeMap_deepcopy)
    .def_readonly("attributeChanged", &AttributeMap::attributeChanged,
         DOXY_VAR(bbp::rtneuron::AttributeMap::attributeChanged))
    .def("help", AttributeMap_doc,
         "Print extra documentation of an AttributeMap instance if available")

;
}
