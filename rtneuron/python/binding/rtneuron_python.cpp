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

#include "helpers.h"

#include <boost/python.hpp>

#include <numpy/_numpyconfig.h>
#if NPY_API_VERSION >= 0x00000007
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

void export_AttributeMap();
void export_Camera();
void export_ColorMap();
void export_CameraManipulators();
void export_Pointer();
void export_RTNeuron();
void export_Scene();
void export_SimulationPlayer();
void export_View();
void export_Types();

void export_rtneuron_data();
void export_rtneuron_sceneops();
void export_rtneuron_net();

namespace
{
void _importModule(const char* module)
{
    /* In Python3 an exception raised from module initialization with
       PyError_SetString gets reported with the cryptic message:
       "SystemError: initialization of _rtneuron raised unreported exception"
       I don't find any way to properly raise an exception from here, the
       documentation is not specific about this and I don't find any solution
       in the Internet. I'm not going to read CPython or Boost.Python code,
       so the import exceptions will be printed and then discarded. */
    try
    {
        boost::python::import(module);
    }
    catch (boost::python::error_already_set&)
    {
        PyErr_Print();
    }
}
}

BOOST_PYTHON_MODULE(_rtneuron)
{
    boost::python::docstring_options doc_options(true, true, false);

    /* Initializing the GIL.
       The interpreter doesn't create it by default so we must ensure that it
       exists after this module is loaded.
       The GIL is acquired when asynchonously calling from C++ to python. */
    PyEval_InitThreads();

    bbp::rtneuron::importArray();

    boost::python::object package = boost::python::scope();
    package.attr("__path__") = "_rtneuron";
    package.attr("__package__") = "rtneuron";

    export_AttributeMap();
    export_Camera();
    export_ColorMap();
    export_Scene();
    export_RTNeuron();
    export_SimulationPlayer();
    export_View();
    export_Pointer();
    export_Types();
    export_CameraManipulators();

    export_rtneuron_sceneops();

#ifdef RTNEURON_USE_ZEROEQ
    export_rtneuron_net();
#endif
}
