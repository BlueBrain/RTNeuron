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

#include "../helpers.h"
#include "docstrings.h"
#include "rtneuron/net/RestInterface.h"
#include "rtneuron/types.h"

using namespace boost::python;
using namespace bbp::rtneuron;

typedef std::shared_ptr<net::RestInterface> RestInterfacePtr;

/* Boost.Python is doing something strange with the shared ptr if we pass it
   by const&. */
RestInterfacePtr initRestInterface(list args, ViewPtr* view)
{
    std::vector<std::string> strings;
    const int argc = len(args);
    char** argv = (char**)alloca((argc + 1) * sizeof(char*));

    boost::python::stl_input_iterator<std::string> end;
    for (boost::python::stl_input_iterator<std::string> i(args); i != end; ++i)
    {
        strings.push_back(*i);
        argv[strings.size() - 1] = &strings.back()[0];
    }
    /* argv must be null terminated */
    argv[argc] = 0;

    return RestInterfacePtr(new net::RestInterface(argc, argv, *view));
}

// export_RestInterface --------------------------------------------------------

void export_RestInterface()
// clang-format off
{
class_<net::RestInterface, RestInterfacePtr, boost::noncopyable>(
    "RestInterface", DOXY_CLASS(bbp::rtneuron::net::RestInterface), no_init)
    .def("__init__",
         make_constructor(initRestInterface, default_call_policies(),
                          (arg("args"), arg("view"))),
         DOXY_FN(bbp::rtneuron::net::RestInterface::RestInterface));
}
