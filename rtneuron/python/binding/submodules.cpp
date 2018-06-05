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

#include "submodules.h"

namespace bbp
{
namespace rtneuron
{
boost::python::scope exportSubmodule(const std::string& name)
{
    using namespace boost::python;

    /*
       I'm not fully sure that the module stack is left in a conventional
       state after this code is run. For example the module sceneops appears
       as built-in in the interpreter, which smells fishy.
       Nonetheless, the current implementation allows to do:

       import _rtneuron
       import _rtneuron._sceneops
       from _rtneuron._sceneops import X (auto-completion of X works in IPython)

       And that's good enough.
    */
    object module(handle<>(
        borrowed(PyImport_AddModule(("rtneuron._rtneuron._" + name).c_str()))));
    scope().attr(("_" + name).c_str()) = module;
    scope moduleScope = module;
    /* Despite these paths are not completely true in the build directory,
       they ensures that _rtneuron._sceneops can be found and that
       _rtneuron.so is not loaded twice */
    moduleScope.attr("__package__") = "rtneuron._rtneuron";
    moduleScope.attr("__path__") = "rtneuron._rtneuron";

    return moduleScope;
}
}
}
