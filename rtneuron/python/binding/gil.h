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

#ifndef RTNEURON_PYTHON_GIL_H
#define RTNEURON_PYTHON_GIL_H

#include <boost/python.hpp>

namespace bbp
{
namespace rtneuron
{
/* Collection of helper classes for GIL handling using RAII */

class ReleaseGIL
{
public:
    ReleaseGIL() { _state = PyEval_SaveThread(); }
    ~ReleaseGIL() { PyEval_RestoreThread(_state); }
private:
    PyThreadState* _state;
};

class EnsureGIL
{
public:
    EnsureGIL() { _state = PyGILState_Ensure(); }
    ~EnsureGIL() { PyGILState_Release(_state); }
protected:
    PyGILState_STATE _state;
};
}
}
#endif
