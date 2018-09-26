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

#include "rtneuron/ui/Pointer.h"
#if defined RTNEURON_USE_WIIUSE and defined RTNEURON_USE_VRPN
#include "rtneuron/ui/WiimotePointer.h"
#endif

#include <boost/python.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

// export_Pointer --------------------------------------------------------------

void export_Pointer()
// clang-format off
{
class_<Pointer, PointerPtr, boost::noncopyable>("Pointer", no_init);

#if defined RTNEURON_USE_WIIUSE and defined RTNEURON_USE_VRPN
class_<WiimotePointer, std::shared_ptr<WiimotePointer>, bases<Pointer>,
       boost::noncopyable>("WiimotePointer", init<std::string>());
#endif

}
