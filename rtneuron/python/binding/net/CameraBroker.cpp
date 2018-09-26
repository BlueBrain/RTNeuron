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
#include "rtneuron/net/CameraBroker.h"

using namespace boost::python;
using namespace bbp::rtneuron;

typedef std::shared_ptr<net::CameraBroker> CameraBrokerPtr;

CameraBrokerPtr initBroker1(CameraPtr* camera)
{
    return CameraBrokerPtr(new net::CameraBroker(*camera));
}

CameraBrokerPtr initBroker2(const std::string& session, CameraPtr* camera)
{
    return CameraBrokerPtr(new net::CameraBroker(session, *camera));
}

CameraBrokerPtr initBroker3(const std::string& publisher,
                            const std::string& subscriber, CameraPtr* camera)
{
    return CameraBrokerPtr(new net::CameraBroker(zeroeq::URI(publisher),
                                                 zeroeq::URI(subscriber),
                                                 *camera));
}

// export_CameraBroker ---------------------------------------------------------

void export_CameraBroker()
// clang-format off
{

/*
  Note: Spaces before & in DOXY_FN are absolutely needed, otherwise breathe
  is not able to find the functions in the doxygen XML. Lines can't be split
  either.
*/
class_<net::CameraBroker, std::shared_ptr<net::CameraBroker>,
        boost::noncopyable>
("CameraBroker", DOXY_CLASS(bbp::rtneuron::net::CameraBroker), no_init)
    .def("__init__", make_constructor(
             initBroker1, default_call_policies(), (arg("camera"))),
         /* Don't wrap this line of change whitespace */
         DOXY_FN(bbp::rtneuron::net::CameraBroker::CameraBroker(const CameraPtr&)))

    .def("__init__", make_constructor(
             initBroker2, default_call_policies(),
             (arg("session"), arg("camera"))),
         /* Don't wrap this line of change whitespace */
         DOXY_FN(bbp::rtneuron::net::CameraBroker::CameraBroker(const std::string&, const CameraPtr&)))

    .def("__init__", make_constructor(
             initBroker3, default_call_policies(),
             (arg("publisher"), arg("subscriber"), arg("camera"))),
         /* Don't wrap this line of change whitespace */
         DOXY_FN(bbp::rtneuron::net::CameraBroker::CameraBroker(const zeroeq::URI&, const zeroeq::URI&, const CameraPtr&)))
;
}
