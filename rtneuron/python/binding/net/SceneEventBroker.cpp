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

#include <boost/python.hpp>

#include "../boost_signal_connect_wrapper.h"
#include "../helpers.h"
#include "docstrings.h"

#include <rtneuron/net/SceneEventBroker.h>

#include <boost/signals2/signal.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

namespace
{
typedef std::shared_ptr<net::SceneEventBroker> SceneEventBrokerPtr;

SceneEventBrokerPtr initSceneEventBroker1()
{
    return SceneEventBrokerPtr(new net::SceneEventBroker());
}

SceneEventBrokerPtr initSceneEventBroker2(const std::string& session)
{
    return SceneEventBrokerPtr(new net::SceneEventBroker(session));
}

SceneEventBrokerPtr initSceneEventBroker3(const std::string& publisher,
                                          const std::string& subscriber)
{
    return SceneEventBrokerPtr(
        new net::SceneEventBroker(zeroeq::URI(publisher),
                                  zeroeq::URI(subscriber)));
}

void SceneEventBroker_sendToggleRequest(net::SceneEventBroker& broker,
                                        object gids)
{
    const auto gidset = brain_python::gidsFromPython(gids);
    ReleaseGIL release;
    broker.sendToggleRequest(gidset);
}
}

// export_SceneEventBroker ----------------------------------------------------

void export_SceneEventBroker()
{
    /* Since ZeroEQ doesn't have any Python wrapping we have to add the wrapping
       of this type somewhere */
    enum_<lexis::data::CellSetBinaryOpType> binaryOp(
        "CellSetBinaryOpType",
        "Binary operation between to cell sets. This is the "
        "enum type originally defined lexis/data/cellSetBinaryOp.h");
    binaryOp.value("SYNAPTIC_PROJECTIONS",
                   lexis::data::CellSetBinaryOpType::Projections);

    class_<net::SceneEventBroker, std::shared_ptr<net::SceneEventBroker>,
           boost::noncopyable>
        sceneEventBrokerWrapper("SceneEventBroker",
                                DOXY_CLASS(
                                    bbp::rtneuron::net::SceneEventBroker),
                                no_init);

    scope sceneEventBrokerScope = sceneEventBrokerWrapper;

    /* Nested classes */

    WRAP_MEMBER_SIGNAL(net::SceneEventBroker, CellSetSelectedSignal)
    WRAP_MEMBER_SIGNAL(net::SceneEventBroker, CellSetBinaryOpSignal)

    /*
      Note: Spaces before & in DOXY_FN are absolutely needed, otherwise breathe
      is not able to find the functions in the doxygen XML. Lines can't be split
      either.
    */
    sceneEventBrokerWrapper
        .def("__init__",
             make_constructor(initSceneEventBroker1, default_call_policies()),
             DOXY_FN(bbp::rtneuron::net::SceneEventBroker::SceneEventBroker()))

        .def("__init__",
             make_constructor(initSceneEventBroker2, default_call_policies(),
                              (arg("session"))),
             /* Don't wrap this line of change whitespace */
             DOXY_FN(bbp::rtneuron::net::SceneEventBroker::SceneEventBroker(
                 const std::string&)))

        .def("__init__",
             make_constructor(initSceneEventBroker3, default_call_policies(),
                              (arg("publisher"), arg("subscriber"))),
             /* Don't wrap this line of change whitespace */
             DOXY_FN(bbp::rtneuron::net::SceneEventBroker::SceneEventBroker(
                 const zeroeq::URI&, const zeroeq::URI&)))

        .add_property("trackState", &net::SceneEventBroker::getTrackState,
                      &net::SceneEventBroker::setTrackState,
                      DOXY_FN(
                          bbp::rtneuron::net::SceneEventBroker::setTrackState))
        .def("trackScene", &net::SceneEventBroker::trackScene,
             DOXY_FN(bbp::rtneuron::net::SceneEventBroker::trackScene))
        .def("sendToggleRequest", SceneEventBroker_sendToggleRequest,
             DOXY_FN(bbp::rtneuron::net::SceneEventBroker::sendToggleRequest))
        .def_readonly("cellsSelected", &net::SceneEventBroker::cellsSelected,
                      DOXY_VAR(
                          bbp::rtneuron::net::SceneEventBroker::cellsSelected))
        .def_readonly(
            "cellSetBinaryOp", &net::SceneEventBroker::cellSetBinaryOp,
            DOXY_VAR(bbp::rtneuron::net::SceneEventBroker::cellSetBinaryOp));
}
