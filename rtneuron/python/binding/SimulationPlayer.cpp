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

#include "boost_signal_connect_wrapper.h"
#include "docstrings.h"
#include "helpers.h"

#include "rtneuron/SimulationPlayer.h"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

// export_SimulationPlayer -----------------------------------------------------

void SimulationPlayer_setWindow(SimulationPlayer* player, object o)
{
    double start, stop;
    extract_pair(o, start, stop);

    /* GIL released to avoid a trivial deadlock if a callback is connected
       to the signal for window changes. */
    ReleaseGIL release;
    player->setWindow(start, stop);
}

tuple SimulationPlayer_getWindow(SimulationPlayer* player)
{
    double start, stop;
    player->getWindow(start, stop);
    return make_tuple(start, stop);
}

void SimulationPlayer_adjustWindow(SimulationPlayer* player)
{
    ReleaseGIL release;
    player->adjustWindow();
}

void SimulationPlayer_play(SimulationPlayer* player)
{
    /* GIL released to avoid a trivial deadlock if a callback is connected
       to the signal for window changes. */
    ReleaseGIL release;
    player->play();
}

void export_SimulationPlayer()
// clang-format off
{

enum_<SimulationPlayer::PlaybackState> playbackState(
    "PlaybackState",
    DOXY_ENUM(bbp::rtneuron::SimulationPlayer::PlaybackState));
playbackState.value("PLAYING", SimulationPlayer::PLAYING);
playbackState.value("PAUSED", SimulationPlayer::PAUSED);
playbackState.value("FINISHED", SimulationPlayer::FINISHED);

class_<SimulationPlayer, SimulationPlayerPtr, boost::noncopyable>
simulationPlayerWrapper(
    "SimulationPlayer", DOXY_CLASS(bbp::rtneuron::SimulationPlayer), no_init);

scope simulationPlayerScope = simulationPlayerWrapper;

scope().attr("PLAYING") = SimulationPlayer::PLAYING;
scope().attr("PAUSED") = SimulationPlayer::PAUSED;
scope().attr("FINISHED") = SimulationPlayer::FINISHED;

WRAP_MEMBER_SIGNAL(SimulationPlayer, PlaybackFinishedSignal)
WRAP_MEMBER_SIGNAL(SimulationPlayer, PlaybackStateChangedSignal)
WRAP_MEMBER_SIGNAL(SimulationPlayer, SimulationDeltaChangedSignal)
WRAP_MEMBER_SIGNAL(SimulationPlayer, WindowChangedSignal)
WRAP_MEMBER_SIGNAL(SimulationPlayer, TimestampChangedSignal)

simulationPlayerWrapper
    .add_property("timestamp",
        &SimulationPlayer::getTimestamp, &SimulationPlayer::setTimestamp,
        (std::string("Get: ") +
         DOXY_FN(bbp::rtneuron::SimulationPlayer::getTimestamp) +
         std::string("\nSet: ") +
         DOXY_FN(bbp::rtneuron::SimulationPlayer::setTimestamp)).c_str())
    .add_property("window",
        &SimulationPlayer_getWindow, &SimulationPlayer_setWindow,
        DOXY_FN(bbp::rtneuron::SimulationPlayer::setWindow))
    .add_property("beginTime",
        &SimulationPlayer::getBeginTime, &SimulationPlayer::setBeginTime,
        DOXY_FN(bbp::rtneuron::SimulationPlayer::setBeginTime))
    .add_property("endTime",
        &SimulationPlayer::getEndTime, &SimulationPlayer::setEndTime,
        DOXY_FN(bbp::rtneuron::SimulationPlayer::setEndTime))
    .add_property("simulationDelta",
        &SimulationPlayer::getSimulationDelta,
        &SimulationPlayer::setSimulationDelta,
        DOXY_FN(bbp::rtneuron::SimulationPlayer::setSimulationDelta))
    .def("adjustWindow", SimulationPlayer_adjustWindow,
         DOXY_FN(bbp::rtneuron::SimulationPlayer::adjustWindow))
    .def("play", SimulationPlayer_play,
         DOXY_FN(bbp::rtneuron::SimulationPlayer::play))
    .def("pause", &SimulationPlayer::pause,
         DOXY_FN(bbp::rtneuron::SimulationPlayer::pause))
    .def_readonly("finished", &SimulationPlayer::finished,
         DOXY_VAR(bbp::rtneuron::SimulationPlayer::finished))
    .def_readonly("playbackStateChanged", &SimulationPlayer::playbackStateChanged,
         DOXY_VAR(bbp::rtneuron::SimulationPlayer::playbackStateChanged))
    .def_readonly("simulationDeltaChanged", &SimulationPlayer::simulationDeltaChanged,
         DOXY_VAR(bbp::rtneuron::SimulationPlayer::simulationDeltaChanged))
    .def_readonly("timestampChanged", &SimulationPlayer::timestampChanged,
         DOXY_VAR(bbp::rtneuron::SimulationPlayer::timestampChanged))
    .def_readonly("windowChanged", &SimulationPlayer::windowChanged,
         DOXY_VAR(bbp::rtneuron::SimulationPlayer::windowChanged))
;

}
// clang-format on
