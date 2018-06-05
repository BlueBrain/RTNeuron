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

#ifndef RTNEURON_API_SIMULATIONPLAYER_H
#define RTNEURON_API_SIMULATIONPLAYER_H

#include "types.h"

#include <boost/signals2/signal.hpp>

namespace bbp
{
namespace rtneuron
{
/**
   \brief Interface to simulation playback control.

   The simulation timestamp if part of the frame data, this implies that
   all views are rendered with the same timestamp.
 */
class SimulationPlayer
{
    friend class RTNeuron;

public:
    /**
       Playback state for the simulation.

       State transitions are:
       PLAYING -> PAUSED if SimulationPlayer::pause is called.
       PLAYING -> FINISHED when one of the simulation window edges is reached.

       FINISHED -> PAUSED if SimulationPlayer::pause is called.
       FINISHED -> PLAYING if setSimulationTimestamp is called with a valid
                           timestamp or setSimulationDelta is called.

       PAUSED -> PLAYING if SimulationPlayer::play is called.
     */
    enum PlaybackState
    {
        PLAYING, /*!< State change emitted when SimulationPlayer::play is
                      called and the previous state was paused or finished */
        PAUSED,  /*!< State change emitted when SimulationPlayer::pause is
                      called and the previous state was playing */
        FINISHED /*!< State change emitted when playback reaches one edge of
                      the playback window. The signal is emmited at the
                      moment the timestamp is requested, but the current
                      timestamp may be older. The signal timestampChanged
                      should be used to know exactly the timestamp of the
                      next frame to be displayed. */
    };

    /*--- Public constructors/destructor ---*/

    SimulationPlayer(const SimulationPlayer&) = delete;
    SimulationPlayer& operator=(const SimulationPlayer&) = delete;

    ~SimulationPlayer();

#define DECLARE_SIGNAL_TYPES(name, ...)              \
    typedef void name##SignalSignature(__VA_ARGS__); \
    typedef boost::signals2::signal<name##SignalSignature> name##Signal;

    DECLARE_SIGNAL_TYPES(PlaybackFinished)
    DECLARE_SIGNAL_TYPES(PlaybackStateChanged, PlaybackState)
    DECLARE_SIGNAL_TYPES(SimulationDeltaChanged, double)
    DECLARE_SIGNAL_TYPES(WindowChanged, double, double)
    DECLARE_SIGNAL_TYPES(TimestampChanged, double)

#undef DECLARE_SIGNAL_TYPES

    /*--- Public member functions ---*/

    /**
       Sets the next timestamp to display and triggers rendering.

       It may throw when trying to move the timestamp beyond the end of a
       stream-based report.
     */
    void setTimestamp(const double milliseconds);

    /**
        The timestamp being displayed currently or NaN if undefined.
     */
    double getTimestamp() const;

    /**
       \if pybind
       A (double, double) tuple with the simulation playback time window.

       If written, the timestamp to display is clamped to the new window,
       a new frame is triggered if necessary and simulation window
       auto-adjustment is turned off (to turn it on again set
       RTNeuron.attributes.auto_adjust_simulation_window to True).
       \else
       Set the simulation playback time window.

       Simulation window auto-adjustment is turned off when this method
       is invoked. Assign true to the RTNeuron attribute
       auto_adjust_simulation_window to turn it on again.

       The timestamp to display is clamped to the new window and a new frame
       is triggered if necessary.
       \endif
     */
    void setWindow(const double begin, const double end);

    /**
       Return the simulation playback time window.
     */
    void getWindow(double& begin, double& end) const;

    /**
       Adjusts the simulation playback window to the reports of the active
       scenes.

       The begin time will be equal to the minimum of the start time of all
       reports and the end time will be equal to the maximum of end time of
       all reports.

       For stream based reports, this function will try to update the end
       timestamp if the playback state is paused.

       The timestamp will be clamped to the new window and a new frame will
       be triggered if necessary.

       @throw runtime_error if there's no active scene with a report attached.
     */
    void adjustWindow();

    void setBeginTime(const double milliseconds);
    double getBeginTime() const;

    void setEndTime(const double milliseconds);
    double getEndTime() const;

    /**
       \if pybind
       The timestep between simulation frames to be used at playback.
       \else
       Set the timestep between simulation frames to be used at playback
       \endif
     */
    void setSimulationDelta(const double milliseconds);

    double getSimulationDelta() const;

    /**
       Start simulation playback from the current timestamp.
    */
    void play();

    /**
       Pause simulation playback.
    */
    void pause();

    /*--- Public signals ---*/

    /**
       @deprecated use the playbackStateChanged signal.
     */
    PlaybackFinishedSignal finished;

    /**
       Signal emitted when simulation playback state is changed.
     */
    PlaybackStateChangedSignal playbackStateChanged;

    /**
       Signal emitted when simulation delta is changed.
     */
    SimulationDeltaChangedSignal simulationDeltaChanged;

    /**
       Signal emitted when simulation window is changed.

       The signal is emitted be either calls to setWindow or by simulation
       window auto-adjustment (see RTNeuron::RTNeuron() for details).
     */
    WindowChangedSignal windowChanged;

    /**
       Signal emitted whenever a new frame with a new timestamp has finished.
    */
    TimestampChangedSignal timestampChanged;

    /** @internal */
    void disconnectAllSlots();

private:
    /*--- Private member attributes ---*/

    class _Impl;
    _Impl* _impl;

    /*--- Private constructors/destructor ---*/

    SimulationPlayer(const RTNeuronPtr& application);
};
}
}
#endif
