# -*- coding: utf-8 -*-
## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Juan Hernando <juan.hernando@epfl.ch>
##
## This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
##
## This library is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License version 3.0 as published
## by the Free Software Foundation.
##
## This library is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along
## with this library; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
from PyQt5 import QtCore, QtWidgets

import rtneuron as _rtneuron

from .QMLComponent import QMLComponent

class SimulationPlayer(QMLComponent):

    simulation_time_changed = QtCore.pyqtSignal(float, float, float)
    simulation_speed_changed = QtCore.pyqtSignal(float)
    open_simulation_clicked = QtCore.pyqtSignal()

    # internal
    _reset_timestamp = QtCore.pyqtSignal()

    def __init__(self, qml=None, parent=None, view=None, simulation=None):
        super(SimulationPlayer, self).__init__(
                parent, 'SimulatorPlayer.qml' if qml == None else qml)
        self._playing = False
        self._looping = False
        self._engine_player = None

        self.simulation_time_changed.connect(self.qml.onSimulationTimeChanged)
        self.simulation_speed_changed.connect(self.qml.onPlaybackSpeedChanged)
        self.qml.openSimulationClicked.connect(self.open_simulation_clicked)
        self._view = view

        self._reset_timestamp.connect(self._on_reset_timestamp)

    def enable_playback_controls(self, enable):
        if self._view and self._view.scene:
            self._view.attributes.display_simulation = enable

        self.qml.enablePlaybackControls(enable)

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, view):
        self._view = view
        self.qml.enableOpenButton()

    @property
    def engine_player(self):
        return self._engine_player

    @engine_player.setter
    def engine_player(self, player):
        self._engine_player = player

        """ Connect GUI signals to player actions and engine signals to GUI
        updates.
        Connects the start/stop button click signal and the slider changed
        signal from the GUI and the timestampChanged signal from the player """

        # Connecting QML signals
        def on_play_pause_clicked():
            if self._playing:
                player.pause()
            else:
                player.play()
            self._playing = not self._playing

        def on_loop_button_toggled(checked):
            self._looping = checked

        def on_time_changed(value):
            if player.endTime != player.endTime:
                # Simulation window still not adjusted
                try:
                    player.adjustWindow()
                except RuntimeError:
                    return
            value = min(1, max(value, 0))
            player.timestamp = \
                (player.endTime - player.beginTime) * value + player.beginTime

        def on_speed_changed(val):
            player.simulationDelta = val

        def on_speed_dialog_requested(val):
            simulation_delta, ok = QtWidgets.QInputDialog.getDouble(
                None, 'Simulation Parameters',
                'Simulation delta (ms):', value=val)

            if simulation_delta > 0.0 and ok:
                player.simulationDelta = simulation_delta

        # Connecting slots
        self.qml.playPauseClicked.connect(on_play_pause_clicked)
        self.qml.loopButtonToggled.connect(on_loop_button_toggled)
        self.qml.timeSliderChanged.connect(on_time_changed)
        self.qml.speedInputDialogRequested.connect(on_speed_dialog_requested)
        self.qml.speedSliderChanged.connect(on_speed_changed)

        # Connecting player signals
        def on_timestamp_changed(value):
            begin = player.beginTime
            end = player.endTime
            self.simulation_time_changed.emit(value, begin, end)

        def on_playback_state_changed(state):
            if state == _rtneuron.SimulationPlayer.PLAYING:
                self.qml.setPlaybackState(True)
            elif state == _rtneuron.SimulationPlayer.PAUSED:
                self.qml.setPlaybackState(False)
            elif self._looping and state == _rtneuron.SimulationPlayer.FINISHED:
                # We don't reset from here because setting the timestamp causes
                # a state transition that triggers a Qt warning.
                self._reset_timestamp.emit()

        def on_simulation_delta_value_changed(value):
            self.simulation_speed_changed.emit(value)

        def on_simulation_window_changed(begin, end):
            self.simulation_time_changed.emit(player.timestamp, begin, end)

        player.timestampChanged.connect(on_timestamp_changed)
        player.simulationDeltaChanged.connect(on_simulation_delta_value_changed)
        player.windowChanged.connect(on_simulation_window_changed)
        player.playbackStateChanged.connect(on_playback_state_changed)

        # Try adjusting the simulation window. This will throw if there's
        # no report attached
        if player.endTime != player.endTime:
            try:
                player.adjustWindow()
                # If this succeeds we need to enable the controls
                self.qml.enablePlaybackControls(True)
            except RuntimeError:
                pass

        self.simulation_speed_changed.emit(player.simulationDelta)


    def _on_reset_timestamp(self):
        player = self._engine_player
        player.timestamp = player.window[player.simulationDelta < 0]
