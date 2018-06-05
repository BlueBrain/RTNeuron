## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

import sys
import os
from PyQt5 import QtCore, QtWidgets, QtQuick

import rtneuron as _rtneuron

from .baseGUI import BaseGUI
from .dialogs import *
from .selectionHandler import SelectionHandler
from .simulationPlayer import SimulationPlayer
from . import dialogs

# Monsteer is imported last because dialogs imports nest and with this
# order it's possible to shut up nest import.
try:
    from monsteer.Nesteer import Simulator
    _steering_available = True
except ImportError as e:
    print(e)
    print("Warning: Could not load monsteer module. "
          "Simulation steering capabilities will not be available")
    _steering_available = False

class GUI(BaseGUI):
    """
    The GUI class to be used in applications. Sets up all the necessary pieces.
    """
    spike_tail_value = QtCore.pyqtSignal('float')

    def __init__(self, *args, **kwargs):
        super(GUI, self).__init__(
            os.path.dirname(os.path.realpath(__file__)) + '/Overlay.qml',
            *args, **kwargs)

        _rtneuron.thegui = self

    def _init_implementation(self):

        qml = self._overlay.rootObject()
        qml.toggleFullscreen.connect(self._toggle_fullscreen)

        # Creating the overlay dialogs
        self._snapshot_dialog = SnapshotDialog(self._overlay)
        self._snapshot_dialog.visible = False

        if _steering_available:
            self._inject_stimuli_dialog = InjectStimuliDialog(self._overlay)
            self._inject_stimuli_dialog.visible = False

        self.spike_tail_value.connect(qml.onSpikeTailSliderValue)

        self._player = SimulationPlayer(
            qml.findChild(QtQuick.QQuickItem, "player",
                          QtCore.Qt.FindDirectChildrenOnly))

        self._exit_signal.connect(self.close)

    def _connect_engine_implementation(self, engine):
        # connect generic attribute_changed signal
        view = engine.views[0]
        self._connect_view(view)

        qml = self._overlay.rootObject()

        def on_take_snapshot_toggled(checked):
            self._snapshot_dialog.visible = checked

        self._snapshot_dialog.done.connect(qml.onSnapshotDone)
        qml.takeSnapshotToggled.connect(on_take_snapshot_toggled)

        # Steering signals
        if _steering_available:
            self._connect_steering_signals()
            qml.setSteeringAvailable(True)

        # Spike tail signals
        def on_spike_tail_dialog_requested(val):
            spike_tail, ok = QtWidgets.QInputDialog.getDouble(
                self, 'Simulation Parameters',
                'Spike tail (ms):', value=val, min=0.0)

            if spike_tail > 0.0 and ok:
               view.attributes.spike_tail = spike_tail

        def on_spike_tail_value_changed(val):
            view.attributes.spike_tail = val

        qml.spikeTailInputDialogRequested.connect(
            on_spike_tail_dialog_requested)
        qml.spikeTailSliderChanged.connect(on_spike_tail_value_changed)

        self._player.engine_player = engine.player
        # Enabling the the spike delay slider if needed
        if engine.player.endTime == engine.player.endTime:
            qml.setPlayerEnabled(True)
        self._player.view = view
        self._player.open_simulation_clicked.connect(self._on_open_simulation)

    def _connect_steering_signals(self):
        """ Connect GUI signals to the button to enable steering and the dialog
        for stimulus injection."""

        qml = self._overlay.rootObject()

        def on_enable_steering_clicked():

            steering_dest, ok = QtWidgets.QInputDialog.getText(
                self, 'Steering', 'Publisher:', text = "nest://")

            if steering_dest and ok:
                try:
                    self._simulator = Simulator(str(steering_dest))
                    qml = self._overlay.rootObject()
                    qml.setSteeringEnabled(True)
                    self._inject_stimuli_dialog.simulator = simulator
                except Exception as e:
                    print("Error while enabling steering channel: " +
                          str(e))

        def on_simulator_state_changed(state):
            if(self._simulator):
                if state:
                    self._simulator.play()
                else:
                    self._simulator.pause()

        def on_inject_stimuli_done():
            self._inject_stimuli_dialog.visible = False

        def on_inject_stimuli_toggled(checked):
            self._inject_stimuli_dialog.visible = checked

        qml.enableSteeringClicked.connect(on_enable_steering_clicked)
        qml.changeSimulatorState.connect(on_simulator_state_changed)
        qml.injectStimuliToggled.connect(on_inject_stimuli_toggled)
        self._inject_stimuli_dialog.done.connect(on_inject_stimuli_done)
        self._inject_stimuli_dialog.done.connect(qml.onInjectStimuliDone)


    def _connect_view(self, view):
        """ Connect GUI signals to view/scene modifications """

        def on_view_attribute_changed(name):
            try:
                if name == "spike_tail":
                    self.spike_tail_value.emit(view.attributes.spike_tail)
            except:
                print("Unhandled attribute " + name)

        def on_attr_changed(category, name, value):
            """ Slot called for attribute changed in GUI """
            if category == 'scene':
                scene = view.scene
                if name == 'transparency':
                    if value:
                        _rtneuron.sceneops.enable_transparency(scene)
                    else:
                        _rtneuron.sceneops.disable_transparency(scene)
                else:
                    scene.attributes.__setattr__(str(name), value)
            elif category == 'view':
                view.attributes.__setattr__(str(name), value)

        self._selection_handler = SelectionHandler(view, self._background)

        self._snapshot_dialog.view = view
        if _steering_available:
            self._inject_stimuli_dialog.view = view

        view.attributes.attributeChanged.connect(on_view_attribute_changed)
        self.spike_tail_value.emit(view.attributes.spike_tail)

        qml = self._overlay.rootObject()
        qml.attrChanged.connect(on_attr_changed)

    def _on_open_simulation(self):

        dialog = OpenSimulationDialog(
            self._overlay, _rtneuron.simulation, self._player.view.scene)

        def disconnect_dialog():
            # Breaking the ref loop between the dialog and this function, which
            # is what keeps the object alive
            dialog.done.disconnect()
            dialog.cancelled.disconnect()

        def on_dialog_done(simulation_applied):
            disconnect_dialog()
            self._player.enable_playback_controls(simulation_applied)
            # Enabling also the spike delay slider
            self._overlay.rootObject().setPlayerEnabled(simulation_applied)

        dialog.done.connect(on_dialog_done)
        dialog.cancelled.connect(disconnect_dialog)

    def _toggle_fullscreen(self):
        """ Slot connected to toggle fullscreen action in UI; switch between
            fullscreen and normal window size
        """
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
