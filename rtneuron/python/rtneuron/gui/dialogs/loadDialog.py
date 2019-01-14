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

import numpy as np
import brain as brain
from PyQt5 import QtCore
import rtneuron

from .. import util

from .QMLBaseDialog import *

__all__ = ['LoadDialog']

class LoadDialog(QMLBaseDialog):
    """
    A pseudo-modal dialog that pops up a rectangle to select a circuit and
    target.

    When accepted, the done signal contains the brain.Simulation, the
    brain.Circuit, a numpy array of ints with the GIDs of the target to load
    and the rtneuron.RepresentationMode to use. The simulation object may be
    None (this is the case in SONATA circuit configs).

    The constructor runs an event loop until the dialog is done. The done
    signal is not emitted and the dialog reamins open if the input contains
    errors.
    There's no cancel button on purpose because there's no need for it at the
    moment.
    """
    done = QtCore.pyqtSignal(object, object, object, object)

    def __init__(self, parent, default_config=None, default_target=None):

        self._default_config = default_config
        self._default_target = default_target
        """The default blue config to load if the user does not set any in
        the dialog."""

        super(LoadDialog, self).__init__(parent, "dialogs/LoadDialog.qml")

        self.dialog.done.connect(self._on_done)

    def _on_done(self, config_file, targets, display_mode):
        # To avoid unicode errors in Python 2.7
        if config_file:
            config_file = str(config_file)
        if targets:
            targets = str(targets)

        # Validate the target string and create a list of targets
        try:
            simulation = rtneuron._init_simulation(
                self._default_config if config_file == "" else config_file)
            circuit = simulation.open_circuit()
        except RuntimeError as error:
            # Try to open the config_file directly as a circuit instead
            try:
                circuit = brain.Circuit(config_file)
                simulation = None
            except RuntimeError:
                # We report the original error
                self.dialog.invalidSimulation(config_file, error.args[0])
                return

        targets = targets.lstrip().rstrip()

        if targets == "":
            if self._default_target:
                try:
                    gids = simulation.gids(self._default_target)
                except RuntimeError as e:
                    self.dialog.invalidTarget(
                        None, "Could not find a valid default target "
                        "for the simulation")
                    return
            else:
                gids = simulation.gids() if simulation else circuit.gids()
        else:
            try:
                gids = util.target_string_to_gids(targets, simulation)
            except RuntimeError as e:
                self.dialog.invalidTarget(None, *e.args)
                return

            if len(np.intersect1d(circuit.gids(), gids)) < len(gids):
                self.dialog.invalidTarget(
                    None, "Found GIDs out of circuit range")
                return

        self.done.emit(simulation, circuit, gids, display_mode)
