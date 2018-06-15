# -*- coding: utf8 -*-
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
## contact: juan.hernando@epfl.ch

from .QMLBaseDialog import *

from PyQt5 import QtCore

import rtneuron as _rtneuron
import brain as _brain

class OpenSimulationDialog(QMLBaseDialog):
    """A pseudo-modal dialog that pops up a rectangle to select a compartment
    and spike report"""

    done = QtCore.pyqtSignal(bool)
    """ Signal emitted when the dialog is succesfully closed or cancelled. The
    parameter is True if any simulation has been applied to the scene and
    false otherwise"""

    cancelled = QtCore.pyqtSignal()

    class ReportListModel(QtCore.QAbstractListModel):

        def __init__(self, simulation, parent=None):
            QtCore.QAbstractListModel.__init__(self, parent)
            self._roles = {QtCore.Qt.UserRole + 0: b'name'}
            self.names = ["None"]
            for report in simulation.compartment_report_names():
                self.names.append(report)

        def roleNames(self):
            return self._roles

        def rowCount(self, parent=QtCore.QModelIndex()):
            return len(self.names)

        # This property is needed to query the number of rows from QML to be
        # able to choose the height of the displaying list. The function
        # rowCount can't be used because if decorated as a property
        # quasi-silent# exception occurs somewhere and the code doesn't work.
        @QtCore.pyqtProperty(int)
        def rows(self):
            return len(self.names)

        def data(self, index, role):
            return QtCore.QVariant(self.names[index.row()])

    def __init__(self, parent, simulation, scene):

        super(OpenSimulationDialog, self).__init__(
            parent, "dialogs/OpenSimulationDialog.qml")

        self.dialog.done.connect(self._on_done)
        self.dialog.cancelled.connect(self._on_cancelled)

        self._report_list = self.ReportListModel(simulation)
        self.dialog.setReportListModel(self._report_list)

        self._simulation = simulation
        self._scene = scene

    def _close(self):
        self.dialog.done.disconnect()
        self.dialog.cancelled.disconnect()

        # If something was typed in the loader dialog the base overlay has
        # lost the focus. Returning the focus because otherwise key presses
        # won't be captured.
        self.parent().rootObject().setProperty("focus", True)

        self.close()

    def _on_cancelled(self):
        self.cancelled.emit()
        self._close()

    def _on_done(self, compartment_report_index, spike_file):

        # Removing any previous simulation
        self._scene.clearSimulation()
        new_simulation = False

        if compartment_report_index != 0:
            report = self._report_list.names[compartment_report_index]
            try:
                _rtneuron.apply_compartment_report(
                    self._simulation, self._scene, report)
            except RuntimeError as error:
                self.dialog.invalidReport(report, str(error))
                return
            new_simulation = True

        if spike_file != "":

            # To avoid unicode errors in Python 2.7
            spike_file = str(spike_file)

            try:
                if spike_file == "/default/":
                    spike_file = "default"
                    report = self._simulation.open_spike_report()
                else:
                    report = _brain.SpikeReportReader(spike_file)
            except RuntimeError as error:
                self.dialog.invalidReport("default", str(error))
                return
            self._scene.setSimulation(report)
            new_simulation = True

        self.done.emit(new_simulation)
        self._close()
