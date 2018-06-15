# -*- coding: utf8 -*-
## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
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

from PyQt5 import QtCore, QtQml, QtQuick, QtWidgets

from .QMLDialog import *

class SnapshotDialog(QMLDialog):
    """
    The snapshot dialog box that will be opened once the user clicks the
    'Take snapshot' button in the GUI. It will ask the user for the filename
    and dimensions for the exported image.
    """

    def __init__(self, parent):
        super(SnapshotDialog, self).__init__(parent,
                                             "dialogs/SnapshotDialog.qml")

        self.view = None

        # Connect the signals
        self.dialog.takeSnapshot.connect(self._on_take_snapshot)

    def _on_take_snapshot(self, filename, width, height):
        filename = str(filename) # Ensuring that filename is using the same
                                 # string type than the wrapping.
        filename = os.path.abspath(filename)
        name, extension = os.path.splitext(filename)
        if not extension:
            # PNG as default
            filename = filename + ".png"
        self.view.snapshot(filename, (width, height))
        self._done()

