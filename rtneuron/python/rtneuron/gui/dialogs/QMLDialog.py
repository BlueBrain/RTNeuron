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
import os

from .QMLBaseDialog import QMLBaseDialog

from PyQt5 import QtCore, QtQml

class QMLDialog(QMLBaseDialog):
    """
    A show/hide re-usable dialog class that can be instantiated using a qml
    file.

    The dialog will added to the QML document of the parent and anchored
    to the bottom right corner by default.

    QML components loaded by this class must inherit from the Dialog component.
    """
    done = QtCore.pyqtSignal()

    def __init__(self, parent, qml_file):
        super(QMLDialog, self).__init__(parent, qml_file)

        self._visible = False

        self.dialog.cancel.connect(self._done)

    @QtCore.pyqtProperty(bool)
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        self.dialog.setVisible(value)
        self.dialog.setEnabled(value)

    def _done(self):
        """Emits the done signal and makes the dialog invisible. To be used
           by derived classes when the dialog has to be closed visually."""
        self.done.emit()
        self.visible = False
