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

from PyQt5 import QtCore

from .QMLBaseDialog import QMLBaseDialog

__all__ = ['Progress']

class Progress(QMLBaseDialog):
    """A simple progres dialog."""

    done = QtCore.pyqtSignal(int)

    # Internal signal used to forward the progress signal from non-GUI threads
    # to the GUI thread
    _step = QtCore.pyqtSignal(str, float)

    def __init__(self, parent):
        super(Progress, self).__init__(parent, "dialogs/Progress.qml")

        self._pass = 0
        self._step.connect(self.dialog.step)

    def set_message(self, msg):

        self.dialog.setProperty("message", msg)

    def step(self, msg, size, total):

        self._step.emit(msg, size/float(total))

        if size == total:
            self._pass += 1
            self.done.emit(self._pass)
