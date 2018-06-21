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

from PyQt5 import QtCore as _Qt

from .QMLBaseDialog import *

class AcceptDialog(object):

    class Dialog(QMLBaseDialog) :
        def __init__(self, parent):
            super(AcceptDialog.Dialog, self).__init__(
                parent, "dialogs/AcceptDialog.qml", False)

    def __init__(self, parent):
        self._parent = parent
        self._done = False
        self._dialog = self.Dialog(self._parent)

    def set_button_labels(self, cancel, accept):
        """Sets the labels for the cancel and accept buttons. An empty
        label makes the button invisible"""
        if cancel == "" and accept == "":
            raise BadArgument("At least one of the labels must be non-empty")
        self._dialog.dialog.setCancelButtonLabel(cancel)
        self._dialog.dialog.setAcceptButtonLabel(accept)

    def set_message(self, message):
        self._dialog.dialog.setMessage(message)

    def run(self):
        import threading
        self._dialog.dialog.done.connect(self._on_done)
        self._dialog.show()

        loop = _Qt.QEventLoop()
        while not self._done:
            loop.processEvents()
        self._dialog.close()

        return self._result

    def _on_done(self, result):
        self._result = result
        self._done = True
