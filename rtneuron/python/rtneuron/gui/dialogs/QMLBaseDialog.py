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

from ..QMLComponent import QMLComponent

__all__ = ['QMLBaseDialog']

class QMLBaseDialog(QMLComponent):
    """
    Base class for all dialog style components.

    Dialogs are instantiated from qml files at construction.
    """

    def __init__(self, parent, qml_file, displayed=True):
        """Create a dialog from a QML file and add it to parent.

        The parent must be a QQuickItem.
        """
        super(QMLBaseDialog, self).__init__(parent, qml_file, displayed)
        self.dialog = self.qml

    def close(self):
        """Remove the dialog from the QML scene containing it."""

        # This is the best way to remove I've found so far.
        self.dialog.setVisible(False)
        self.dialog.setProperty("parent", None)
        self.setParent(None)
