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

from PyQt5 import QtCore, QtQml, QtQuick

class QMLComponent(QtCore.QObject):
    """
    Class for creating QML components from qml files.
    """

    def __init__(self, parent, qml, show=True):
        """Take a QML component or create it from a file and add it to parent.

        The parent must be a QQuickItem.
        qml can be a file name of QQuickItem
        """
        super(QMLComponent, self).__init__()

        self.setParent(parent)

        if type(qml) == str:
            self._load_qml(parent.engine(), qml)

            self.qml = self._component.create()

            if show:
                self.show()
        else:
            assert(type(qml) == QtQuick.QQuickItem)
            self.qml = qml

    def show(self):
        # Hooking the dialog to the parent QML document. This is what makes
        # it displayable.
        self.qml.setProperty("parent", self.parent().rootObject())

    def _load_qml(self, engine, qml_file):
        self._component = QtQml.QQmlComponent(engine)
        url = QtCore.QUrl(
            os.path.dirname(os.path.realpath(__file__)) + "/" + qml_file)
        # Apparently QQmlEngine caches loaded components so there shouldn't
        # be a big performance penalty creating many instances of the same
        # component using this code.
        self._component.loadUrl(url, QtQml.QQmlComponent.PreferSynchronous)
        if self._component.isError():
            for error in self._component.errors():
                print(error.toString())
