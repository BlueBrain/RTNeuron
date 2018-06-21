# -*- coding: utf-8 -*-
## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Daniel Nachbaur <danielnachbaur@googlemail.com>
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
import sip
from PyQt5 import QtCore, QtGui, QtQuickWidgets

class Overlay(QtQuickWidgets.QQuickWidget):
    """
    The widget contains the QML GUI which is rendered transparently as an
    overlay.
    """

    def __init__(self, parent, qml_file, *args):

        url = QtCore.QUrl(qml_file)
        super(Overlay, self).__init__(url, parent, *args)

        self.setResizeMode(QtQuickWidgets.QQuickWidget.SizeRootObjectToView)

        format = QtGui.QSurfaceFormat()
        format.setAlphaBufferSize(8)
        self.setFormat(format)

        # Qt doesn't get blending right in this case, but there's little we
        # can do about it. The problem is that the background is blended with
        # the content and when the result is going to be blended with the
        # widgets underneath, Qt doesn't considered that the alpha channel is
        # already multiplied (as it should be doing front-to-back compositing,
        # but it probably applies the back-to-front formula). The result is that
        # the "transparent" background stains all semitransparent colors in
        # the QML GUI making them darker.
        self.setAttribute(QtCore.Qt.WA_AlwaysStackOnTop)
        self.setClearColor(QtCore.Qt.transparent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # The two functions below are needed because QtQuickWidget is not
        # forwarding the key events to the parent widget in the stack order.
        # There is no direct translation from KeyEvent to QKeyEvent, indeed
        # what we get from QML is a plain QObject (and QEvent doesn't inherit
        # from QObject), so we can to extract the event properties and create
        # a new event for forwarding.
        def onKeyPressed(qml_event):
            key = qml_event.property("key")
            modifiers = \
                QtCore.Qt.KeyboardModifiers(qml_event.property("modifiers"))
            text = qml_event.property("text")
            event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                    key, modifiers, text)
            self.parent().event(event)

        def onKeyReleased(qml_event):
            key = qml_event.property("key")
            modifiers = \
                QtCore.Qt.KeyboardModifiers(qml_event.property("modifiers"))
            text = qml_event.property("text")
            event = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease,
                                    key, modifiers, text)
            self.parent().event(event)

        qml = self.rootObject()
        qml.keyPressed.connect(onKeyPressed)
        qml.keyReleased.connect(onKeyReleased)
