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

from PyQt5 import QtCore, QtGui, QtWidgets

import rtneuron as _rtneuron

from .background import Background
from .overlay import Overlay

class BaseGUI(QtWidgets.QWidget):

    # Internal
    _exit_signal = QtCore.pyqtSignal()

    def __init__(self, qml_file, *args, **kwargs):

        super(BaseGUI, self).__init__(*args, **kwargs)

        self._app = QtWidgets.QApplication.instance()

        self._background = Background(self)
        format = QtGui.QSurfaceFormat()
        format.setAlphaBufferSize(0)
        format.setDepthBufferSize(0)
        self._background.setFormat(format)

        self._exit_signal.connect(self.close)

        # Creating the overlay
        if not qml_file:
            qml_file = os.path.dirname(
                os.path.realpath(__file__)) + '/BaseOverlay.qml'
        self._overlay = Overlay(self._background, qml_file)

        self._init_implementation()

        self._resize_and_show()

    def background(self):
        """ Return the OpenGL widget on which the Equalizer output is displayed.
        This returned object can be used to add extra paintable objects on top
        of the circuit scene and below the QML overlay.
        """
        return self._background

    def overlay(self):
        """ Return the QQuickWidget with the QML overlay. """
        return self._overlay

    def resizeEvent(self, event):
        """ Forward the resize event to the graphics view and overlay """
        self._background.resize(event.size())
        self._overlay.resize(event.size())

    def run(self):
        """ Execute the GUI event loop """
        return self._app.exec_()

    def get_background_context(self):
        """ Return the OpenGL context of the background widget.
            This context must be shared with the RTNeuron application object
            before RTNeuron.init is called in order for the compositing to
            work."""
        return self._background.context()

    def connect_engine(self, engine):
        """ Connect this GUI and the RTNeuron engine object.
        This can only be done once RTNeuron.init has been called, otherwise
        the result is undefined."""
        try:
            def on_exit():
                """ Slot called when application is done """
                self._background.event_sink = None
                self._background.texture = None
                # self.close() cannot be called directly because the widget is
                # not owned by the calling thread. Posting a QCloseEvent to
                # self doesn't work either, so an custom signal is connected
                # to the slot self.close.
                self._exit_signal.emit()

            engine.textureUpdated.connect(self.update)
            engine.exited.connect(on_exit)

            self._background.event_sink = \
                engine.getActiveViewEventProcessor()

            self._connect_engine_implementation(engine)

        except Exception as e:
            print(e)
            exit()

    def update(self, texture):
        """ Trigger update/rendering of the GUI and the background """
        self._background.texture = texture
        self._background.update()

    def _init_implementation(self):
        """Called from __init__ to do derived classes initialization."""
        pass

    def _connect_engine_implementation(self, engine):
        """Called from connect_engine to connect the signals of a specific GUI
        from a derived class"""
        pass

    def _resize_and_show(self):
        if not 'EQ_WINDOW_IATTR_HINT_FULLSCREEN' in os.environ:
            size = [1280, 800]
            try:
                size[0] = _rtneuron.global_attributes.window_width
            except AttributeError:
                pass
            try:
                size[1] = _rtneuron.global_attributes.window_height
            except AttributeError:
                pass
            self.resize(*size)

            screen = self._app.desktop().availableGeometry();
            self.move((screen.width() - size[0]) / 2,
                      (screen.height() - size[1]) / 2)
            self.show()

        else:
            # Honoring the fullscreen requests to Equalizer.
            size = self._app.desktop().screenGeometry()
            size = [size.width(), size.height()]
            self.resize(*size)
            self.showFullScreen()

        # Enforcing the window size in the Equalizer side. This size may be
        # later overriden if the window manager resizes the window (e.g. when
        # it's bigger than the available screen space).
        os.environ['EQ_WINDOW_IATTR_HINT_WIDTH'] = '%d' % size[0]
        os.environ['EQ_WINDOW_IATTR_HINT_HEIGHT'] = '%d' % size[1]

        self.setWindowTitle("RTNeuron")

        # Ensuring that the OpenGL context for the background is created to
        # pass it to RTNeuron for sharing.
        QtWidgets.QApplication.processEvents();
        assert(self._background.context())

