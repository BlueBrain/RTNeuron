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

from OpenGL import GLU # Imported first as a workaround for an issue with the
                       # import statement below in Ubuntu 16.04.
from OpenGL import GL

from PyQt5 import QtCore, QtWidgets, QtGui

from .interactors import DefaultInteractionMode

class Background(QtWidgets.QOpenGLWidget):
    """
    This OpenGL widget paints the engine rendering as background and
    handles keyboard/mouse actions for selections
    """

    rectangular_selection = QtCore.pyqtSignal('QRect', 'bool')
    """ Signal emitted when a rectangular selection action finishes.

    The first parameter is the selected rectangle in scene coordinates,
    the second indicates whether the selection is selecting or unselecting.
    """

    def __init__(self, *args, **kwargs):
        super(Background, self).__init__(*args, **kwargs)

        self.interaction_mode = DefaultInteractionMode()
        self.clear_color = [0.0, 0.0, 0.0, 0.0]
        self.event_sink = None
        self.texture = None

        self._paintables = set()

    def add_paintable(self, paintable):
        """ Add an object to be rendered on top or below of the background
            texture.

        Paintable objects must defined method paint which takes a QPainter and
        a QOpenGLWidget as parameters.
        """
        self._paintables.add(paintable)

    def remove_paintable(self, paintable):
        self._paintables.remove(paintable)

    def event(self, event):
        """ Forward the events to the application QObject """
        self._forward = True # This is needed to distinguish between events
                             # that need to be accepted by QOpenGLWidget and
                             # those being accepted by the iteraction mode
        ret = QtWidgets.QOpenGLWidget.event(self, event)

        if self._forward and self.event_sink:
            QtCore.QCoreApplication.sendEvent(self.event_sink, event)

        return True

    def keyPressEvent(self, event):
        self.interaction_mode = \
            self.interaction_mode.keyPressEvent(event, self)
        self._forward = not event.isAccepted()

    def keyReleaseEvent(self, event):
        self.interaction_mode = \
            self.interaction_mode.keyReleaseEvent(event, self)
        self._forward = not event.isAccepted()

    def mouseMoveEvent(self, event):
        self.interaction_mode = \
            self.interaction_mode.mouseMoveEvent(event, self)
        self._forward = not event.isAccepted()

    def mousePressEvent(self, event):
        self.interaction_mode = \
            self.interaction_mode.mousePressEvent(event, self)
        self._forward = not event.isAccepted()

    def mouseReleaseEvent(self, event):
        self.interaction_mode = \
            self.interaction_mode.mouseReleaseEvent(event, self)
        self._forward = not event.isAccepted()

    def wheelEvent(self, event):
        self.interaction_mode = \
            self.interaction_mode.wheelEvent(event, self)
        self._forward = not event.isAccepted()

    def initializeGL(self):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_TEXTURE)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_LIGHTING)

    def paintGL(self):
        if self.texture == None:
            GL.glClearColor(0.4, 0.4, 0.4, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            return

        GL.glClearColor(*self.clear_color)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        painter = QtGui.QPainter()

        width = float(self.rect().width())
        height = float(self.rect().height())

        # Letting the interactor paint its content behind the texture
        painter.begin(self)
        self.interaction_mode.paint(painter, self, -1)
        painter.end()

        try:
            GL.glEnable(GL.GL_TEXTURE_RECTANGLE)
            GL.glBindTexture(GL.GL_TEXTURE_RECTANGLE, self.texture)

            # Blend the current buffer with the rendering from RTNeuron
            GL.glEnable(GL.GL_BLEND)
            # We don't multiply by alpha again because the colors come
            # premultiplied from RTNeuron.
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, height)
            GL.glVertex3f(-1.0, 1.0, 0.0)
            GL.glTexCoord2f(width, height)
            GL.glVertex3f(1.0, 1.0, 0.0)
            GL.glTexCoord2f(width, 0.0)
            GL.glVertex3f(1.0, -1.0, 0.0)
            GL.glTexCoord2f(0.0, 0.0)
            GL.glVertex3f(-1.0, -1.0, 0.0)
            GL.glEnd()

            GL.glBindTexture(GL.GL_TEXTURE_RECTANGLE, 0)
            GL.glDisable(GL.GL_TEXTURE_RECTANGLE)
        except GL.error.GLError as exc:
            print(exc)

        # Letting the interactor paint its content on top of the texture
        # and then the other paintable objects.
        painter.begin(self)
        self.interaction_mode.paint(painter, self, 1)

        for paintable in self._paintables:
            paintable.paint(painter, self)

        painter.end()
