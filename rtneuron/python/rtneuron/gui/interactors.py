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

from PyQt5 import QtCore, QtGui

class InteractionMode(object):
    """Base class of all keyboard/mouse interaction modes.

    An interaction mode is a keyboard/mouse event processor that can
    perform special actions on a GraphicsView, such as selections and
    manipulations of the background scene.

    Each keyboard/mouse event processing function returns an interaction
    mode. This object can be self, to indicate that the mode has not changed
    or a new object when a mode transition has occurred.

    The default implementation of each handler flags the event as ignored.
    Derived classes must flag the event as ignored if it must be sent to
    the scene for further processing (which includes overlay widgets and
    the background scene).
    """

    def keyPressEvent(self, event, background):
        return self._noAction(event)

    def keyReleaseEvent(self, event, background):
        return self._noAction(event)

    def mouseMoveEvent(self, event, background):
        return self._noAction(event)

    def mousePressEvent(self, event, background):
        return self._noAction(event)

    def mouseReleaseEvent(self, event, background):
        return self._noAction(event)

    def wheelEvent(self, event, background):
        return self._noAction(event)

    def paint(self, painter, background, layer) :
        """Functions invoked from Background.paintGL to let this
        InteractionMode paint its visual elements

        Parameters:
        - painter: An active QPainter object
        - background: The Background object, a QOpenGLWidget.
        - layer: The number of layer to be painted, -1 for graphics under the
          background texture and 1 for graphics on top.
        """
        pass

    def _noAction(self, event):
        event.ignore()
        return self

class DefaultInteractionMode(InteractionMode):
    """Default interaction mode.

    This mode simple ignores all events except this key presses:
    - CTRL: rectangular selection
    - CTRL+SHIFT: rectangular unselection
    """

    def keyPressEvent(self, event, background):
        event.ignore()
        if event.key() == QtCore.Qt.Key_Control:
            mode = RectangularSelectionMode.modeFromKeyModifiers(
                event.modifiers())
            return RectangularSelectionMode(mode)

        return self

class RectangularSelectionMode(DefaultInteractionMode):
    """Rectangular selection mode"""

    SELECT = 0
    UNSELECT = 1
    FILL_COLOR = QtGui.QColor(127, 127, 127, 32)

    @staticmethod
    def modeFromKeyModifiers(modifiers):
        if (int(modifiers) & QtCore.Qt.ShiftModifier) == 0:
            return RectangularSelectionMode.SELECT
        return RectangularSelectionMode.UNSELECT

    def __init__(self, mode):
        DefaultInteractionMode.__init__(self)
        self._mode = mode
        self._rectangle = None
        self._exit_on_release = False

    def keyPressEvent(self, event, background):
        event.ignore()

        # Changing to unselect mode if shift is pressed
        if event.key() == QtCore.Qt.Key_Shift:
            self._mode = self.UNSELECT

        return self

    def keyReleaseEvent(self, event, background):
        event.ignore()

        key = event.key()
        # Changing to select mode if shift is released
        if key == QtCore.Qt.Key_Shift:
            self._mode = self.SELECT

        # Chaning to default mode if control is released
        elif key == QtCore.Qt.Key_Control:
            if self._rectangle:
                # Already drawing a rectangle, don't exit this mode yet
                return self
            return DefaultInteractionMode()

        return self

    def mousePressEvent(self, event, background):
        if event.button() != QtCore.Qt.LeftButton:
            return self

        # Starting rectangle drawing
        event.accept()
        pos = event.pos()
        self._anchorPoint = pos
        self._rectangle = QtCore.QRect(pos.x(), pos.y(), 1, 1)
        background.update(self._rectangle)
        return self

    def mouseMoveEvent(self, event, background):
        if (int(event.buttons()) & QtCore.Qt.LeftButton == 0 or
            not self._rectangle):
            return self

        # Updating the rectangle
        event.accept()
        pos = event.pos()
        x = max(0, min(pos.x(), self._anchorPoint.x()))
        y = max(0, min(pos.y(), self._anchorPoint.y()))
        X = min(background.rect().width() - 1,
                max(pos.x(), self._anchorPoint.x()))
        Y = min(background.rect().height() - 1,
                max(pos.y(), self._anchorPoint.y()))

        self._rectangle.setRect(x, y, X - x + 1, Y - y + 1)
        background.update(self._rectangle)
        return self

    def mouseReleaseEvent(self, event, background):
        if event.button() != QtCore.Qt.LeftButton or not self._rectangle:
            return self

        event.accept()

        # Emitting the rectangle selection signal
        background.rectangular_selection.emit(self._rectangle,
                                              self._mode == self.SELECT)

        # Rectangle drawing finished, removing from the scene
        rect = self._rectangle
        self._rectangle = None
        background.update(rect)

        # Staying in this mode if CTRL is still pressed
        if int(event.modifiers()) & QtCore.Qt.ControlModifier == 0:
            return DefaultInteractionMode()
        return self

    def paint(self, painter, glwidget, layer) :

        if layer == -1 or not self._rectangle:
            return

        fill = QtGui.QBrush(self.FILL_COLOR)
        painter.setBrush(fill)
        painter.drawRect(self._rectangle)
