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

from PyQt5 import QtGui, QtCore

from rtneuron.gui.QMLComponent import QMLComponent

__all__ = ['QMLComponent']

class QMLAnnotation(QMLComponent):

    closed = QtCore.pyqtSignal(object)

    def __init__(self, gui, camera):
        super(QMLAnnotation, self).__init__(gui.overlay(), "Annotation.qml")
        self._point = None
        self._camera = camera

        background = gui.background()
        background.add_paintable(self)
        background.update()

        self.qml.updated.connect(background.update)
        self.qml.closeClicked.connect(self._on_closed)

    def track_point(self, point):
        self._point = point

    def remove(self, gui):
        self.qml.setVisible(False)
        self.qml.setProperty("parent", None)
        background = gui.background()
        background.remove_paintable(self)

    def paint(self, painter, glwidget):
        # This method paints the line connection the 3D point tracked in the
        # scene and the annotation
        if self._point is None:
            return

        end = self._compute_projected_point(glwidget)

        qml = self.qml
        position = QtGui.QVector2D(qml.property("x"), qml.property("y"))
        size = QtGui.QVector2D(qml.property("width"), qml.property("height"))
        begin = position + size * 0.5
        # Finding out in which subspace of an X-partition centered at the
        # annotation center the target is lying.
        p = end - begin
        if p[0] > p[1]: # Below x=y diagonal
            if p[0] > -p[1]: # Above x=-y diagonal
                begin.setX(begin.x() + size.x() * 0.5)
            else: # Below x=-y diagonal
                begin.setY(begin.y() - size.y() * 0.5)
        else:  # Above x=y diagonal
            if p[0] > -p[1]: # Above x=-y diagonal
                begin.setY(begin.y() + size.y() * 0.5)
            else: # Below x=-y diagonal
                begin.setX(begin.x() - size.x() * 0.5)

        color = self.qml.property("color")
        opacity = self.qml.property("opacity")
        width = self.qml.property("lineWidth")
        # Before using the color we have to repeat the premultiplied alpha
        # fiasco done by Qt with QQuickWidget
        color.setRed(color.red() * opacity)
        color.setGreen(color.green() * opacity)
        color.setBlue(color.blue() * opacity)
        color.setAlphaF(opacity)

        painter.setPen(QtGui.QPen(QtGui.QBrush(color), width))
        painter.drawLine(begin.x(), begin.y(), end.x(), end.y())

    def _compute_projected_point(self, glwidget):
        p = self._camera.projectPoint(self._point)
        return QtGui.QVector2D(
            (p[0] + 1) * 0.5 * float(glwidget.rect().width()),
            (1 - p[1]) * 0.5 * float(glwidget.rect().height()))

    @QtCore.pyqtSlot()
    def _on_closed(self):
        self.qml.setVisible(False)
        self.qml.setProperty("parent", None)
        self.closed.emit(self)
