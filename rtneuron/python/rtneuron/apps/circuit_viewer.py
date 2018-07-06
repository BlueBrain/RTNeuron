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

import os as _os
import numpy as _np
import random as _random

from PyQt5 import QtCore, QtGui, QtQuick

from OpenGL import GL

import rtneuron as _rtneuron
from rtneuron.gui import LoaderGUI, SelectionHandler, \
                         display_empty_scene_with_GUI
from rtneuron.gui.QMLAnnotation import QMLAnnotation
from rtneuron.gui.dialogs import AcceptDialog, OpenSimulationDialog
from rtneuron.gui.enums import QmlRepresentationMode
from rtneuron.gui.simulationPlayer import SimulationPlayer
from rtneuron.gui import interactors
from rtneuron.gui import util as _util

import brain as _brain

filepath = _os.path.dirname(_os.path.realpath(__file__))

def qtcolor_to_list(color):
    return [color.redF(), color.greenF(), color.blueF(), color.alphaF()]

class AbstractListModel(QtCore.QAbstractListModel):

    def __init__(self, parent=None, *args):
        super(AbstractListModel, self).__init__(parent, *args)
        self._data = []

    def add(self, *args):
        self.beginResetModel()
        self._data.append(list(args))
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = []
        self.endResetModel()

    def remove(self, index):
        self.beginResetModel()
        del self._data[index]
        self.endResetModel()

    def roleNames(self):
        return self._roles

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def data(self, index, role):
        return QtCore.QVariant(
            self._data[index.row()][role - QtCore.Qt.UserRole])

class SliceEditor(QtCore.QObject):

    class InteractionMode(interactors.InteractionMode):

        ROTATE = 1
        MOVE = 2
        glyph_radius = 18

        def __init__(self, editor):
            self._editor = editor
            self._mode = self.MOVE
            self._glyph_position = None
            self._engaged = False
            self._moved = False
            self._move_icon = QtGui.QImage(filepath + "/../gui/icons/move.png")
            self._rotate_icon = \
                QtGui.QImage(filepath + "/../gui/icons/rotate.png")

        def mouseMoveEvent(self, event, background):

            if (not self._engaged or
                int(event.buttons()) & QtCore.Qt.LeftButton == 0):
                return self._noAction(event)

            self._moved = True
            if self._mode == self.ROTATE:
                self._rotate_planes(event.pos())
            else:
                assert(self._mode == self.MOVE)
                width = float(background.rect().width())
                height = float(background.rect().height())
                self._editor.move_planes([event.pos().x() / width * 2 - 1,
                                          1 - event.pos().y() / height * 2])
            background.update()
            event.accept()
            return self

        def wheelEvent(self, event, background):

            position = event.pos()
            d = self._glyph_position - position
            if QtCore.QPointF.dotProduct(d, d) < self.glyph_radius ** 2:
                delta = event.angleDelta().y() / 120.0
                width = self._editor._width
                if delta > 0:
                    self._editor.set_width(width * 1.1 * delta)
                else:
                    self._editor.set_width(width * 0.9 * -delta)
                event.accept()
                return self

            return self._noAction(event)

        def mousePressEvent(self, event, background):

            if (int(event.buttons()) & QtCore.Qt.LeftButton == 0):
                return self._noAction(event)

            position = event.pos()
            d = self._glyph_position - position
            if QtCore.QPointF.dotProduct(d, d) < self.glyph_radius ** 2:
                self._rotation_ref = event.pos()
                self._moved = False
                self._engaged = True
                event.accept()

            return self

        def mouseReleaseEvent(self, event, background):

            if not self._moved and self._engaged:
                # Toggle between move and rotate modes
                self._mode = self.ROTATE if self._mode == self.MOVE \
                             else self.MOVE
            background.update()
            self._engaged = False

            return self._noAction(event)

        def paint(self, painter, background, layer):

            self._editor.paint_planes(painter, background, layer)

            if layer == 1:
                # Paint the interaction glyph
                camera = self._editor._view.camera
                p = self._editor._mid_point
                p = camera.projectPoint([p[0], p[1], p[2]])
                width = float(background.rect().width())
                height = float(background.rect().height())
                self._glyph_position = QtCore.QPointF((p[0] + 1) * 0.5 * width,
                                                      (1 - p[1]) * 0.5 * height)

                color = QtGui.QColor(224, 224, 224, 255)
                painter.setPen(QtGui.QPen(color))
                font = painter.font()
                radius = self.glyph_radius
                p = self._glyph_position - QtCore.QPointF(radius, radius)
                _os.path.dirname(_os.path.realpath(__file__))
                if self._mode == self.MOVE:
                    painter.drawImage(p, self._move_icon)
                else:
                    painter.drawImage(p, self._rotate_icon)

        def _rotate_planes(self, position):

            d = (position - self._rotation_ref)
            self._rotation_ref = position
            # v is the perpendicular vector to the displacement, but since y
            # points down in screen coordintes we have also to negate it.
            v = QtGui.QVector3D(d.y(), d.x(), 0)
            amount = v.length()
            self._editor.rotate_planes(v / amount, amount * 0.25)

    slice_width_changed = QtCore.pyqtSignal(float)

    def __init__(self, view, background):

        super(SliceEditor, self).__init__()

        self._view = view

        # Using the camera and home position to figure out a reasonable initial
        # position for the clipping planes.
        cam_position, cam_orientation = self._view.camera.getView()
        cam_position = QtGui.QVector3D(*cam_position)
        eye, center, up = view.cameraManipulator.getHomePosition()
        center = QtGui.QVector3D(*center)
        cam_to_world_rot = QtGui.QQuaternion.fromAxisAndAngle(
            QtGui.QVector3D(*cam_orientation[0]), cam_orientation[1])
        world_to_cam_rot = QtGui.QQuaternion(cam_to_world_rot.scalar(),
                                             -cam_to_world_rot.vector())
        p = world_to_cam_rot.rotatedVector(center - cam_position)
        point = QtGui.QVector3D(0, 0, p.z())
        # Transforming the mid point to world coordinates
        self._mid_point = cam_to_world_rot.rotatedVector(point) + cam_position
        self._width = abs(point.z()) / 20.0
        self._plane_side = point.z() * 0.8

        axis = cam_to_world_rot.rotatedVector(QtGui.QVector3D(1, 0, 0))
        self._axis = axis
        # We use the camera front and up vectors in world coordinates as
        # hints for the u, v vectors to calculate the corners. The plane
        # axis is then used to compute the good vectors
        self._x_hint = cam_to_world_rot.rotatedVector(QtGui.QVector3D(0, 0, 1))
        self._y_hint = cam_to_world_rot.rotatedVector(QtGui.QVector3D(0, 1, 0))
        self._first_plane_index = 0

        self._update_scene_clip_planes()

    def show_and_edit(self, background):

        self._old_interaction_mode = background.interaction_mode
        background.interaction_mode = self.InteractionMode(self)
        background.update()

    def hide(self, background):

        background.interaction_mode = self._old_interaction_mode
        background.update()

    def paint_planes(self, painter, background, layer):

        self._paint_plane(painter, background, layer, 0)
        self._paint_plane(painter, background, layer, 1)

    def rotate_planes(self, screen_vector, amount):

        camera = self._view.camera
        position, orientation = camera.getView()
        cam_to_world_rot = QtGui.QQuaternion.fromAxisAndAngle(
            QtGui.QVector3D(*orientation[0]), orientation[1])
        axis = cam_to_world_rot.rotatedVector(screen_vector)
        rotation = QtGui.QQuaternion.fromAxisAndAngle(axis, amount)

        # Since the transformation is a rotation, we can transform the normal
        # of the planes using the rotation itself (R-1^t = R)
        self._axis = rotation.rotatedVector(self._axis)
        # We need to rotate the hint vectors for u and v, because otherwise
        # the rotation displayed is really confusing.
        self._x_hint = rotation.rotatedVector(self._x_hint)
        self._y_hint = rotation.rotatedVector(self._y_hint)

        self._update_scene_clip_planes()

    def move_planes(self, normalized_screen_position):

        camera = self._view.camera
        position, orientation = camera.getView()
        position = QtGui.QVector3D(*position)
        world_to_cam_rot = QtGui.QQuaternion.fromAxisAndAngle(
            QtGui.QVector3D(*orientation[0]), -orientation[1])
        camera_pos = world_to_cam_rot.rotatedVector(self._mid_point - position)
        self._mid_point = QtGui.QVector3D(
            *camera.unprojectPoint(normalized_screen_position, camera_pos.z()))
        self._update_scene_clip_planes()

    def set_width(self, width):

        if self._width == width:
            return

        self._width = width
        self._update_scene_clip_planes()
        self.slice_width_changed.emit(width)

    def _paint_plane(self, painter, background, layer, plane):

        camera = self._view.camera
        position = QtGui.QVector3D(*camera.getView()[0])

        axis = self._axis if plane == 0 else -self._axis
        d = self._planes[plane][3]

        # To decide if a clip plane has to be rendered in front or behind
        # the scene we have to check if the camera is in the negative
        # (front layer) or positive hemispace (back layer).
        # When the sign of inserting the camera position on the plane
        # equation is the sign of the layer, then we have to render the plane.
        if (QtGui.QVector3D.dotProduct(axis, position) + d) * layer > 0:
            return

        # Computing the corners of the plane representation
        v = QtGui.QVector3D.crossProduct(axis, self._x_hint).normalized()
        if v.lengthSquared() == 0:
            v = self._y_hint
        u = QtGui.QVector3D.crossProduct(v, axis).normalized()
        if u.lengthSquared() == 0:
            u = self._x_hint

        corners = []
        halfL = self._plane_side * 0.5
        # Each plane is placed apart from mid_point in the opposite direction
        # to which its axes points.
        p = self._mid_point + -axis * self._width
        corners = [p - halfL * u - halfL * v, p + halfL * u - halfL * v,
                   p + halfL * u + halfL * v, p - halfL * u + halfL * v]

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix();
        GL.glLoadMatrixf(camera.getViewMatrix().flatten(order="F"))
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix();
        GL.glLoadMatrixf(camera.getProjectionMatrix().flatten(order="F"))

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glColor4f(1, 0, 0, 0.125)
        GL.glBegin(GL.GL_QUADS)
        for corner in corners:
            GL.glVertex3f(corner[0], corner[1], corner[2])
        GL.glEnd()

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix();
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix();

    def _update_scene_clip_planes(self):

        # Adding clipping planes to the scene
        a = self._axis
        m = self._mid_point
        d = self._width

        d1 = -QtGui.QVector3D.dotProduct(a, m - a * d)
        d2 = -QtGui.QVector3D.dotProduct(-a, m + a * d)

        self._planes = ([a[0], a[1], a[2], d1], [-a[0], -a[1], -a[2], d2])
        scene = self._view.scene
        scene.setClipPlane(self._first_plane_index, self._planes[0])
        scene.setClipPlane(self._first_plane_index + 1, self._planes[1])

class ModelList(AbstractListModel):

    def __init__(self, parent=None, *args):
        self._roles = {QtCore.Qt.UserRole + 0: b'file_name',
                       QtCore.Qt.UserRole + 1: b'model_color'}
        super(ModelList, self).__init__(parent, *args)

    def update(self, index, color):
        self._data[index][1] = color

class ModelHandler(QtCore.QObject):

    def __init__(self, scene, qml):
        super(ModelHandler, self).__init__()

        self._modelList = ModelList()
        self._handles = []
        self._scene = scene
        self._qml = qml

        qml.addModel.connect(self._on_model_add)
        modelList = qml.getModelList()
        modelList.setModel(self._modelList)
        modelList.colorChanged.connect(self._on_color_changed)
        modelList.removeItem.connect(self._on_model_remove)

    def _on_model_add(self, name):

        try:
            attributes = _rtneuron.AttributeMap()
            attributes.color = [1, 1, 1, 1]
            handle = self._scene.addModel(name, attributes=attributes)
        except RuntimeError as e:
            self._qml.showError(str(e))
            return

        view = _rtneuron.engine.views[0]
        view.computeHomePosition()

        self._modelList.add(name, "#ffffffff")
        self._handles.append(handle)

    def _on_color_changed(self, index, color):

        self._modelList.update(index, color)
        handle = self._handles[index]
        handle.attributes.color = qtcolor_to_list(color)
        handle.update()

    def _on_model_remove(self, index):

        self._modelList.remove(index)
        self._scene.remove(self._handles[index])
        del self._handles[index]

        view = _rtneuron.engine.views[0]
        view.computeHomePosition()

class CellDyeList(AbstractListModel):

    def __init__(self, parent=None, *args):
        self._roles = {QtCore.Qt.UserRole + 0: b'key',
                       QtCore.Qt.UserRole + 1: b'fraction',
                       QtCore.Qt.UserRole + 2: b'primary_color',
                       QtCore.Qt.UserRole + 3: b'secondary_color',
                       QtCore.Qt.UserRole + 4: b'mode'}
        super(CellDyeList, self).__init__(parent, *args)

    def update(self, index, primary_color, secondary_color, mode):
        self._data[index][2] = primary_color
        self._data[index][3] = secondary_color
        self._data[index][4] = mode


class TargetDisplayHandler(QtCore.QObject):

    # Signals used to make sure that callbacks invoked from the RTNeuron thread
    # get dispatched to the GUI thread if they need to modify the GUI.
    _cell_selected = QtCore.pyqtSignal(int, int, int)

    class Dye:
        def __init__(self, handle, colors, mode):
            self.handle = handle
            self.colors = colors
            self.mode = mode

    def __init__(self, scene, simulation, qml):
        super(TargetDisplayHandler, self).__init__()

        self._scene = scene
        scene.cellSelected.connect(self._cell_selected.emit)
        self._cell_selected.connect(self._on_cell_selected)

        self._gid = None
        self._dyes = []
        self._all_neurons = scene.objects[0]
        self._default_mode = self._all_neurons.attributes.mode

        self._cellDyeModel = CellDyeList()

        dyeList = qml.getCellDyeList()
        dyeList.setModel(self._cellDyeModel)
        dyeList.removeItem.connect(self._on_cell_dye_remove)
        dyeList.colorsChanged.connect(self._on_colors_changed)
        dyeList.toggleMode.connect(self._on_toggle_mode)

        qml.addCellDye.connect(self._on_add_cell_dye)
        qml.clearDyes.connect(self._reset)
        self._qml = qml

    def _on_cell_selected(self, gid, section, segment):

        if self._gid != None:
            # Returning to previous dislays modes
            self._scene.neuronSelectionMask = []
            if len(self._dyes) != 0:
                # Restoring the attributes of the cell subsets
                self._apply_dyes()
            else:
                self._all_neurons.attributes.mode = self._default_mode
                self._all_neurons.update()
            self._gid = None
        else:
            # Making all neurons invisible except the chosen one
            neurons = self._all_neurons
            neurons.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            neurons.update()
            # Making these neurons also unselectable
            mask = _np.delete(neurons.object, gid)
            self._scene.neuronSelectionMask = mask

            cell = neurons.query([gid])
            cell.attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
            cell.update()
            self._gid = gid

    def _on_add_cell_dye(self, key, fraction, primary_color, secondary_color):

        # Checking that the target is valid
        try:
            gids = _util.target_string_to_gids(str(key), _rtneuron.simulation)
        except ValueError as e:
            self._qml.showError(" ".join(map(str, e.args)))
            return
        # Getting the requested fraction of the target
        _random.shuffle(gids)
        last = int(len(gids) * fraction / 100.0)
        gids.resize((last))

        # Checking that the target doesn't overlap an existing one
        handle = self._all_neurons.query(gids)
        neurons = handle.object
        for dye in self._dyes:
            if len(_np.intersect1d(dye.handle.object, neurons)) != 0:
                self._qml.showError(
                    "Requested cell set overlaps an existing set")
                return
            pass

        # Applying the attributes

        if len(self._dyes) == 0:
            # Making everything invisible if this is the first dye added
            neurons = self._all_neurons
            neurons.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            neurons.update()

        handle.attributes.primary_color = qtcolor_to_list(primary_color)
        handle.attributes.secondary_color = qtcolor_to_list(secondary_color)
        handle.attributes.color_scheme = _rtneuron.ColorScheme.BY_BRANCH_TYPE
        handle.attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
        handle.update()

        # Adding the dye info to the list model
        self._cellDyeModel.add(
            key, fraction, primary_color, secondary_color,
            QmlRepresentationMode.Values.WHOLE_NEURON)
        self._dyes.append(self.Dye(handle, (primary_color, secondary_color),
                                   QmlRepresentationMode.Values.WHOLE_NEURON))

    def _on_cell_dye_remove(self, index):

        self._cellDyeModel.remove(index)
        handle = self._dyes[index].handle
        del self._dyes[index]

        # Restoring the original display attributes if the handle list gets
        # empty and no cell is being uniquely displayed.
        if len(self._dyes) == 0 and self._gid == None:
            self._all_neurons.attributes.mode = self._default_mode
            self._all_neurons.update()
        else:
            handle.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            handle.update()

    def _on_colors_changed(self, index, primary, secondary):

        dye = self._dyes[index]

        handle = dye.handle
        handle.attributes.primary_color = qtcolor_to_list(primary)
        handle.attributes.secondary_color = qtcolor_to_list(secondary)
        dye.colors = (primary, secondary)
        self._cellDyeModel.update(index, primary, secondary, dye.mode)
        handle.update()

    def _on_toggle_mode(self, index):

        dye = self._dyes[index]

        handle = dye.handle
        mode = QmlRepresentationMode.Values.WHOLE_NEURON \
               if dye.mode == QmlRepresentationMode.Values.NO_AXON  \
               else QmlRepresentationMode.Values.NO_AXON
        dye.mode = mode

        handle.attributes.mode = _rtneuron.RepresentationMode.values[mode]
        self._cellDyeModel.update(index, dye.colors[0], dye.colors[1], mode)
        handle.update()

    def _apply_dyes(self):

        for dye in self._dyes:
            attributes = dye.handle.attributes
            attributes.primary_color = qtcolor_to_list(dye.colors[0])
            attributes.secondary_color = qtcolor_to_list(dye.colors[1])
            attributes.color_scheme = _rtneuron.ColorScheme.BY_BRANCH_TYPE
            attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
            dye.handle.update()

    def _reset(self):

        self._dyes = []
        self._cellDyeModel.clear()
        self._all_neurons.attributes.mode = self._default_mode
        self._all_neurons.update()
        self._gid = None

class GUI(LoaderGUI):

    def __init__(self, *args, **kwargs):

        if not 'simulation_config' in kwargs:
            try:
                kwargs['simulation_config'] = _brain.test.blue_config
            except AttributeError:
                pass

        super(GUI, self).__init__(filepath + '/CircuitViewer.qml',
                                  *args, **kwargs)

        qml = self._overlay.rootObject()
        qml.enableSlice.connect(self._on_enable_slice)
        qml.showSlice.connect(self._on_show_slice)
        qml.sliceWidthChanged.connect(self._on_slice_width_changed)

        self._player = SimulationPlayer(
            qml.findChild(QtQuick.QQuickItem, "player",
                          QtCore.Qt.FindDirectChildrenOnly))

        self._background.clear_color = [1, 1, 1, 1]

    def scene_created(self, scene):

        qml = self._overlay.rootObject()
        self._coloring_handler = \
            TargetDisplayHandler(scene, self._simulation, qml)
        self._model_handler = ModelHandler(scene, qml)
        self._scene = scene

    def _connect_engine_implementation(self, engine):

        self._player.engine_player = engine.player
        self._player.view = engine.views[0]
        self._player.open_simulation_clicked.connect(self._on_open_simulation)

    def _on_enable_slice(self, enable):
        background = self.background()
        if enable:
            view = _rtneuron.engine.views[0]
            self._slice_editor = SliceEditor(view, background)
            self._slice_editor.show_and_edit(background)

            self._slice_editor.slice_width_changed.connect(
                self._on_slice_width_changed)
            qml = self._overlay.rootObject()
            qml.setSliceWidth(self._slice_editor._width)
        else:
            del self._slice_editor
            background.interaction_mode = interactors.DefaultInteractionMode()
            self._scene.clearClipPlanes()

    def _on_show_slice(self, show):

        background = self.background()
        if show:
            self._slice_editor.show_and_edit(background)
        else:
            self._slice_editor.hide(background)

    def _on_slice_width_changed(self, width):

        self._slice_editor.set_width(width)
        qml = self._overlay.rootObject()
        qml.setSliceWidth(width)

    def _on_open_simulation(self):

        def disconnect_dialog():
            # Breaking the ref loop between the dialog and this function, which
            # is what keeps the object alive
            dialog.done.disconnect()
            dialog.cancelled.disconnect()

        def on_dialog_done(simulation_applied):
            self._player.enable_playback_controls(simulation_applied)

        dialog = OpenSimulationDialog(
            self._overlay, _rtneuron.simulation, self._player.view.scene)
        dialog.done.connect(on_dialog_done)
        dialog.cancelled.connect(disconnect_dialog)

class App(object):
    """Power application for visualizing circuit slices.

       This application allows the user to load a circuit to be displayed in
       soma mode. Once the circuit is loaded the user can perform 3 types of
       actions:
       - Add remove extra polygonal models to the scene, e.g. meshes of the
         hippocampus regions. The color of the model may be changed after adding
         it to the scene if it doesn't provide any materials or per vertex
         colors.
       - Select a cell set by target name, gid ranges or a list of both and
         display this set (or a fraction of it) with whole neuron models. The
         user can pick the colors to be used for the axon and dendrites/soma.
         Multiple non-overlapping subsets can be added and removed.
       - Add two parallel clipping planes to the scene to obtain a slice of the
         circuit, change their position, distance and orientation.
    """

    def __init__(self, *args, **kwargs):

        attributes = _rtneuron.AttributeMap()
        # Don't use meshes because this tool is going to be used with circuits
        # that lack them
        attributes.use_meshes = False
        # We want the initial circuit building to assume that there are no
        # morphologies to speed it up.
        attributes.load_morphologies = False

        self._gui = display_empty_scene_with_GUI(
            GUI, attributes, *args, **kwargs)

        view = _rtneuron.engine.views[0]
        view.attributes.auto_compute_home_position = False
        view.attributes.auto_adjust_model_scale = False
        view.attributes.background = [0, 0, 0, 0]
        view.attributes.highlight_color = [1, 0.8, 0, 1]

def start(*args, **kwargs):
    """Startup the application for browsing connection pairs.

    Arguments:
    - simulation_config: Path to the default blue config to use in the loader
      menu
    - target: Default target string to use in the loader menu
    """
    return App(*args, **kwargs)

