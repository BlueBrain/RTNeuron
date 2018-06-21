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
import weakref as _weakref

from PyQt5 import QtCore, QtGui

import rtneuron as _rtneuron
from rtneuron.gui import LoaderGUI, display_empty_scene_with_GUI
from rtneuron.gui.QMLAnnotation import QMLAnnotation
from rtneuron.gui.dialogs import AcceptDialog
from rtneuron.gui import interactors

import rtneuron.sceneops.util as _util
from rtneuron.sceneops import NeuronClipping as _NeuronClipping

import brain as _brain

def _compute_postsynaptic_distance_to_soma(morphology, synapse):
    try:
        section = morphology.section(synapse.post_section())
    except RuntimeError:
        # soma synapse
        return 0
    distances = section.sample_distances_to_soma()
    return distances[synapse.post_segment()] + synapse.post_distance()

def _compute_position(morphology, transform, sec_id, seg_id, seg_distance):
    section = morphology.section(sec_id)
    position = _np.array(section[seg_id][:3])
    segment = _np.array(section[seg_id + 1][:3]) - position
    length = _np.linalg.norm(segment)
    position += segment * (seg_distance / length)
    # transforming the position using homogeneous coordinates
    position.resize((4))
    position[3] = 1
    position = (transform.dot(position.reshape(4, 1))).reshape(4)
    return position[:3]

def _compute_presynaptic_position(cell_pair, synapse):
    return list(map(float, _compute_position(
        cell_pair.morphologies[0], cell_pair.transforms[0],
        synapse.pre_section(), synapse.pre_segment(), synapse.pre_distance())))

def _compute_postsynaptic_position(cell_pair, synapse):
    section = synapse.post_section()
    morphology = cell_pair.morphologies[1]
    transform = cell_pair.transforms[1]
    if section == 0:
        # soma synapse
        bouton = _np.array(_compute_presynaptic_position(cell_pair, synapse))
        soma_pos = transform[:3 ,3].reshape(3)
        d = bouton - soma_pos
        d /= _np.linalg.norm(d)
        return list(map(float, soma_pos + d * morphology.soma().max_radius()))

    return list(map(float, _compute_position(
        morphology, transform, section,
        synapse.post_segment(), synapse.post_distance())))

def _synaptic_properties(post_morphology):

    def distance_to_soma(synapse):
        return _compute_postsynaptic_distance_to_soma(post_morphology,
                                                      synapse)

    return [["Distance to soma", "µm", distance_to_soma],
            ["Release probability", "", _brain.Synapse.utilization],
            ["Conductance", "nS", _brain.Synapse.conductance],
            ["\u03C4 facilication", "ms", _brain.Synapse.facilitation],
            ["\u03C4 depression", "ms", _brain.Synapse.depression],
            ["\u03C4 conductance decay", "ms", _brain.Synapse.decay]]

class KeyHandler(interactors.InteractionMode):

    def __init__(self):
        super(KeyHandler, self).__init__()
        self.shiftPressed = False

    def keyPressEvent(self, event, background):
        event.ignore()
        if event.key() == QtCore.Qt.Key_Shift:
            self.shiftPressed = True
        return self

    def keyReleaseEvent(self, event, background):
        event.ignore()
        if event.key() == QtCore.Qt.Key_Shift:
            self.shiftPressed = False
        return self

class CellPair():

    def __init__(self):
        self.reset()

    def reset(self):
        self.handlers = [None, None]
        self.gids = [None, None]
        self.transforms = None
        self.morphologies = None

class SimpleSelectableListModel(QtCore.QAbstractListModel):

    # We define our own signal because it's much easier to understand than the
    # API from QtCore.QAbstractListModel):
    item_selection_changed = QtCore.pyqtSignal(int, bool) # index, selected

    def __init__(self, parent=None, *args):

        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self._roles = {QtCore.Qt.UserRole + 0: b'name',
                       QtCore.Qt.UserRole + 1: b'value',
                       QtCore.Qt.UserRole + 2: b'selected'}
        self._names = ['']
        self._values = ['']
        self._selected = ['']

    def set(self, keyvalue_pairs):
        self.beginResetModel()
        self._names = []
        self._values = []
        for key, value in keyvalue_pairs:
            self._names.append(key)
            self._values.append(value)
        self._selected = [False] * len(self._values)
        self.endResetModel()

    def roleNames(self):
        return self._roles

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._names)

    @QtCore.pyqtSlot(int)
    def toggleItem(self, index):
        self._selected[index] = not self._selected[index]
        modelIndex = self.index(index, 0)
        self.dataChanged.emit(modelIndex, modelIndex, [QtCore.Qt.UserRole+2])
        self.item_selection_changed.emit(index, self._selected[index])

    # This property is needed to query the number of rows from QML to be
    # able to choose the height of the displaying list. The function rowCount
    # can't be used because if decorated as a property a quasi-silent
    # exception occurs somewhere and the code doesn't work.
    @QtCore.pyqtProperty(int)
    def rows(self):
        return len(self._names)

    def data(self, index, role):
        if role == QtCore.Qt.UserRole:
            return QtCore.QVariant(self._names[index.row()])
        elif role == QtCore.Qt.UserRole + 1:
            return QtCore.QVariant(self._values[index.row()])
        elif role == QtCore.Qt.UserRole + 2:
            return QtCore.QVariant(self._selected[index.row()])

class TargetDisplayHandler(QtCore.QObject):
    """This class handles the coloring and display mode of the set of cells
    that can be picked by the user at any given moment."""

    filter_mask_changed = QtCore.pyqtSignal(_np.ndarray) # index, selected

    class Coloring(object):
        NAMES = ["solid", "layer", "mtype", "metype"]
        SOLID = 0
        LAYER = 1
        MTYPE = 2
        METYPE = 3

    def __init__(self, scene, simulation, qml):
        super(TargetDisplayHandler, self).__init__()

        Coloring = self.Coloring
        self._scene = scene
        self._mode = self.Coloring.SOLID
        self._qml_legend = qml.getLegend()

        qml.coloringChanged.connect(self._on_coloring_changed)

        neurons = scene.objects[0]
        self._cell_handler = neurons
        self._synapse_handlers = None
        self._neuron_solid_color = neurons.attributes.color
        self._cells_visible = True

        # Needed to restore the display mode in the subhandlers used in some
        # cases.
        self._display_mode = neurons.attributes.mode

        try:
            all_gids = neurons.object
            layer_colors = [
                ['Layer %d' % i for i in range(1, 7)],
                # We don't need the full targets for color_lists, just the
                # intersection between the targets and the neurons being
                # displayed
                [_np.intersect1d(simulation.gids("Layer%d" % i), all_gids)
                 for i in range(1, 7)],
                [(1, 0.1, 0.1, 1), (1, 0.5, 0.1, 1), (1, 0.8, 0.2, 1),
                 (0.2, 0.8, 0.2, 1), (0.4, 0.8, 1.0, 1.0), (0.6, 0.4, 1.0, 1)]]
        except RuntimeError:
            # Layer targets do not exist
            layer_colors = None

        # Color generation deferred for for Mtype and MEtype colors
        self._color_lists = {Coloring.MTYPE: None, Coloring.METYPE: None}
        if layer_colors:
            self._color_lists[Coloring.LAYER] = layer_colors

        # We want to keep the explicit lists of what is masked by selections
        # in the legend and what not to be able to work with subsets of the
        # target handler.
        self._masked = _np.zeros((0), dtype="u4")
        self._unmasked = all_gids
        # We need to keep track of the selected items to recompute the masked
        # and unmasked sets when set_target_cells is called.
        self._selected_legend_items = set()

    def set_target_cells(self, handler, solid_color, visible=True):
        self._cell_handler = handler
        self._neuron_solid_color = solid_color
        self._cells_visible = visible

        # Computing the target with all the items selected in the legend
        self._init_masks()

    def set_cells_visible(self, visible):
        self._cells_visible = visible
        self._colorize_neurons()

    def set_target_synapses(self, handlers, solid_color):
        self._synapse_handlers = handlers
        self._synapse_solid_color = solid_color

    def colorize(self):
        _rtneuron.engine.pause()
        self._colorize_neurons()
        self._colorize_synapses()
        _rtneuron.engine.resume()

    def _create_coloring_info_from_neurons(self, gids, coloring):

        if coloring == self.Coloring.MTYPE:
            type_to_colors = _util.create_mtype_colors
        elif coloring == self.Coloring.METYPE:
            type_to_colors = _util.create_metype_colors
        else:
            raise ValueError("Invalid coloring mode")

        def resort_colors(coloring_data):
            first = []
            second = []
            for label, target, color in zip(*coloring_data):
                # Trying to separate inhibitory and excitatory colors for
                # mtype and metype colorings.
                if color[2] < 0.3:
                    first.append((label, target, color))
                else:
                    second.append((label, target, color))
            return [x for x in zip(*(first + second))]

        return resort_colors(type_to_colors(gids, self._scene.circuit))

    def _get_coloring_info(self, mode):
        coloring_info = self._color_lists[mode]
        if coloring_info is None:
            # Creating the coloring info for this mode
            gids = self._scene.objects[0].object
            coloring_info = \
                self._create_coloring_info_from_neurons(gids, mode)
            self._color_lists[self._mode] = coloring_info
        return coloring_info

    def _colorize_neurons(self):
        handler = self._cell_handler
        if not handler:
            return

        if not self._cells_visible:
            handler.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            handler.update()
            return

        if self._masked.size != 0:
            invisible = handler.query(self._masked)
            invisible.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            invisible.update()
            # We don't need this query if we know that all cells from
            # the target handler are unmasked.
            handler = handler.query(self._unmasked)

        handler.attributes.mode = self._display_mode
        handler.attributes.color_scheme = _rtneuron.ColorScheme.SOLID

        if self._mode == self.Coloring.SOLID:
            handler.attributes.color = self._neuron_solid_color
            handler.update()
        else:
            handler.update()

            dummy, targets, colors = self._get_coloring_info(self._mode)

            for gids, color in zip(targets, colors):
                subset = handler.query(gids)
                subset.attributes.color = color
                subset.update()

    def _colorize_synapses(self):
        handlers = self._synapse_handlers
        if not handlers:
            return

        if self._mode == self.Coloring.SOLID:
            for key, handler in handlers.items():
                handler.attributes.color = self._synapse_solid_color
                handler.attributes.visible = True
                handler.update()
        else:
            dummy, targets, colors = self._get_coloring_info(self._mode)

            for gids, color in zip(targets, colors):
                visible = _np.intersect1d(gids, self._unmasked)
                for gid in visible:
                    handler = handlers[gid]
                    handler.attributes.color = color
                    handler.attributes.visible = True
                    handler.update()
                invisible = _np.intersect1d(gids, self._masked)
                for gid in invisible:
                    handler = handlers[gid]
                    handler.attributes.visible = False
                    handler.update()

    def _init_masks(self):

        if not self._cell_handler:
            # The legend doesn't take effect if the pair is fully selected
            self._unmasked = _np.zeros((0), dtype="u4")
            self._masked = _np.zeros((0), dtype="u4")
            return

        selected = _np.zeros((0), dtype="u4")
        for item in self._selected_legend_items:
            selected = _np.union1d(selected,
                                   self._color_lists[self._mode][1][item])

        target = self._cell_handler.object

        if selected.size == 0:
            # When nothing is selected in the legend all the cells from
            # handler are unmasked
            self._unmasked = target
            self._masked = _np.zeros((0), dtype="u4")
        else:
            self._unmasked = _np.intersect1d(selected, target)
            self._masked = _np.setdiff1d(target, selected)

        self.filter_mask_changed.emit(self._masked)

    def _on_coloring_changed(self, mode):
        Coloring = self.Coloring

        mode = Coloring.NAMES.index(mode)
        if mode == self._mode:
            return # Nothing to do

        if mode == Coloring.LAYER:
            if Coloring.LAYER not in self._color_lists:
                mode = Coloring.SOLID
                print("Per layer colors not available in the current circuit")

        self._mode = mode

        # Resetting the masking
        self._selected_legend_items = set()
        self._init_masks()

        # Updating the legend
        if self._mode != Coloring.SOLID:
            model = SimpleSelectableListModel()
            labels, dummy, colors = self._get_coloring_info(self._mode)

            legend_info = []
            for label, rgba in zip(labels, colors):
                color = "#%.2x%.2x%.2x%.2x" % (
                    int(rgba[3] * 255), int(rgba[0] * 255),
                    int(rgba[1] * 255), int(rgba[2] * 255))
                legend_info.append((label, color))
            model.set(legend_info)
            self._qml_legend.setModel(model)
            model.item_selection_changed.connect(
                self._on_legend_item_selection_changed)
            # we need to keep a reference to this object because qml_legend
            # is just connecting signals
            self._model = model

        self.colorize()

    def _on_legend_item_selection_changed(self, item, selected):

        if not self._cell_handler:
            # If there's a pair selected, we need at least to keep track of
            # what is being selected on the legend
            if selected:
                self._selected_legend_items.add(item)
            else:
                self._selected_legend_items.remove(item)
            return

        selected_target = _np.intersect1d(
            self._color_lists[self._mode][1][item],
            _np.union1d(self._masked, self._unmasked))

        # Handling the special cases first

        if len(self._selected_legend_items) == 0:
            assert(selected)
            assert(self._masked.size == 0)
            # We only need to make cells and synapses invisible in this case
            self._masked = _np.setdiff1d(self._unmasked, selected_target)
            self._unmasked = selected_target

            # Updating cells
            invisible = self._cell_handler.query(self._masked)
            invisible.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            invisible.update()
            # and then synapses
            if self._synapse_handlers:
                for gid in self._masked:
                    handler = self._synapse_handlers[gid]
                    handler.attributes.visible = False
                    handler.update()

            self._selected_legend_items.add(item)
            self.filter_mask_changed.emit(self._masked)
            return

        if len(self._selected_legend_items) == 1 and not selected:
            assert(item in self._selected_legend_items)
            # Reverting to all visible. We have to colorize everything back
            # because the subhandlers that were used for that purpose are
            # thrown away.
            self._unmasked = _np.union1d(self._unmasked, self._masked)
            self._masked = _np.zeros((0), dtype="u4")
            self._selected_legend_items = set()
            self.colorize()
            self.filter_mask_changed.emit(self._masked)
            return

        # In the rest of the cases we have to update the masks and then
        # hide or show cells.
        subset = self._cell_handler.query(selected_target)
        if selected:
            self._masked = _np.setdiff1d(self._masked, selected_target)
            self._unmasked = _np.union1d(self._unmasked, selected_target)
            # If there's a legend, we are not using a solid color
            assert(self._mode != self.Coloring.SOLID)
            color = self._color_lists[self._mode][2][item]
            subset.attributes.mode = \
                self._display_mode if self._cells_visible \
                else _rtneuron.RepresentationMode.NO_DISPLAY
            subset.attributes.color = color
            self._selected_legend_items.add(item)
        else:
            self._masked = _np.union1d(self._masked, selected_target)
            self._unmasked = _np.setdiff1d(self._unmasked, selected_target)
            subset.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
            self._selected_legend_items.remove(item)
        subset.update()

        # And finally we hide or show synapses of the selected target
        if self._synapse_handlers:
            for gid in selected_target:
                handler = self._synapse_handlers[gid]
                handler.attributes.visible = selected
                handler.update()

        self.filter_mask_changed.emit(self._masked)

class SynapseAnnotations(object):

    class Annotation(QMLAnnotation):

        width = 250

        def __init__(self, synapse, connection_side, cell_pair, gui, camera):
            super(SynapseAnnotations.Annotation, self).__init__(gui, camera)

            post_gid = synapse.post_gid()
            pre_gid = synapse.pre_gid()

            properties = _synaptic_properties(
                cell_pair.morphologies[GUI.POSTSYNAPTIC_CELL])

            self.qml.setProperty("width", self.width)

            if connection_side == "post":
                try :
                    position = synapse.post_surface_position()
                except:
                    position = _compute_postynaptic_position(cell_pair, synapse)

                self.qml.setProperty(
                    "headerText", "Post-GID: %d, idx: %d" % synapse.gid())
            else:
                try :
                    position = synapse.pre_surface_position()
                except:
                    position = _compute_preynaptic_position(cell_pair, synapse)

                # The synapse idx in the post-synaptic set is unavailable
                # because we have loaded from the efferent view.
                self.qml.setProperty(
                    "headerText", "Post-GID: %d, idx: unavailable" %
                    synapse.post_gid())

            text = ""
            for name, unit, functor in properties:
                text += " %s: %.4f %s\n" % (name, functor(synapse), unit)
            self.qml.setProperty("text", text)

            self.track_point(position)
            self.gid = SynapseAnnotations._synapse_id(synapse)

            self._set_initial_position(gui.background())

        def _set_initial_position(self, glwidget):
            height = self.qml.property("height")
            # Placing the annotations at a reasonable initial position
            p = self._compute_projected_point(glwidget)

            min_distance = 50
            if p.x() > self.width + min_distance:
                x = max(p.x() - self.width - 100, 0)
            else:
                x = min(p.x() + 100, glwidget.rect().width() - self.width)
            self.qml.setProperty("x", x)

            if p.y() > height + min_distance:
                y = max(p.y() - height - min_distance, 0)
            else:
                y = min(p.y() + min_distance, glwidget.rect().height() - height)
            self.qml.setProperty("y", y)


    def __init__(self, gui, view):
        self._gui = _weakref.ref(gui)
        self._camera = view.camera
        self._circuit = view.scene.circuit
        self._annotations = dict()

    def add(self, synapse, cell_pair, connection_side):
        gid = self._synapse_id(synapse)
        if gid in self._annotations:
            return

        annotation = self.Annotation(synapse, connection_side, cell_pair,
                                     self._gui(), self._camera)
        annotation.closed.connect(self._on_annotation_closed)
        self._annotations[gid] = annotation

    def clear(self):
        for gid, annotation in self._annotations.items():
            annotation.remove(self._gui())
        self._annotations = dict()

    def expand_all(self):
        for annotation in self._annotations.values():
            annotation.qml.expand()

    @QtCore.pyqtSlot(object)
    def _on_annotation_closed(self, annotation):
        annotation.remove(self._gui())
        del self._annotations[annotation.gid]

    @staticmethod
    def _synapse_id(synapse):
        # We use the presynaptic side because for soma synapses the
        # postsynaptic_cell can be ambiguous
        return (synapse.pre_gid(), synapse.pre_section(),
                synapse.pre_segment(), synapse.pre_distance())

class GUI(LoaderGUI):

    PRESYNAPTIC_CELL = 0
    POSTSYNAPTIC_CELL = 1

    _clipping_enabled = QtCore.pyqtSignal(bool)
    _connected_cells_hideable = QtCore.pyqtSignal(bool)
    _annotate_synapse = QtCore.pyqtSignal(_brain.Synapse)
    _update_connection_info = QtCore.pyqtSignal()
    _cell_selected = QtCore.pyqtSignal(int, int, int)

    def __init__(self, *args, **kwargs):

        # Changing the default configuration to one that provides efferent
        # and afferent synapses (unless one was provided)
        if not 'simulation_config' in kwargs:
            try:
                kwargs['simulation_config'] = _brain.test.circuit_config
            except AttributeError:
                pass

        super(GUI, self).__init__(
            _os.path.dirname(_os.path.realpath(__file__)) +
            '/ConnectionViewer.qml',
            *args, **kwargs)

        self._keyHandler = KeyHandler()
        self.background().interaction_mode = self._keyHandler

        qml = self._overlay.rootObject()
        # Connecting render options signals
        qml.resetClicked.connect(self._on_reset_clicked)
        qml.inflationFactorChanged.connect(self._on_inflation_factor_changed)
        qml.synapseRadiusChanged.connect(self._on_synapse_radius_changed)
        qml.clipNeurons.connect(self._on_clip_neurons)
        qml.connectedCellsVisible.connect(self._on_connected_cells_visible)
        self._clipping_enabled.connect(qml.enableClipButton)
        qml.enableClipButton(False)
        self._connected_cells_hideable.connect(qml.enableShowHideButton)
        qml.enableShowHideButton(False)
        # Cell selection signals
        qml.activatePicking.connect(self._on_activate_picking)
        qml.cellEntered.connect(self._on_cell_entered)
        # Other signals
        qml.expandAnnotationsClicked.connect(self._on_expand_annotations)

        self._connection_info = SimpleSelectableListModel()
        qml.setConnectionInfoModel(self._connection_info)

        # This signals are needed to make sure that functions that need to be
        # invoked from scene callbacks and modify the GUI run in the GUI thread.
        self._annotate_synapse.connect(self._on_annotate_synapse)
        self._update_connection_info.connect(self._on_update_connection_info)
        self._cell_selected.connect(self._on_cell_selected)

        self._cell_pair = CellPair()
        self._cell_to_pick = GUI.POSTSYNAPTIC_CELL
        self._clip_cells = False
        self._show_connected_cells = True
        self._base_selection_mask = _np.zeros((0), dtype="u4")

        self._cell_colors = [[0.3, 0.6, 1, 1], [1, 0.3, 0.3, 1]]
        self._synapse_colors = [[0.5, 0.8, 1, 1], [1, 0.5, 0.5, 1]]
        self._synapse_radius = 2.5
        self._synapses = None
        self._synapse_handlers = dict()

        # Setting the default message for the connection info
        self._on_update_connection_info()

    def _connect_engine_implementation(self, engine):
        self._engine = engine

    def scene_created(self, scene):
        self._scene = scene
        self._annotations = SynapseAnnotations(self, self._engine.views[0])
        neurons = scene.objects[0]
        self._all_gids = neurons.object
        self._default_display_mode = neurons.attributes.mode
        self._default_color = neurons.attributes.primary_color
        scene.cellSelected.connect(self._cell_selected.emit)
        scene.synapseSelected.connect(self._on_synapse_selected)

        self._coloring_handler = \
            TargetDisplayHandler(self._scene, _rtneuron.simulation,
                                 self._overlay.rootObject())
        self._coloring_handler.filter_mask_changed.connect(
            self._on_filter_mask_changed)

    def _on_inflation_factor_changed(self, factor):
        view = _rtneuron.engine.views[0]
        view.attributes.inflation_factor = factor

    def _on_synapse_radius_changed(self, radius):
        self._synapse_radius = radius

        _rtneuron.engine.pause()
        for key, handler in self._synapse_handlers.items():
            handler.attributes.radius = radius
            handler.update()
        _rtneuron.engine.resume()

    def _on_reset_clicked(self):
        self._reset(True)
        qml = self._overlay.rootObject()
        qml.reset()
        qml.activePicking("post")

        # Resetting the camera home position
        view = _rtneuron.engine.views[0]
        view.computeHomePosition()

    def _on_cell_selected(self, gid, section, segment):

        to_pick = self._cell_to_pick
        if to_pick == None:
            return

        # Deciding whether the cell picked is the first or the second cell
        # of the synaptic pair
        if not self._cell_pair.handlers[1 - to_pick]:
            self._pick_first_cell(gid)
        else:
            self._pick_second_cell(gid)

        # Updating the interface with the GID of the pick cell and changing
        # the activation of the pick buttons
        qml = self._overlay.rootObject()
        qml.setCellRoleGID(self._role_name(to_pick), gid)
        qml.activePicking(self._role_name(self._cell_to_pick))

    def _on_synapse_selected(self, synapse):

        if self._cell_to_pick == None or self._keyHandler.shiftPressed:
            self._annotate_synapse.emit(synapse)
            return

        assert(self._cell_pair.handlers[0] or self._cell_pair.handlers[1])

        # Selecting the second cell from the pair from the synapses
        if self._cell_to_pick == self.PRESYNAPTIC_CELL:
            gid = synapse.pre_gid()
        else:
            gid = synapse.post_gid()
        self._cell_selected.emit(gid, 0, 0)

    def _on_clip_neurons(self, clip):
        self._clip_cells = clip
        if all(self._cell_pair.handlers):
            if clip:
                self._clip_pair()
            else:
                self._unclip_pair()

    def _on_connected_cells_visible(self, visible):
        self._show_connected_cells = visible
        # If one cell is already picked we set the visiblity of the connected
        # set
        if not self._cell_pair.handlers[self._cell_to_pick]:
            self._coloring_handler.set_cells_visible(visible)

    def _on_activate_picking(self, cell_role):

        if cell_role == "pre":
            to_pick = self.PRESYNAPTIC_CELL
        else:
            to_pick = self.POSTSYNAPTIC_CELL

        if to_pick == self._cell_to_pick:
            return

        # If the cell to pick is empty there is no need to do anything
        # regardless of the other cell being already picked or not.
        if self._cell_pair.handlers[to_pick] == None:
            self._cell_to_pick = to_pick
            return

        self._overlay.rootObject().setCellRoleGID(cell_role, -1)

        # If only one cell had been picked so far an it has the same role
        # of the cell for which picking is going to be activated, this is
        # basically a reset.
        if self._cell_pair.handlers[1 - to_pick] == None:
            self._cell_to_pick = to_pick
            self._reset(True)
            return

        # Both cells are picked. We want to make available all the cells
        # connected to the cell of the comlementary role (1 - to_pick).
        # The easiest way to proceed is to reset the scene to remove the
        # synapses and reuse the code for picking the first cell.
        first_gid = self._cell_pair.gids[1 - to_pick]
        self._reset() # Do not move above the line up
        self._cell_to_pick = 1 - to_pick
        self._pick_first_cell(first_gid)

    def _on_cell_entered(self, cell_role, gid):
        if cell_role == "pre":
            to_pick = self.PRESYNAPTIC_CELL
        else:
            to_pick = self.POSTSYNAPTIC_CELL

        def find_gid(gids, gid):
            i = gids.searchsorted(gid)
            return i < len(gids) and gids[i] == gid

        def warn_invalid_gid():
            dialog = AcceptDialog(self._overlay)
            dialog.set_message(
                "The cell with GID %d is not part of the scene" % gid)
            dialog.set_button_labels("", "OK")
            dialog.run()

        if self._cell_to_pick == to_pick:

            # We will do the selection as if it came with the mouse after
            # checking the boundary conditions.

            if not find_gid(self._all_gids, gid):
                warn_invalid_gid()
                self._overlay.rootObject().setCellRoleGID(cell_role, -1)
                return

            # Checking that the cell is in the set of cells connected to
            # the already picked (if there's already one picked)
            # If that's not the case we will ask the user what to do.
            if (self._cell_pair.handlers[1 - to_pick] != None and
                not find_gid(self._connected_cells.object, gid)):

                qml = self._overlay.rootObject()
                dialog = AcceptDialog(self._overlay)
                dialog.set_message(
                    "Cell GID %d is not connected to the first cell picked."
                    " Do you want to reset the scene and pick %d as the first"
                    " cell?" % (gid, gid))
                accepted = dialog.run()
                if accepted:
                    self._reset()
                    qml.setCellRoleGID(self._role_name(1 - to_pick), -1)
                    self._cell_to_pick = to_pick
                    self._on_cell_selected(gid, 0, 0)
                else:
                    # The content of the input line needs clearing.
                    qml.setCellRoleGID(cell_role, -1)
                return

            self._on_cell_selected(gid, 0, 0)

        else:
            if not find_gid(self._all_gids, gid):
                warn_invalid_gid()
                # Restoring the GID
                gid = self._cell_pair.gids[to_pick]
                if gid is None:
                    gid = -1
                self._overlay.rootObject().setCellRoleGID(cell_role, gid)
                return

            # If both cells are selected then we reset the scene to pick the
            # entered cell as the first cell.
            if all(self._cell_pair.handlers):
                self._reset()

            self._cell_to_pick = to_pick
            self._on_cell_selected(gid, 0, 0)

    def _on_update_connection_info(self):
        info = []

        def add_cell_info(role, prefix):
            gid = self._cell_pair.gids[role]
            if gid is None:
                return
            circuit = self._scene.circuit
            # Not very efficient, but it's not critical
            mtype = circuit.morphology_type_names()[
                circuit.morphology_types([gid])[0]]
            etype = circuit.electrophysiology_type_names()[
                circuit.electrophysiology_types([gid])[0]]

            info.append(
                (prefix + "synaptic cell", "%d %s %s" % (gid, etype, mtype)))

        add_cell_info(self.PRESYNAPTIC_CELL, "Pre")
        add_cell_info(self.POSTSYNAPTIC_CELL, "Post")

        if not all(self._cell_pair.handlers):
            self._connection_info.set(info)
            return

        # Computing the synaptic properties of the connection
        count = len(self._synapses)
        info.append(("Number of synapses", count))

        properties = _synaptic_properties(
            self._cell_pair.morphologies[self.POSTSYNAPTIC_CELL])
        statistics = []
        info.append(("Average values (\u00B1 std. deviation)", ""))
        for name, unit, functor in properties:
            values = _np.array([functor(s) for s in self._synapses])
            info.append(("  " + name, " %.4f %s (\u00B1 %.4f)" % (
                                          values.mean(), unit, values.std())))

        self._connection_info.set(info)

    def _on_annotate_synapse(self, synapse):
        side = "pre" if self._cell_to_pick == self.POSTSYNAPTIC_CELL else "post"

        if not(all(self._cell_pair.handlers)):
            # We don't have all the info yet
            cell_pair = CellPair()
            circuit = self._scene.circuit
            gids = [synapse.pre_gid(), synapse.post_gid()]
            cell_pair.gids = gids
            cell_pair.morphologies = \
                circuit.load_morphologies(gids, circuit.Coordinates.local)
            cell_pair.transforms = \
                circuit.load_morphologies(gids, circuit.Coordinates.local)
        else:
            cell_pair = self._cell_pair

        self._annotations.add(synapse, cell_pair, side)

    def _on_expand_annotations(self):
        self._annotations.expand_all()

    def _on_filter_mask_changed(self, mask):
        self._scene.neuronSelectionMask = \
            _np.union1d(self._base_selection_mask, mask)

    def _pick_first_cell(self, gid):

        to_pick = self._cell_to_pick
        def other(cell_to_pick):
            return 1 - cell_to_pick

        neurons = self._scene.objects[0]

        # Loading synapses
        circuit = self._scene.circuit
        if to_pick == self.POSTSYNAPTIC_CELL:
            synapses = circuit.afferent_synapses([gid])
        else:
            synapses = circuit.efferent_synapses([gid])
        self._synapses = synapses

        # Hiding all the cells
        neurons.attributes.mode = _rtneuron.RepresentationMode.NO_DISPLAY
        neurons.update()

        # Highlighting the cell picked and displaying it in detailed mode.
        selected = neurons.query([gid])
        selected.attributes.mode = \
            _rtneuron.RepresentationMode.WHOLE_NEURON
        selected.attributes.color_scheme = _rtneuron.ColorScheme.SOLID
        selected.attributes.primary_color = self._cell_colors[to_pick]
        selected.update()
        # Make sure the cell is unclipped at this point.
        selected.apply(_NeuronClipping().unclipAll())

        # Finding the connected cells and grouping the synapses per connection.
        if to_pick == self.PRESYNAPTIC_CELL:
            connected = synapses.post_gids()
        else:
            connected = synapses.pre_gids()
        connected = _np.unique(connected)

        connections = dict()

        for i in connected:
            pre = [gid if to_pick == self.PRESYNAPTIC_CELL else int(i)]
            post = [int(i) if to_pick == self.PRESYNAPTIC_CELL else gid]
            connections[i] = circuit.projected_synapses(pre, post)

        # Setting the mask for making only the connected cells selectable.
        mask = _np.setdiff1d(neurons.object, connected)
        self._base_selection_mask = mask
        connected = neurons.query(connected)

        # Adding one synapes set per connection. Doing this instead
        # of adding a single set makes it easier to apply a different color
        # to each synapse subset.
        functors = [_rtneuron.Scene.addEfferentSynapses,
                    _rtneuron.Scene.addAfferentSynapses]
        attributes = _rtneuron.AttributeMap()
        attributes.radius = self._synapse_radius
        self._synapse_handlers = dict()
        _rtneuron.engine.pause()
        for key, synapses in connections.items():
            self._synapse_handlers[key] = \
                    functors[to_pick](self._scene, synapses, attributes)
        _rtneuron.engine.resume()

        # Applying the coloring based on the user selection
        self._coloring_handler.set_target_cells(
            connected, self._cell_colors[other(to_pick)],
            visible=self._show_connected_cells)
        self._coloring_handler.set_target_synapses(
            self._synapse_handlers, self._synapse_colors[to_pick])
        self._coloring_handler.colorize()

        # Updating state variables
        self._cell_pair.handlers[to_pick] = selected
        self._cell_pair.gids[to_pick] = gid
        self._connected_cells = connected
        self._cell_to_pick = other(to_pick)
        self._update_connection_info.emit()
        self._connected_cells_hideable.emit(True)

        # Setting the new home position for the view
        position = list(map(float, circuit.positions([gid])[0]))
        view = _rtneuron.engine.views[0]
        view.cameraManipulator.setHomePosition(
            view.camera.getView()[0], position, [0, 1, 0])

    def _pick_second_cell(self, gid):

        to_pick = self._cell_to_pick
        neurons = self._scene.objects[0]

        # No cell remains selectable after this step.
        self._base_selection_mask = self._all_gids

        selected = neurons.query([gid])
        handlers = self._cell_pair.handlers
        handlers[to_pick] = selected
        self._cell_pair.gids[to_pick] = gid

        # Loading the morphologies and transformations
        circuit = self._scene.circuit
        gids = self._cell_pair.gids
        self._cell_pair.morphologies = \
            circuit.load_morphologies(gids, circuit.Coordinates.local)
        self._cell_pair.transforms = circuit.transforms(gids)

        self._synapses = self._synapse_handlers[gid].object
        self._display_connection_synapses(self._synapses)

        # When the afferent/efferent synapses of the first cell picked are
        # removed from the scene, it triggers the recreation of the scene and
        # the attributes of the whole circuit handler are reapplied to all
        # subsets. Thus, we need to reset the attributes of the first cell.
        first = handlers[1 - to_pick]
        first.attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
        first.attributes.color_scheme = _rtneuron.ColorScheme.SOLID
        first.attributes.primary_color = self._cell_colors[1 - to_pick]
        first.update()

        # And then highlight the selected one again.
        selected.attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
        selected.attributes.color_scheme = _rtneuron.ColorScheme.SOLID
        selected.attributes.primary_color = self._cell_colors[to_pick]
        selected.update()

        del self._connected_cells
        self._cell_to_pick = None
        self._coloring_handler.set_target_cells(None, None)
        self._coloring_handler.set_target_synapses(None, None)

        self._update_connection_info.emit()
        self._clipping_enabled.emit(True)
        self._connected_cells_hideable.emit(False)

        # Applying current clipping to cells
        if self._clip_cells:
            self._clip_pair()
        else:
            self._unclip_pair()

        # Setting the new home position for the view
        view = _rtneuron.engine.views[0]
        centroid = _np.mean(self._cell_pair.transforms[:,:-1,3], axis=0)
        view.cameraManipulator.setHomePosition(
            view.camera.getView()[0], list(map(float, centroid)), [0, 1, 0])

    def _display_connection_synapses(self, synapses):

        self._remove_all_synapses()
        self._engine.frame()

        pre, post = self._cell_pair.gids
        attributes = _rtneuron.AttributeMap()
        attributes.radius = self._synapse_radius

        attributes.color = self._synapse_colors[0]
        self._synapse_handlers[pre] = \
            self._scene.addEfferentSynapses(synapses, attributes)\

        attributes.color = self._synapse_colors[1]
        self._synapse_handlers[post] = \
            self._scene.addAfferentSynapses(synapses, attributes)

    def _clip_pair(self):

        handlers = self._cell_pair.handlers
        assert(all(handlers))
        gids = self._cell_pair.gids
        morphologies = self._cell_pair.morphologies

        handlers[0].apply(_NeuronClipping().clipAll().unclipEfferentBranches(
            gids[0], morphologies[0], self._synapses))
        handlers[1].apply(_NeuronClipping().clipAll().unclipAfferentBranches(
            gids[1], morphologies[1], self._synapses))

    def _unclip_pair(self):
        for handler in self._cell_pair.handlers:
            handler.apply(_NeuronClipping().unclipAll())

    def _remove_all_synapses(self):
        for gid, handler in self._synapse_handlers.items():
            self._scene.remove(handler)
        self._synapse_handlers = dict()
        self._annotations.clear()

    def _reset(self, colorize=False):
        self._base_selection_mask = _np.zeros((0), dtype="u4")
        objects = self._scene.objects

        self._remove_all_synapses()
        _rtneuron.engine.frame()

        if colorize == True:
            neurons = objects[0]
            self._coloring_handler.set_target_cells(
                neurons, neurons.attributes.primary_color)
            self._coloring_handler.set_target_synapses(None, None)
            self._coloring_handler.colorize()

        self._cell_pair.reset()
        self._synapses = None
        self._clipping_enabled.emit(False)
        self._connected_cells_hideable.emit(False)
        self._on_update_connection_info()

    def _role_name(self, to_pick):
        if to_pick == self.PRESYNAPTIC_CELL:
            return "pre"
        elif to_pick == self.POSTSYNAPTIC_CELL:
            return "post"
        else:
            return "none"

class App(object):
    """Power application for the analysis of connections between
       pairs of cells.

       This application allows the user to load a circuit and target set and
       then choose pairs of pre and postsynaptic cells to show the synapses
       that connect them.

       Cells can be picked by clicking on their somas with the mouse or
       typing their GIDs.
       It is possible to pick one of the cells of the pair first, which will
       reduce the set of visible and pickable cells to only those that are
       connected. The cells can be further filtered by choosing a coloring mode
       different from the default and then clicking on the legend to choose
       the sets of interest.
    """

    def __init__(self, *args, **kwargs):

        attributes = _rtneuron.AttributeMap()
        # Don't use meshes because branch level culling of the meshes from the
        # default circuit provides terrible results on meshes.
        attributes.use_meshes = False
        # We want circuit building to assume that there are no morphologies to
        # speed it up.
        attributes.load_morphologies = False
        attributes.inflatable_neurons = True

        self._gui = display_empty_scene_with_GUI(
            GUI, attributes, *args, **kwargs)

        view = _rtneuron.engine.views[0]
        view.attributes.auto_compute_home_position = False
        view.attributes.auto_adjust_model_scale = False
        view.attributes.background = [1, 1, 1, 1]
        view.attributes.highlight_color = [1, 0.8, 0, 1]

    def pair(self):
        """Return the pair of cells selected or None if no pair has been
        selected yet."""
        if not all(self._gui._cell_pair.handlers):
            return None
        return self._gui._cell_pair

    def synapses(self):
        """Return the synapses connecting the selected pair or None if no pair
        has been selected yet."""
        return self._gui._synapses

def start(*args, **kwargs):
    """Startup the application for browsing connection pairs.

    Arguments:
    - simulation_config: Path to the default blue config to use in the loader
      menu
    - target: Default target string to use in the loader menu
    """
    return App(*args, **kwargs)
