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

import math as _math
import numpy as _np
import os as _os

import rtneuron as _rtneuron
from rtneuron.gui import LoaderGUI, SelectionHandler, \
                         display_empty_scene_with_GUI

from ..sceneops import SynapticProjections, enable_transparency

import brain as _brain

class GUI(LoaderGUI):
    def __init__(self, *args, **kwargs):

        # Changing the default configuration to one that provides efferent
        # and afferent synapses (unless one was provided)
        if not 'simulation_config' in kwargs:
            try:
                kwargs['simulation_config'] = _brain.test.circuit_config
            except AttributeError:
                pass

        super(GUI, self).__init__(
            _os.path.dirname(_os.path.realpath(__file__)) + '/Projections.qml',
            *args, **kwargs)

        qml = self._overlay.rootObject()
        qml.inflationFactorChanged.connect(self._on_inflation_factor_changed)
        qml.presynapticOpacityChanged.connect(
            self._on_presynaptic_opacity_changed)
        qml.postsynapticOpacityChanged.connect(
            self._on_postsynaptic_opacity_changed)
        qml.circuitOpacityChanged.connect(self._on_circuit_opacity_changed)
        qml.clearClicked.connect(self._on_clear_clicked)

        if _rtneuron.options.sync_selections:
            session = _rtneuron.options.sync_selections
            if session == True:
                self._broker = _rtneuron.net.SceneEventBroker()
            else:
                self._broker = _rtneuron.net.SceneEventBroker(session)
            self._broker.trackState = _rtneuron.options.track_selections
        else:
            self._broker = None

        self._presynaptic_color = [0, 0, 1, 0.5]
        self._postsynaptic_color = [1, 0, 0, 0.5]
        self._unselected_color = [0.5, 0.5, 0.5, 0.5]

    def _init_implementation(self):
        pass

    def scene_created(self, scene):

        scene = _rtneuron.engine.views[0].scene

        # Creating the object that manipulates the scene objects
        self.projections = SynapticProjections(
            scene, presynaptic_color=self._presynaptic_color,
            postsynaptic_color=self._postsynaptic_color,
            unselected_color=self._unselected_color,
            target_mode=_rtneuron.RepresentationMode.WHOLE_NEURON,
            clip_branches=True)

        # If there's a SceneEventBroker it means that the synaptic projections
        # to display are going to be requested externally.
        # The selection handlers from SynapticProjections will be removed to
        # use a SelectionHandler instead.
        if self._broker:
            scene.cellSelected.disconnectAll()
            scene.cellSetSelected.disconnectAll()
            self._selection_handler = \
                SelectionHandler(_rtneuron.engine.views[0],
                                 self._background, self._broker)

            self._broker.cellSetBinaryOp.connect(self._on_cell_set_operation)


        # Enabling alpha blending now. This way, the first frames, which contain
        # detailed neurons, will render faster.
        enable_transparency(scene)

    def _on_inflation_factor_changed(self, factor):
        view = _rtneuron.engine.views[0]
        view.attributes.inflation_factor = factor

    def _on_presynaptic_opacity_changed(self, opacity):
        self._presynaptic_color[3] = opacity
        self.projections.set_presynaptic_color(self._presynaptic_color)

    def _on_postsynaptic_opacity_changed(self, opacity):
        self._postsynaptic_color[3] = opacity
        self.projections.set_postsynaptic_color(self._postsynaptic_color)

    def _on_circuit_opacity_changed(self, opacity):
        self._unselected_color[3] = opacity
        self.projections.set_unselected_color(self._unselected_color)

    def _on_cell_set_operation(self, pre_target, post_target, operation):
        if operation == _rtneuron.net.CellSetBinaryOpType.SYNAPTIC_PROJECTIONS:
            scene = _rtneuron.engine.views[0].scene
            gids = scene.objects[0].object
            mask = _np.setdiff1d(
                _np.setdiff1d(_neurons, pre_target), post_target)
            scene.neuronSelectionMask = mask
            self.projections.show_projections(pre_target, post_target)

    def _on_clear_clicked(self):
        scene = _rtneuron.engine.views[0].scene
        scene.neuronSelectionMask = _np.zeros((0), dtype="u4")
        self.projections.clear()

class Demo(object):
    """Synaptic projections demo.

       This application allows the user to load a circuit and target set and
       then pick cells to see their anterograde and retrograde connections.

       See rtneuron.sceneops.SynapticProjections for details on how synaptic
       pathways are displayed.
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
        view.attributes.background = [1, 1, 1, 1]
        view.attributes.highlight_color = [1, 0.8, 0, 1]

    def projections(self):
        try:
            return self._gui.projections
        except NameError:
            return None

def start(*args, **kwargs):
    """Startup the synaptic projections demo.

    Arguments:
    - simulation_config: Path to the default simulation config to use in the
      loader menu
    - target: Default target string to use in the loader menu
    """
    return Demo(*args, **kwargs)
