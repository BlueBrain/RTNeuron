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

# For Python3 forward compatiblity
from __future__ import print_function

import brain as _brain
from .. import _rtneuron
from . import NeuronClipping
from . import util as _util

import numpy as _np

# The unselected color alpha is multiplied by this factor and used as the
# alpha channel of dimmed colors.
_ALPHA_MULTIPLIER = 4.0

def _set_attributes(handlers, attributes):
    if type(handlers) != list:
        handlers = [handlers]

    for handler in handlers:
        for attribute in dir(attributes):
            handler.attributes.__setattr__(attribute,
                                           attributes.__getattr__(attribute))
        handler.update()

def _find_neuron(handlers, gid):
    """Find a neuron in a list of object handlers and return a temporary
    handler for it. Return None if the neuron is not found.
    Return a tuple (neuron, handler) if the neuron is found, None otherwise.
    """
    for handler in handlers:
        if _util.is_neuron_handler(handler):
            # Converting to int in case gid is numpy.int32
            handler = handler.query([int(gid)])
            if len(handler.object) == 1:
                return handler
    return None

def _find_handler(handlers, gid):
    """Find a neuron in a list of object handlers and return a temporary
    handler for it.
    """
    for handler in handlers:
        if _util.is_neuron_handler(handler):
            subhandler = handler.query([gid])
            if subhandler.object.size() == 1:
                # No need to keep searching as neurons can only be present in a
                # unique object handler.
                return handler
    return None

def _get_subhandlers(gids, handlers):
    subhandlers = []
    for handler in handlers:
        subhandler = handler.query(gids)
        if len(subhandler.object) == 0:
            continue
        subhandlers.append(subhandler)
    return subhandlers

class SynapticProjections:
    """This class provides functions to show synaptic projections in a
    given scene/microcircuit.

    It provides an easy way to display retrograde and anterograde projections
    for cells selected in the scene and a method to display the synaptic
    pathways from a pre-synaptic target to a post-synaptic target.

    A callback is hooked to the cellSelected signal from a scene to show:

    - No cell selected, all displayed with somas
    - Retrograde projections: A post-synaptic cell and its pre-synaptic cells
      with the selected representation mode and colors
    - Anterograde projections: A pre-synaptic cell and its post-synaptic cells
      with the selected representation mode and colors

    The representation modes for pre/post synaptic sets are:

    - Soma only
    - Whole detailed neurons
    - Detailed neuron with branch-level clipping to show only the portions
      of the branches that run along the path that connects the soma of the
      presynaptic cell to the soma of the postsynaptic cell through each
      synapse.

    For synaptic projections between two sets, the detailed representation with
    branch level culling is always used.
    """
    def __init__(self, scene,
                 presynaptic_color=[0.0, 0.5, 1.0, 1.0],
                 postsynaptic_color=[1.0, 0.0, 0.0, 1.0],
                 unselected_color=[0, 0, 0, 0.1],
                 target_mode=_rtneuron.RepresentationMode.WHOLE_NEURON,
                 clip_branches=True):
        """Store the circuit and scene information and hook a callback to
        the scene.cellCelected signal to switch between the different
        synaptic projections modes.

        Parameters:

        - presynaptic_color: Color to use for presynaptic cells
        - postsynaptic_color: Color to use for postsynaptic cells
        - unselected_color: Color to use for cells which are not part of
          the anterograde or retrograde set of a selected cell.
        - target_mode: Representation mode to use for the anterograde/retrograde
          cell set
        - clip_branches: target_mode is WHOLE_NEURON, apply fine-grained
          clipping to branches to highlight only the paths that connect the
          pre and post-synaptic somas through each synapse.
        """
        if scene.circuit is None:
            raise ValueError("The scene must have a circuit assigned")
        self._circuit = scene.circuit
        self._scene = scene

        self._presynaptic_color = presynaptic_color
        self._postsynaptic_color = postsynaptic_color
        self._unselected_color = unselected_color
        self._target_mode = target_mode
        self._clip_branches = clip_branches and \
            target_mode == _rtneuron.RepresentationMode.WHOLE_NEURON

        self._num_clicks = 0

        class CellTargets:
            def __init__(self):
                self.reset()

            def reset(self):
                self.post_target = _np.zeros((0), dtype="u4")
                self.pre_target = _np.zeros((0), dtype="u4")
                self.pre_and_post_target = _np.zeros((0), dtype="u4")
                self.post_handlers = []
                self.pre_handlers = []
                self.pre_and_post_handlers = []

            def find_pre_subhandlers(self, handlers):
                self.pre_handlers = _get_subhandlers(self.pre_target, handlers)

            def find_post_subhandlers(self, handlers):
                self.post_handlers = \
                    _get_subhandlers(self.post_target, handlers)

            def find_pre_and_post_subhandlers(self, handlers):
                self.pre_and_post_handlers = \
                    _get_subhandlers(self.pre_and_post_target, handlers)

            def find_subhandlers(self, handlers):
                self.find_pre_subhandlers(handlers)
                self.find_post_subhandlers(handlers)
                self.find_pre_and_post_subhandlers(handlers)

        self._connected_cells = CellTargets()
        self._disconnected_cells = CellTargets()

        self.clear()

        scene.cellSelected.disconnectAll()
        scene.cellSelected.connect(self._cell_selected)

    def show_projections(self, presynaptic=None, postsynaptic=None):

        gids, handlers = _util.get_neurons(self._scene)
        if len(gids) == 0:
            return
        # Reset attributes
        _set_attributes(handlers, self._unselected_attributes())

        unselected_cells = gids

        if presynaptic is None:
            presynaptic = gids
        else:
            presynaptic = _np.intersect1d(presynaptic, gids)

        if postsynaptic is None:
            postsynaptic = presynaptic
            pre_only = _np.zeros((0), dytpe="u4")
            post_only = _np.zeros((0), dytpe="u4")
            common = presynaptic
            full_target = presynaptic
        else:
            presynaptic = _np.intersect1d(presynaptic, gids)
            pre_only = _np.setdiff1d(presynaptic, postsynaptic)
            post_only = _np.setdiff1d(postsynaptic, presynaptic)
            common = _np.intersect1d(postsynaptic, presynaptic)
            full_target = _np.union1d(postsynaptic, presynaptic)

        unselected_cells = _np.setdiff1d(gids, full_target)
        self._unselected_handlers = \
            _get_subhandlers(unselected_cells, handlers)
        self._connected_cells.reset()
        self._disconnected_cells.reset()

        self.presynaptic_neurons = presynaptic
        self.postsynaptic_neurons = postsynaptic

        circuit = self._scene.circuit
        synapses = circuit.projected_synapses(presynaptic, postsynaptic)
        if len(synapses) == 0:
            return

        local = circuit.Coordinates.local

        # Presynaptic only cells
        pre_gids = _np.unique(synapses.pre_gids())
        self._connected_cells.pre_target = pre_gids
        self._disconnected_cells.pre_target = _np.setdiff1d(presynaptic,
                                                            pre_gids)

        # Computing the definite subhandlers use for presynaptic cells and
        # updating the color of the disconnected cells.
        self._connected_cells.find_pre_subhandlers(handlers)
        self._disconnected_cells.find_pre_subhandlers(handlers)
        _set_attributes(self._connected_cells.pre_handlers,
                        self._presynaptic_attributes())
        _set_attributes(self._disconnected_cells.pre_handlers,
                        self._presynaptic_attributes(False))

        # Applying clippings
        if self._clip_branches:
            morphologies = circuit.load_morphologies(pre_gids, local)

            for gid, morphology in zip(pre_gids, morphologies):
                clipping = NeuronClipping().clipAll()
                # Don't waste time with unconnected neurons
                i = pre_gids.searchsorted(gid)
                if i < len(pre_gids) and pre_gids[i] == gid:
                    clipping.unclipEfferentBranches(int(gid), morphology,
                                                    synapses)

                _find_neuron(handlers, gid).apply(clipping)

        # Postsynaptic only cells
        post_gids = _np.unique(synapses.post_gids())
        self._connected_cells.post_target = post_gids
        self._disconnected_cells.post_target = _np.setdiff1d(postsynaptic,
                                                             post_gids)

        # Computing the definite subhandlers use for presynaptic cells and
        # updating the color of the disconnected cells.
        self._connected_cells.find_post_subhandlers(handlers)
        self._disconnected_cells.find_post_subhandlers(handlers)
        _set_attributes(self._connected_cells.post_handlers,
                        self._postsynaptic_attributes())
        _set_attributes(self._disconnected_cells.post_handlers,
                        self._postsynaptic_attributes(False))

        # Applying clippings
        if self._clip_branches:
            morphologies = circuit.load_morphologies(post_gids, local)

            for gid, morphology in zip(post_gids, morphologies):
                clipping = NeuronClipping().clipAll()
                # Don't waste time with unconnected neurons
                i = post_gids.searchsorted(gid)
                if i < len(post_gids) and post_gids[i] == gid:
                    clipping.unclipAfferentBranches(int(gid), morphology,
                                                    synapses)

                _find_neuron(handlers, gid).apply(clipping)

        # Cells in both sets are different
        _set_attributes(_get_subhandlers(common, handlers),
                        self._pre_and_postsynaptic_attributes())

        if self._clip_branches:
            morphologies = circuit.load_morphologies(common, local)

            for gid, morphology in zip(common, morphologies):
                gid = int(gid)
                clipping = NeuronClipping().clipAll()
                i = pre_gids.searchsorted(gid)
                if i < len(pre_gids) and pre_gids[i] == gid:
                    clipping.unclipEfferentBranches(gid, morphology, synapses)
                i = post_gids.searchsorted(gid)
                if i < len(post_gids) and post_gids[i] == gid:
                    clipping.unclipAfferentBranches(gid, morphology, synapses)

                _find_neuron(handlers, gid).apply(clipping)

        pre_post_gids = _np.union1d(pre_gids, post_gids)
        self._connected_cells.pre_and_post_target = \
            _np.intersect1d(common, pre_post_gids)
        self._disconnected_cells.pre_and_post_target = \
            _np.setdiff1d(common, pre_post_gids)

        # Computing the definite subhandlers used for cells in both sets and
        # updating the color of the disconnected ones.
        self._connected_cells.find_pre_and_post_subhandlers(handlers)
        self._disconnected_cells.find_pre_and_post_subhandlers(handlers)
        _set_attributes(self._disconnected_cells.pre_and_post_handlers,
                        self._presynaptic_attributes(False))

    def show_retrograde_projections(self, gid, subset=None):
        """Find the presynaptic cells of the given one and display them
        according to the current attributes for mode, color and clipping.
        """

        gids, handlers = _util.get_neurons(self._scene)
        if len(gids) == []:
            return
        # Reset attributes
        _set_attributes(handlers, self._unselected_attributes())

        self.presynaptic_neurons = None
        self.postsynaptic_neurons = None

        post_handler = _find_neuron(handlers, gid)
        if not post_handler:
            return

        circuit = self._scene.circuit
        if subset is None:
            subset = gids
        synapses = circuit.projected_synapses(subset, [gid])
        if len(synapses) == 0:
            post_handler.update()
            return

        self.postsynaptic_neurons = _np.array([gid], dtype="u4")
        presynaptic_neurons = _np.unique(synapses.pre_gids())
        self.presynaptic_neurons = presynaptic_neurons

        # Setting up targets used for updating the colors
        self._connected_cells.reset()
        self._connected_cells.post_target = self.postsynaptic_neurons
        self._connected_cells.pre_target = self.presynaptic_neurons
        self._connected_cells.find_subhandlers(handlers)

        self._disconnected_cells.reset()

        selected = _np.union1d(self._connected_cells.pre_target,
                               self.postsynaptic_neurons)
        unselected = _np.setdiff1d(gids, selected)
        self._unselected_handlers = _get_subhandlers(unselected, handlers)

        _set_attributes(self._connected_cells.post_handlers,
                        self._postsynaptic_attributes())
        attributes = self._presynaptic_attributes()
        attributes.mode = self._target_mode
        _set_attributes(self._connected_cells.pre_handlers, attributes)

        if self._clip_branches:
            local = circuit.Coordinates.local
            morphology = circuit.load_morphologies([gid], local)[0]

            post_handler.apply(
                NeuronClipping().clipAll().unclipAfferentBranches(
                    gid, morphology, synapses))

            morphologies = circuit.load_morphologies(
                presynaptic_neurons, local)
            for neuron, morphology in zip(presynaptic_neurons, morphologies):
                _find_neuron(handlers, neuron).apply(
                    NeuronClipping().clipAll().unclipEfferentBranches(
                        int(neuron), morphology, synapses))


    def show_anterograde_projections(self, gid, subset=None):
        """Find the postsynaptic cells of the given one and display them
        according to the current attributes for mode, color and clipping.
        """
        gids, handlers = _util.get_neurons(self._scene)
        if len(gids) == 0:
            return
        # Reset attributes
        _set_attributes(handlers, self._unselected_attributes())

        self.postsynaptic_neurons = None
        self.presynaptic_neurons = None

        pre_handler = _find_neuron(handlers, gid)
        if not pre_handler:
            return

        circuit = self._scene.circuit
        if subset is None:
            subset = gids
        synapses = circuit.projected_synapses([gid], subset)
        if len(synapses) == 0:
            pre_handler.update()
            return

        self.presynaptic_neurons = _np.array([gid], dtype="u4")
        postsynaptic_neurons = _np.unique(synapses.post_gids())
        self.postsynaptic_neurons = postsynaptic_neurons

        # Setting up handlers used for updating the colors
        self._connected_cells.reset()
        self._connected_cells.pre_target = self.presynaptic_neurons
        self._connected_cells.post_target = self.postsynaptic_neurons
        self._connected_cells.find_subhandlers(handlers)

        self._disconnected_cells.reset()

        selected = _np.union1d(self._connected_cells.post_target,
                               self.presynaptic_neurons)
        unselected = _np.setdiff1d(gids, selected)
        self._unselected_handlers = _get_subhandlers(unselected, handlers)

        # Updating color and mode attributes of all participating neurons
        # This speeds up the upgrade of neuron models from soma only to whole
        # neuron when needed.
        _set_attributes(self._connected_cells.pre_handlers,
                        self._presynaptic_attributes())
        attributes = self._postsynaptic_attributes()
        attributes.mode = self._target_mode
        _set_attributes(self._connected_cells.post_handlers, attributes)

        if self._clip_branches:
            local = circuit.Coordinates.local
            morphology = circuit.load_morphologies([gid], local)[0]

            pre_handler.apply(
                NeuronClipping().clipAll().unclipEfferentBranches(
                    gid, morphology, synapses))

            morphologies = circuit.load_morphologies(
                postsynaptic_neurons, local)
            for neuron, morphology in zip(postsynaptic_neurons, morphologies):
                _find_neuron(handlers, neuron).apply(
                    NeuronClipping().clipAll().unclipAfferentBranches(
                        int(neuron), morphology, synapses))

    def clear(self):
        # Toggling selection to none
        self._selected = None
        neurons, handlers = _util.get_neurons(self._scene)
        if len(neurons) == 0:
            return
        attributes = _rtneuron.AttributeMap(
            {'mode': _rtneuron.RepresentationMode.SOMA,
             'color': self._unselected_color})
        _set_attributes(handlers, attributes)
        self._unselected_handlers = handlers
        self._connected_cells.reset()
        self._disconnected_cells.reset()

    def set_presynaptic_attributes(self, attributes):
        """Sets the given attributes on the handlers of connected, presynaptic
        only cells."""
        _set_attributes(self._connected_cells.pre_handlers, attributes)

    def set_postsynaptic_attributes(self, attributes):
        """Sets the given attributes on the handlers of connected, postsynaptic
        only cells."""
        _set_attributes(self._connected_cells.post_handlers, attributes)

    def set_presynaptic_color(self, color):
        self._presynaptic_color = color

        _set_attributes(self._connected_cells.pre_handlers,
                        _rtneuron.AttributeMap({"color": color}))
        _set_attributes(self._connected_cells.pre_and_post_handlers,
                        _rtneuron.AttributeMap({"secondary_color": color}))

        dim_color = self._dim_color(color)
        _set_attributes(self._disconnected_cells.pre_handlers,
                        _rtneuron.AttributeMap({"color": dim_color}))
        _set_attributes(self._connected_cells.pre_and_post_handlers,
                        _rtneuron.AttributeMap({"secondary_color": dim_color}))

    def set_postsynaptic_color(self, color):
        self._postsynaptic_color = color

        attributes = _rtneuron.AttributeMap({"color": color})
        _set_attributes(self._connected_cells.post_handlers, attributes)
        _set_attributes(self._connected_cells.pre_and_post_handlers, attributes)

        attributes = _rtneuron.AttributeMap({"color": self._dim_color(color)})
        _set_attributes(self._disconnected_cells.post_handlers, attributes)
        _set_attributes(self._connected_cells.pre_and_post_handlers, attributes)

    def set_unselected_color(self, color):
        self._unselected_color = color

        _set_attributes(self._unselected_handlers,
                          _rtneuron.AttributeMap({"color": color}))

        # The disconnected cells also need to be updated since their
        # color depends on the unselected color
        post_dim_color = self._dim_color(self._postsynaptic_color)
        pre_dim_color = self._dim_color(self._presynaptic_color)
        _set_attributes(self._disconnected_cells.pre_handlers,
                        _rtneuron.AttributeMap({"color": pre_dim_color}))
        _set_attributes(self._disconnected_cells.post_handlers,
                        _rtneuron.AttributeMap({"color": post_dim_color}))

        _set_attributes(
            self._disconnected_cells.pre_and_post_handlers,
            _rtneuron.AttributeMap({"color": post_dim_color,
                                    "secondary_color": pre_dim_color}))

    def _presynaptic_attributes(self, connected=True):
        attributes = _rtneuron.AttributeMap()
        if connected:
            attributes.color = self._presynaptic_color
            attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
        else:
            attributes.color = self._dim_color(self._presynaptic_color)
            attributes.mode = _rtneuron.RepresentationMode.SOMA
        attributes.color_scheme = _rtneuron.ColorScheme.SOLID
        return attributes

    def _postsynaptic_attributes(self, connected=True):
        attributes = _rtneuron.AttributeMap()
        if connected:
            attributes.color = self._postsynaptic_color
            attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
        else:
            attributes.color = self._dim_color(self._postsynaptic_color)
            attributes.mode = _rtneuron.RepresentationMode.SOMA
        attributes.color_scheme = _rtneuron.ColorScheme.SOLID
        return attributes

    def _pre_and_postsynaptic_attributes(self):
        attributes = _rtneuron.AttributeMap()
        attributes.color = self._postsynaptic_color
        attributes.secondary_color = self._presynaptic_color
        attributes.color_scheme = _rtneuron.ColorScheme.BY_BRANCH_TYPE
        attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON
        return attributes

    def _unselected_attributes(self):
        attributes = _rtneuron.AttributeMap()
        attributes.color = self._unselected_color
        attributes.color_scheme = _rtneuron.ColorScheme.SOLID
        attributes.mode = _rtneuron.RepresentationMode.SOMA
        return attributes

    def _cell_selected(self, gid, section, segment):

        if gid != self._selected:
            self._num_clicks = 1
            self._selected = gid
            self.show_anterograde_projections(gid)
        else:
            self._num_clicks += 1
            if self._num_clicks == 2:
                self.show_retrograde_projections(gid)
            else:
                self._num_clicks = 0
                self._selected = None
                self.clear()

    def _dim_color(self, color):
        color = list(color)
        color[3] = min(color[3], self._unselected_color[3] * _ALPHA_MULTIPLIER)
        return color
