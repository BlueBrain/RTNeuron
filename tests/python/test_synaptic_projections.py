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

import setup
import image_compare

from rtneuron import *
from rtneuron.sceneops.SynapticProjections import \
    _find_neuron, SynapticProjections, NeuronClipping
import brain

import unittest

class TestBranchClippingHelpers(unittest.TestCase):

    def setUp(self):
        scene_attributes = AttributeMap()
        scene_attributes.use_meshes = False
        scene_attributes.inflatable_neurons = True

        self.gid = 200
        setup.setup_simple(
            self, self.gid, scene_attributes=scene_attributes,
            blue_config=brain.test.circuit_config,
            eq_config=setup.small_rect_eq_config)

        view = self.view
        view.attributes.inflation_factor = 2
        view.attributes.auto_compute_home_position = False

    def tearDown(self):
        del self.engine

    def test_efferent_clipping_helpers(self):

        view = self.view
        handler, morphology, synapses = self._display_synapses(self.gid, False)

        view.camera.setView([-601, 1158, 418], ([-0.452, -0.537, 0.711], 84.0))

        clipping = NeuronClipping().clipAll().unclipEfferentBranches(
            self.gid, morphology, synapses)
        handler.apply(clipping)

        image_compare.capture_and_compare(view, "efferent_synapse_clipping.png")

    def test_afferent_clipping_helpers(self):

        view = self.view
        handler, morphology, synapses = self._display_synapses(self.gid, True)

        view.camera.setView([360, 1515, 410], ([0.487, 0.378, 0.787], 86))

        clipping = NeuronClipping().clipAll().unclipAfferentBranches(
            self.gid, morphology, synapses)
        handler.apply(clipping)

        image_compare.capture_and_compare(view, "afferent_synapse_clipping.png")


    def _display_synapses(self, gid, afferent=True):

        scene = self.view.scene
        circuit = scene.circuit

        synapse_attributes = AttributeMap()
        synapse_attributes.radius = 4
        synapse_attributes.color = [1, 0, 0, 1]

        handler = _find_neuron(scene.objects, gid)

        if afferent:
            synapses = circuit.afferent_synapses([gid])
            self.view.scene.addAfferentSynapses(synapses, synapse_attributes)
        else:
            synapses = circuit.efferent_synapses([gid])
            self.view.scene.addEfferentSynapses(synapses, synapse_attributes)
        self.engine.frame()
        self.engine.waitFrame()
        self.engine.frame()
        self.engine.waitFrame()

        morphology = \
            circuit.load_morphologies([gid], circuit.Coordinates.local)[0]

        return handler, morphology,  synapses


class TestShowProjections(unittest.TestCase):

    def setUp(self):
        scene_attributes = AttributeMap()
        scene_attributes.use_meshes = False
        scene_attributes.inflatable_neurons = True

        setup.setup_simple(
            self, 'MiniColumn_0', scene_attributes=scene_attributes,
            blue_config=brain.test.circuit_config,
            eq_config=setup.small_rect_eq_config)
        self.engine.frame()
        self.engine.waitFrame()
        self.engine.frame()
        self.engine.waitFrame()

        view = self.view
        view.attributes.inflation_factor = 1
        view.attributes.auto_compute_home_position = False
        view.attributes.background = [1, 1, 1, 1]

        self.projections = SynapticProjections(view.scene)

    def tearDown(self):
        del self.engine

    def test_anterograde_projections(self):

        self.projections.show_anterograde_projections(200)
        view = self.view
        view.camera.setView([100, 1325, -387], ([0.5, 0.8650, 0.05], 174))

        image_compare.capture_and_compare(view, "anterograde_projections.png")

    def test_retrograde_projections(self):

        self.projections.show_retrograde_projections(230)
        view = self.view
        view.camera.setView([-15, 1664, 381], ([-0.122, -0.114, 0.986], 84.5))

        image_compare.capture_and_compare(view, "retrograde_projections.png")

    def test_synaptic_projections(self):

        self.projections.show_projections(range(200,310,10), range(400,510,10))
        view = self.view
        view.camera.setView([146, 1382, -566], ([0.694, 0.718, 0.0552], 175))

        image_compare.capture_and_compare(view, "synaptic_projections.png")


if __name__ == '__main__':
    unittest.main()


