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

import os
import setup
import image_compare
from rtneuron import *

import brain
import numpy

import unittest

class TestIncrementalAdditions(unittest.TestCase):

    def setUp(self):
        setup.setup_empty_scene(self, eq_config=setup.medium_eq_config)

        self.view.attributes.auto_compute_home_position = False
        self.view.camera.setView([26, 965, 600], ([0.0, 0.0, 1.0], 0.0))

        circuit_config = brain.test.circuit_config
        simulation = brain.Simulation(circuit_config)
        self.scene.circuit = simulation.open_circuit()
        self.simulation = simulation

        self.tmp_images = []

    def tearDown(self):
        for image in self.tmp_images:
            #os.remove(image)
            pass

    def test(self):

        mc0 = self.simulation.gids('MiniColumn_2')
        mc1 = self.simulation.gids('MiniColumn_5')
        mc2 = self.simulation.gids('MiniColumn_8')

        target = numpy.union1d(numpy.union1d(mc0, mc1), mc2)
        circuit = self.simulation.open_circuit()
        synapses = circuit.afferent_synapses(target)

        # Reference image with two columns
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.SOMA
        attributes.color = [1, 1, 1, 1]
        self.scene.addNeurons(mc0, attributes)
        attributes.color = [1, 0, 0, 1]
        self.scene.addNeurons(mc1, attributes)
        self.tmp_images.append(image_compare.capture_temporary(self.view))

        self.scene.clear()
        self.engine.frame()

        # Reference image with three columns and synapses
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.SOMA
        attributes.color = [1, 1, 1, 1]
        self.scene.addNeurons(mc0, attributes)
        attributes.color = [1, 0, 0, 1]
        self.scene.addNeurons(mc1, attributes)
        attributes.color = [0, 0, 1, 1]
        self.scene.addNeurons(mc2, attributes)
        self.scene.addAfferentSynapses(synapses)
        self.tmp_images.append(image_compare.capture_temporary(self.view))

        self.scene.clear()
        self.engine.frame()

        # Now do the same with incremental loads
        attributes.color = [1, 1, 1, 1]
        self.scene.addNeurons(mc0, attributes)
        self.engine.frame()
        attributes.color = [1, 0, 0, 1]
        self.scene.addNeurons(mc1, attributes)
        self.engine.frame()
        self.compare_with(0)

        attributes.color = [0, 0, 1, 1]
        handler1 = self.scene.addNeurons(mc2, attributes)
        self.engine.frame()
        handler2 = self.scene.addAfferentSynapses(synapses)
        self.engine.frame()
        self.compare_with(1)

        # Compare after removing a neuron and a synapse target
        self.scene.remove(handler1)
        self.engine.frame()
        self.scene.remove(handler2)
        self.engine.frame()
        self.compare_with(0)

    def test_with_subset_modifications(self):

        simulation = self.simulation
        mc0 = simulation.gids('MiniColumn_2')
        mc1 = simulation.gids('MiniColumn_5')
        gids = numpy.append(mc0, mc1)

        circuit = self.simulation.open_circuit()
        synapses = circuit.afferent_synapses(numpy.union1d(mc0, mc1))

        # Reference image
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.SOMA
        attributes.color = [1, 1, 1, 1]
        handler = self.scene.addNeurons(gids, attributes)
        self.scene.addAfferentSynapses(synapses)
        self.engine.frame()
        subset = handler.query(mc0)
        subset.attributes.color = [1, 0, 1, 1]
        subset.update()
        self.tmp_images.append(image_compare.capture_temporary(self.view))

        self.scene.clear()
        self.engine.frame()

        # And now the same but doing the subset modification before adding
        # the synapses.
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.SOMA
        attributes.color = [1, 1, 1, 1]
        handler = self.scene.addNeurons(gids, attributes)
        subset = handler.query(mc0)
        subset.attributes.color = [1, 0, 1, 1]
        subset.update()
        self.engine.frame()
        self.scene.addAfferentSynapses(synapses)
        self.compare_with(0)

    def compare_with(self, index):
        try:
            tmp = image_compare.capture_temporary(self.view)
            image_compare.compare(tmp, self.tmp_images[index])
        finally:
            os.remove(tmp)

if __name__ == '__main__':
    unittest.main()
