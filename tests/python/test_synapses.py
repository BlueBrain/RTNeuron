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

import setup
import image_compare
from rtneuron import *
import rtneuron
import brain

import unittest

class Common:
    def setUp(self):
        self._engine = RTNeuron([])

        rtneuron._init_simulation(brain.test.circuit_config)

        self._gids = rtneuron.simulation.gids('MiniColumn_0')[30:35]

        rtneuron.engine = self._engine

        scene = self._engine.createScene(
            AttributeMap({"circuit": brain.test.circuit_config}))

        self._engine.init(setup.medium_rect_eq_config)
        view = self._engine.views[0]
        self._view = view
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        view.scene = scene


    def tearDown(self):
        import rtneuron
        rtneuron.engine = None
        del self._engine

class TestDisplaySynapses(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def display_target(self, target):
        display_synapses(target)
        # To make sure the scene is created and synapses are displayed
        for i in range(2):
            self._engine.frame()
            self._engine.waitFrame()

    def test_display_gids_synapses(self):
        self.display_target(self._gids)

    def test_display_synapses_incremental(self):
        for gid in self._gids:
            display_synapses(int(gid))
            self._engine.frame()
            self._engine.waitFrame()

    def test_display_cell_target_synapses(self):
        self.display_target(self._gids)

    def test_display_target_name_synapses(self):
        self.display_target('MiniColumn_0')

    def test_display_shared_synapses(self):
        pass

class TestSynapsePositions(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def add_synapses(self, attributes, afferent=True):

        circuit = rtneuron.simulation.open_circuit()
        if afferent:
            synapses = circuit.afferent_synapses(self._gids[:1])
            self._view.scene.addAfferentSynapses(synapses, attributes)
        else:
            synapses = circuit.efferent_synapses(self._gids[:1])
            self._view.scene.addEfferentSynapses(synapses, attributes)

        self._view.camera.setView([-115, 1359, 221], ([0, -1, 0], 45))
        self._engine.frame()
        self._engine.waitFrame()

    def test_afferent_center_positions(self):
        attributes = AttributeMap({"surface": False})
        self.add_synapses(attributes)
        image_compare.capture_and_compare(
            self._view, "synapses_afferent_center.png")

    def test_efferent_center_positions(self):
        attributes = AttributeMap({"surface": False})
        self.add_synapses(attributes, afferent=False)
        image_compare.capture_and_compare(
            self._view, "synapses_efferent_center.png")

    def test_afferent_surface_positions(self):
        attributes = AttributeMap({"surface": True})
        self.add_synapses(attributes)
        image_compare.capture_and_compare(
            self._view, "synapses_afferent_surface.png")

    def test_efferent_surface_positions(self):
        attributes = AttributeMap({"surface": True})
        self.add_synapses(attributes, afferent=False)
        image_compare.capture_and_compare(
            self._view, "synapses_efferent_surface.png")

class TestRemoveSynapses(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)
        self._tmp_files = []

    def tearDown(self):
        import os
        for f in self._tmp_files:
            os.remove(f)
        Common.tearDown(self)

    def test_remove_synapses(self):
        view = self._view
        engine = self._engine
        blank = image_compare._get_tmp_file_name()
        self._tmp_files.append(blank)
        view.snapshot(blank)

        display_synapses(self._gids)
        # To make sure the scene is created and synapses are displayed
        engine.frame()
        engine.waitFrame()
        synapses = image_compare._get_tmp_file_name()
        self._tmp_files.append(synapses)
        view.snapshot(synapses)

        self.assertRaises(AssertionError,
                          lambda: list(image_compare.compare(blank, synapses)))

        view.scene.remove(view.scene.objects[0])
        view.scene.update()
        engine.frame()
        engine.waitFrame()
        blank2 = image_compare._get_tmp_file_name()
        self._tmp_files.append(blank2)
        view.snapshot(blank2)

        image_compare.compare(blank, blank2)

class TestSynapsesAttributes(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def add_synapses(self, attributes):

        circuit = rtneuron.simulation.open_circuit()
        for i in range(3):
            synapses = circuit.afferent_synapses(self._gids[i:i+1])
            self._view.scene.addAfferentSynapses(synapses, attributes[i])
        self._view.camera.setView([-50, 1100, 128],
                                  ([0.7400, -0.5621, 0.3692], 80))
        self._engine.frame()
        self._engine.waitFrame()

    def test_radius(self):
        view = self._view

        radii = [1, 2, 5]
        attributes = [AttributeMap({"radius": radius}) for radius in radii]
        self.add_synapses(attributes)
        image_compare.capture_and_compare(view, "synapse_radius.png")

        view.attributes.auto_compute_home_position = False
        radii = [5, 1, 2]
        for obj, radius in zip(view.scene.objects, radii):
            obj.attributes.radius = radius
            obj.update()
        self._engine.frame()
        self._engine.waitFrame()
        image_compare.capture_and_compare(view, "synapse_radius_2.png")

    def test_color(self):
        view = self._view

        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        attributes = [AttributeMap({"color": color}) for color in colors]
        self.add_synapses(attributes)
        image_compare.capture_and_compare(view, "synapse_color.png")

        view.attributes.auto_compute_home_position = False
        colors = [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]
        for obj, radius in zip(view.scene.objects, colors):
            obj.attributes.color = radius
            obj.update()
        view.scene.update()
        self._engine.frame()
        self._engine.waitFrame()
        image_compare.capture_and_compare(view, "synapse_color_2.png")

if __name__ == '__main__':
    unittest.main()

