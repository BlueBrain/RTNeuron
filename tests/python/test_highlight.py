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

import setup
import image_compare
from rtneuron import *
import numpy

import brain

import unittest

lods = [("tubelets", "tubelets", None),
        ("cylinders", "high_detail_cylinders", None),
        ("mesh", "mesh", None),
        ("soma", "spherical_soma", None)]

class TestHighlightLODs(unittest.TestCase) :
    def setUp(self) :
        setup.setup_lod_scenes(self, lods)
        self.view.attributes.highlight_color = [0.7, 0.85, 1.0, 1.0]

    def tearDown(self) :
        del self.engine
        del self.scenes
        del self.view

    def test_mesh_highight(self) :
        self.run_lod_test('mesh')

    def test_soma_highlight(self) :
        self.run_lod_test('soma')

    def test_cylinders_highlight(self) :
        self.run_lod_test('cylinders')

    def test_tubelets_highglight(self) :
        self.run_lod_test('tubelets')

    def run_lod_test(self, scene_name) :
        scene = self.scenes[scene_name]
        self.view.scene = scene
        self.engine.frame()
        scene.highlight([406], True)

        file_name = 'lod_' + scene_name + '.png'
        image_compare.capture_and_compare(self.view, file_name)

class TestHighlightSomaMode(unittest.TestCase) :
    def setUp(self) :
        # This scene is not created the same as the one for soma above.
        attributes = AttributeMap({'mode' : RepresentationMode.SOMA})
        setup.setup_simple(self, 406, target_attributes = attributes)

        view = self.view
        view.camera.setView(
            [87.08049011230469, 676.59033203125, 148.541748046875],
            ([-0.16840891540050507, 0.9842529892921448, 0.053707364946603775],
             40.37215805053711))
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        view.attributes.auto_compute_home_position = False
        view.attributes.highlight_color = [0.7, 0.85, 1.0, 1.0]

    def tearDown(self) :
        del self.engine

    def test_soma_highlight(self) :
        self.engine.frame()
        self.scene.highlight([406], True)
        image_compare.capture_and_compare(self.view, 'lod_soma.png')

class TestHighlightAndSimulation(unittest.TestCase):
    def setUp(self) :
        setup.setup_lod_scenes(self, lods, eq_config = setup.small_eq_config)
        self.view.attributes.highlight_color = [-1.0, 1.0, 0.5, 0.0]

    def tearDown(self) :
        del self.engine

    def test_mesh_highlight(self) :
        self.run_lod_test('mesh')

    def test_soma_highlight(self) :
        self.run_lod_test('soma')

    # There is no need to make comparisons for any other level of detail
    # because for pseudo-cylinders and tubelets:
    # - The state management to turn one highlighting is already tested
    #   with the test case without simulation.
    # - The GLSL functions that choose the object color and do the shading
    #   are the same as for meshes (shading/membrane_color.frag,
    #   simulation_value_to_color.glgl and geom/reported_variable.vert)
    #
    # So no extra tests are neeed.

    def run_lod_test(self, scene_name) :
        scene = self.scenes[scene_name]
        self.view.scene = scene
        apply_compartment_report(self.simulation, self.view, "allCompartments")
        self.engine.frame()
        scene.highlight([406], True)

        file_name = 'lod_' + scene_name + '_with_sim_and_highlight.png'
        image_compare.capture_and_compare(self.view, file_name)

class TestHighlightFunctions(unittest.TestCase):
    def setUp(self):
        self.engine = RTNeuron([])
        gids = [310, 320, 330, 340, 350, 360]
        self.scene = self.engine.createScene(AttributeMap(
            {"circuit": brain.test.blue_config}))
        self.scene.addNeurons(gids)

    def tearDown(self):
        del self.engine

    def test_highlight_unexistent(self):
        assert(len(self.scene.highlightedNeurons) == 0)
        self.scene.highlight([400], True)
        assert(len(self.scene.highlightedNeurons) == 0)
        self.scene.highlight([400], False)

    def test_highlight_cell(self):
        assert(len(self.scene.highlightedNeurons) == 0)

        self.scene.highlight([320], True)
        assert(all(self.scene.highlightedNeurons == [320]))

        self.scene.highlight([320], False)
        assert(len(self.scene.highlightedNeurons) == 0)

        self.scene.highlight([320], True)
        self.scene.highlight([330], True)
        assert(all(self.scene.highlightedNeurons == [320, 330]))

    def test_highlight_cell_list(self):
        assert(len(self.scene.highlightedNeurons) == 0)

        self.scene.highlight([320, 330, 340], True)
        assert(all(self.scene.highlightedNeurons == [320, 330, 340]))

        self.scene.highlight([330], False)
        assert(all(self.scene.highlightedNeurons == [320, 340]))

    def test_highlight_cell_array(self):
        assert(len(self.scene.highlightedNeurons) == 0)

        gids = numpy.array([320, 330, 340])
        self.scene.highlight(gids, True)
        assert(all(self.scene.highlightedNeurons == gids))


if __name__ == '__main__':
    unittest.main()
