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
import rtneuron
import image_compare

import brain
import unittest
import sys

blue_config = brain.test.blue_config

def display_circuit(eq_config = setup.medium_rect_eq_config) :
    target = [429, 441, 442, 446, 447, 453, 454, 463, 475, 478]
    rtneuron.display_circuit(blue_config, (target, {'color' : [1, 0, 0, 1]}),
                             eq_config = eq_config)

class TestViewAttributeNamesAndTypes(unittest.TestCase) :

    def setUp(self) :
        self.engine = rtneuron.RTNeuron(sys.argv)
        self.engine.init(setup.medium_rect_eq_config)
        self.view = self.engine.views[0]

    def tearDown(self) :
        # Explicit deletion is needed, otherwise the RTNeuron instance is not
        # destroyed before a new one is created.
        del self.engine

    def test_attributes(self) :
        self.view.attributes.auto_adjust_model_scale = True
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "auto_adjust_model_scale", 10)
        self.view.attributes.auto_compute_home_position = True
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "auto_compute_home_position", -1)
        self.view.attributes.background = [1, 1, 1]
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "background", [1, 1])
        self.view.attributes.clod_threshold = 2000
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "clod_threshold", 'wrong')
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "auto_compute_home_position", 10)
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "idle_AA_steps", 'foo')
        self.view.attributes.inflation_factor = 5
        self.view.attributes.inflation_factor = 0
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "inflation_factor", 'toto')
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "inflation_factor", -2)
        self.view.attributes.lod_bias = 2
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "lod_bias", 'wrong')
        self.view.attributes.output_file_prefix = 'prefix'
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "output_file_prefix", False)
        self.view.attributes.output_file_format = 'giff'
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "output_file_prefix", 10.0)
        self.view.attributes.probe_color = [1, 1, 1]
        self.view.attributes.probe_color = [1, 1, 1, 1]
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "probe_color", [1, 1])
        self.view.attributes.probe_threshold = 40
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "probe_threshold", rtneuron.AttributeMap())
        self.view.attributes.snapshot_at_idle = False
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "snapshot_at_idle", 10)
        self.view.attributes.spike_tail = 10000.0
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "spike_tail", 'bad')
        self.view.attributes.stereo = True
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "stereo", -1)
        self.view.attributes.stereo_correction = 1.4
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "stereo_correction", 'no')
        self.view.attributes.use_roi = True
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "use_roi", -0xBAD)
        self.view.attributes.zero_parallax_distance = -111
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "zero_parallax_distance", 'error')
        self.assertRaises(RuntimeError, setattr, self.view.attributes,
                          "testing_at_least_one_unknown_attribute", 10)

class TestSingleViewAttributes(unittest.TestCase) :

    def setUp(self) :
        display_circuit()
        view = rtneuron.engine.views[0]
        view.attributes.background = [0, 0, 0, 1]
        view.attributes.idle_AA_steps = 32
        view.camera.setView(
            [36.77288055419922, 664.2935791015625, 868.2318115234375],
            ([0.0, 0.0, 1.0], 0.0))

    def tearDown(self) :
        del rtneuron.engine

    def test_lod_bias_1(self) :
        view = rtneuron.engine.views[0]
        view.attributes.lod_bias = 0
        rtneuron.engine.frame()
        rtneuron.engine.waitFrame()
        view.attributes.lod_bias = 1
        image_compare.capture_and_compare(
            view, 'view_attributes_lod_bias_1.png')

    def test_lod_bias_0(self) :
        view = rtneuron.engine.views[0]
        view.attributes.lod_bias = 1
        rtneuron.engine.frame()
        rtneuron.engine.waitFrame()
        view.attributes.lod_bias = 0
        image_compare.capture_and_compare(
            view, 'view_attributes_lod_bias_0.png')

    def test_display_simulation_true(self) :
        view = rtneuron.engine.views[0]
        view.attributes.display_simulation = False
        rtneuron.engine.frame()
        rtneuron.engine.waitFrame()
        view.attributes.display_simulation = True
        image_compare.capture_and_compare(
            view, 'view_attributes_with_sim.png')

    def test_display_simulation_false(self) :
        view = rtneuron.engine.views[0]
        view.attributes.display_simulation = True
        rtneuron.engine.frame()
        rtneuron.engine.waitFrame()
        view.attributes.display_simulation = False
        image_compare.capture_and_compare(
            view, 'view_attributes_without_sim.png')

if __name__ == '__main__':
    unittest.main()


