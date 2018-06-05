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
os.environ['RTNEURON_MAX_FB_WIDTH'] = '4294967296'
os.environ['RTNEURON_MAX_FB_HEIGHT'] = '4294967296'     # No tiled snapshots

import setup
import image_compare
import unittest

import rtneuron

import sys

os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12' # FBO
del os.environ['EQ_WINDOW_IATTR_PLANES_SAMPLES']    # No hw AA

def compare_with_original(view, name, arg = 1):
    ''' Takes a screenshot with the original snapshot
    function and compares the result to the image generated
    with the new function '''

    original = "original_" + name
    view.snapshot(original)
    view.snapshot(name, arg)
    try:
        image_compare.compare(original, name)
    finally:
        if 'KEEP_IMAGES' not in os.environ:
            os.remove(name)
            os.remove(original)

class TestBadSnapshot(unittest.TestCase):

    def setUp(self):
       self.engine = rtneuron.RTNeuron([])
       self.engine.init(setup.medium_rect_eq_config)
       self.view = self.engine.views[0]

    def tearDown(self):
        del self.engine

    def test_scaled_snapshot_error_1(self):
        self.assertRaises(ValueError,
            lambda: self.view.snapshot("snapshot_scale_error1.png", -1))

    def test_scaled_snapshot_error_2(self):
        self.assertRaises(ValueError,
            lambda: self.view.snapshot("snapshot_scale_error2.png", 0))

    def test_scaled_snapshot_error_3(self):
        # Will not use tiled snapshot due to variables exported at the top.
        self.assertRaises(RuntimeError,
            lambda: self.view.snapshot("snapshot_scale_error3.png", 100000))

    def test_sized_snapshot_error_1(self):
        self.assertRaises(TypeError,
            lambda: self.view.snapshot("snapshot_size_error1.png", ('','')))

    def test_sized_snapshot_error_2(self):
        self.assertRaises(ValueError,
            lambda: self.view.snapshot("snapshot_size_error2.png", (0, 0)))

    def test_sized_snapshot_error_3(self):
        self.assertRaises(OverflowError,
            lambda: self.view.snapshot("snapshot_size_error3.png", (-1, 0)))

    def test_sized_snapshot_error_4(self):
        # Will not use tiled snapshot due to variables exported at the top.
        self.assertRaises(RuntimeError,
            lambda: self.view.snapshot("snapshot_size_error4.png",
                                       (100000, 100000)))

class TestSnapshot(unittest.TestCase):

    def setUp(self):
        attributes = rtneuron.AttributeMap(
            {'mode': rtneuron.RepresentationMode.WHOLE_NEURON})
        setup.setup_simple(self, 'L5CSPC', target_attributes = attributes)

        self.view = self.engine.views[0]
        self.view.auto_compute_home_position = False
        self.view.scene = self.scene
        # To make sure the scene is ready before the timestamps are modified
        self.engine.frame()
        self.engine.frame()

        position = [30, 600, 300]
        orientation = ([0, 0, 1], 0)
        self.view.camera.setView(position,orientation)

    def tearDown(self):
        del self.engine
        del self.scene
        del self.view
        del self.simulation

    def view_snapshot(self, arg):
        def snapshot(view, name):
            return view.snapshot(name, arg)
        return snapshot

    # Default snapshot with no parameters
    def test_simple_snapshot(self):
        neurons = self.scene.objects[0]
        neurons.attributes.mode = rtneuron.RepresentationMode.SOMA
        compare_with_original(self.view, "snapshot_simple.png")

    # Scaled snapshot
    def test_scaled_snapshot(self):
        # So results are not GPU/drivers dependant
        self.view.attributes.idle_AA_steps = 32
        snapshot_with_scale = self.view_snapshot(0.5)
        neurons = self.scene.objects[0]
        neurons.attributes.mode = rtneuron.RepresentationMode.SOMA
        image_compare.capture_and_compare(
            self.view, 'snapshot_scaled.png', 1, snapshot_with_scale)

    # Sized snapshot
    def test_res_snapshot(self):
        # So results are not GPU/drivers dependant
        self.view.attributes.idle_AA_steps = 32
        snapshot_with_res = self.view_snapshot((512, 512))
        neurons = self.scene.objects[0]
        neurons.attributes.mode = rtneuron.RepresentationMode.SOMA
        image_compare.capture_and_compare(
            self.view, 'snapshot_sized.png', 1, snapshot_with_res)

    # Scene modification
    def test_modified_em_shading(self):
        # Activate electron shading
        self.scene.attributes.em_shading = True
        compare_with_original(self.view, "snapshot_shading.png")

    # View modification
    def test_modified_background(self):
        # Modify background color
        self.view.attributes.background = [1.0, 0.0, 0.0]
        neurons = self.scene.objects[0]
        neurons.attributes.mode = rtneuron.RepresentationMode.SOMA
        compare_with_original(self.view, "snapshot_background.png")

    def test_display_simulation(self):
        # Display simulation
        neurons = self.scene.objects[0]
        neurons.attributes.mode = rtneuron.RepresentationMode.SOMA
        rtneuron.apply_compartment_report(self.simulation, self.view,
                                          "allCompartments")
        self.engine.player.window = [0, 10]
        self.engine.player.timestamp = 5
        compare_with_original(self.view, "snapshot_simulation.png")

    def test_colormap(self):
        # Modify the spikes colormap
        self.view.attributes.colormaps.spikes.setPoints({0: [0, 0, 0, 0],
                                                         0.33: [1, 0, 0, 0.25],
                                                         0.66: [1, 1, 0, 0.5],
                                                         1.0: [1, 1, 1, 1.0]})
        rtneuron.apply_spike_data(self.simulation, self.view)
        self.engine.player.window = [0, 10]
        self.engine.player.timestamp = 5
        compare_with_original(self.view, "snapshot_colormap.png")


if __name__ == '__main__':
    unittest.main()
