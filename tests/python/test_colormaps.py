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
import rtneuron
import brain
import tempfile

from math import floor
from numpy import arange
import unittest
import os

BLACK = (0, 0, 0, 0)
WHITE = (1, 1, 1, 1)
GRAY = (0.5, 0.5, 0.5, 0.5)
EPSILON = 1/ 255.0 + 1e-16

def checkColorsEqual(color1, color2, epsilon = 1e-16) :
    for i in range(4) :
        assert(abs(color1[i] - color2[i]) < epsilon)

def interpolate(color1, color2, t) :
    return [c1 * (1 - t) + c2 * t for c1, c2 in zip(color1, color2)]

class TestColorMapObject(unittest.TestCase) :

    def setUp(self) :
        self.colorMap = ColorMap()

    def tearDown(self) :
        del self.colorMap
        try :
            os.remove("tmp_colormap.txt")
        except :
            pass

    def test_empty_constructor(self) :
        assert(self.colorMap.getRange() == (0, 0))
        assert(self.colorMap.getColor(0) == BLACK)

    def test_set_get_points(self) :
        assert(self.colorMap.getPoints() == {})
        points = {-10 : GRAY, 10 : WHITE}
        self.colorMap.setPoints(points)
        storedPoints = self.colorMap.getPoints()
        for value, color in points.items() :
            checkColorsEqual(color, storedPoints[value])

    def test_get_set_range(self) :
        self.colorMap.setPoints({-10 : GRAY, 10 : WHITE})
        assert(self.colorMap.getRange() == (-10, 10))
        self.colorMap.setRange(-2, 2)
        assert(self.colorMap.getRange() == (-2, 2))
        self.assertRaises(RuntimeError,
                          lambda: list(self.colorMap.setRange(2, -1)))
        self.colorMap.setRange(2, 2)
        assert(self.colorMap.getRange() == (2, 2))

    def test_set_range_collapse_uncollapse(self) :
        points = {-7 : GRAY, -3 : BLACK, 13 : WHITE}
        self.colorMap.setPoints(points)
        self.colorMap.setRange(-1, 1)
        self.colorMap.setRange(-0.0001, 0.0001)
        self.colorMap.setRange(0, 0)
        self.colorMap.setRange(-7, 13)
        assert(self.colorMap.getPoints() == points)

    def test_get_set_texture_size(self) :
        self.colorMap.textureSize = 256
        assert(self.colorMap.textureSize == 256)
        self.colorMap.textureSize = 512
        assert(self.colorMap.textureSize == 512)

    def test_get_set_texture_size_smaller_than_2(self) :
        self.colorMap.textureSize = 0
        assert(self.colorMap.textureSize == 2)
        self.colorMap.textureSize = 1
        assert(self.colorMap.textureSize == 2)

    def test_get_edge_colors(self) :
        self.colorMap.setPoints({-10 : GRAY, 10 : WHITE})
        checkColorsEqual(self.colorMap.getColor(-10), GRAY, EPSILON)
        # No epsilon needed for white because the quantization of colors
        # into 8 bits per channel preserves 1.0 unchanged (the same can't
        # be said for 0.5)
        checkColorsEqual(self.colorMap.getColor(10), WHITE)

    def test_get_colors_2points(self) :
        offset = lambda x : (x + 5.0) / 9.0
        first = (0.2, 0.4, 0.6, 0.8)
        second = (0.8, 0.6, 0.4, 0.2)
        self.colorMap.setPoints({-5 : first, 4 : second})
        for x in arange(-4.5, 3.7, 0.37) :
            checkColorsEqual(self.colorMap.getColor(x),
                             interpolate(first, second, offset(x)), EPSILON)

    def test_get_colors_3points(self) :
        first = (0.2, 0.4, 0.6, 0.8)
        middle = (0.1, 0.9, 0.1, 0.9)
        last = (0.8, 0.6, 0.4, 0.2)
        self.colorMap.setPoints({2 : first, 5 : middle, 17 : last})

        offset = lambda x : (x - 2.0) / 3.0
        for x in arange(2, 5, 0.29) :
            checkColorsEqual(self.colorMap.getColor(x),
                             interpolate(first, middle, offset(x)), EPSILON)

        offset = lambda x : (x - 5.0) / 12.0
        for x in arange(5.17, 17, 1.89) :
            checkColorsEqual(self.colorMap.getColor(x),
                             interpolate(middle, last, offset(x)), EPSILON)

    def test_get_color_2points_after_set_range(self) :
        first = (0.2, 0.4, 0.6, 0.8)
        second = (0.8, 0.6, 0.4, 0.2)
        self.colorMap.setPoints({-5 : first, 4 : second})
        self.colorMap.setRange(0, 1)
        for x in arange(0, 1, 0.13) :
            checkColorsEqual(self.colorMap.getColor(x),
                             interpolate(first, second, x), EPSILON)

    def test_get_color_3points_after_set_range(self) :
        first = (0.2, 0.4, 0.6, 0.8)
        middle = (0.1, 0.9, 0.1, 0.9)
        last = (0.8, 0.6, 0.4, 0.2)

        self.colorMap.setPoints({2 : first, 5 : middle, 17 : last})
        self.colorMap.setRange(0, 1)
        self.colorMap.textureSize = 256

        offset = lambda x : x / (3 / 15.0)
        for x in arange(0, 3 / 15.0, 0.13) :
            checkColorsEqual(self.colorMap.getColor(x),
                             interpolate(first, middle, offset(x)), EPSILON)

        offset = lambda x : (x * 15 - 3) / 12.0
        for x in arange(3 / 15.0, 1.0, 0.13) :
            checkColorsEqual(self.colorMap.getColor(x),
                             interpolate(middle, last, offset(x)), EPSILON)

    def test_get_color_different_texture_sizes(self) :
        second = (0.2, 0.4, 0.6, 0.8)
        self.colorMap.setPoints({-10 : GRAY, 10 : second})

        def checkColor() :
            checkColorsEqual(self.colorMap.getColor(-10), GRAY, EPSILON)
            checkColorsEqual(self.colorMap.getColor(0),
                             interpolate(GRAY, second, 0.5), EPSILON)
            checkColorsEqual(self.colorMap.getColor(10), second, EPSILON)

        self.colorMap.textureSize = 2
        checkColor()
        self.colorMap.textureSize = 4
        checkColor()
        self.colorMap.textureSize = 512
        checkColor()

    def test_get_off_bounds_color(self) :
        self.colorMap.setPoints({-10 : GRAY, 0 : BLACK, 10 : WHITE})
        checkColorsEqual(self.colorMap.getColor(-11), GRAY, EPSILON)
        assert(self.colorMap.getColor(11) == WHITE)
        self.colorMap.setRange(-2, 2)
        checkColorsEqual(self.colorMap.getColor(-3), GRAY, EPSILON)
        assert(self.colorMap.getColor(3) == WHITE)

    def test_load_good(self) :
        self.colorMap.load(setup.misc_data_path + 'rainbow.txt')
        controlPoints = {
               0.0 : (1, 0, 0, 1), 1 : (1, 1, 0, 1), 2 : (0, 1, 0, 1),
               3.0 : (0, 1, 1, 1), 4 : (0, 0, 1, 1), 5 : (1, 0, 1, 1)}
        assert(self.colorMap.getPoints() == controlPoints)
        assert(self.colorMap.getRange() == (0, 5))

        # Adjusting the texture size so the control points fall exactly into
        # the center of a texel. Otherwise, comparing the result of getColor
        # can't be directly compared to the control points
        self.colorMap.textureSize = 6
        for value, color in controlPoints.items() :
            assert(self.colorMap.getColor(value) == color)

    def test_load_old(self) :
        self.colorMap.load(setup.misc_data_path + 'rainbow_old.txt')
        controlPoints = self.colorMap.getPoints()
        assert(len(controlPoints) == 16)
        assert(self.colorMap.getRange() == (0, 5))
        assert(self.colorMap.textureSize == 16)
        checkPoints = {
               0.0 : (1, 0, 0, 1), 1 : (1, 1, 0, 1), 2 : (0, 1, 0, 1),
               3.0 : (0, 1, 1, 1), 4 : (0, 0, 1, 1), 5 : (1, 0, 1, 1)}
        for value, color in checkPoints.items() :
            # The control points match exactly but with getColor there
            # are rounding errors in the sampling calculations and an
            # epsilon is needed
            assert(controlPoints[value] == color)
            checkColorsEqual(self.colorMap.getColor(value), color, 1e-6)

    def test_load_nonexistent(self) :
        self.assertRaises(RuntimeError,
                          lambda: list(self.colorMap.load('nonexistent')))

    def test_load_bad_file(self) :
        self.assertRaises(RuntimeError,
                          lambda: list(self.colorMap.load(setup.misc_data_path + 'broken_colormap.txt')))

    def test_save(self) :
        points = {0 : (1, 0, 0, 1), 1 : (1, 1, 0, 1), 2 : (0, 1, 0, 1),
                  3 : (0, 1, 1, 1), 4 : (0, 0, 1, 1), 5 : (1, 0, 1, 1)}
        self.colorMap.setPoints(points)
        self.colorMap.save('tmp_colormap.txt')

        # The output file can't be directly compared to a reference due to
        # a difference in the file header between Boost serialization after
        # version 1.53
        reloaded = ColorMap()
        reloaded.load('tmp_colormap.txt')
        reloadedPoints = reloaded.getPoints()
        for value, color in reloadedPoints.items() :
            assert(points[value] == color)
        for value, color in points.items() :
            assert(reloadedPoints[value] == color)

class TestCompartmentColorMapImages(unittest.TestCase) :
    def setUp(self) :
        rtneuron.display_circuit(
            brain.test.blue_config, 'L5CSPC', report="allCompartments",
            eq_config=setup.medium_rect_eq_config)
        view = rtneuron.engine.views[0]
        view.attributes.idle_AA_steps = 0
        view.attributes.background = [0.5, 0.5, 0.5, 1]
        view.camera.setView([38, 650, 1003],
                            ([0.0, 0.0, 1.0], 0.0))

    def tearDown(self) :
        rtneuron.engine = None

    def test_compartment_colormaps(self) :
        # A single test is used to test several colormaps for speed reasons
        view = rtneuron.engine.views[0]
        rtneuron.engine.player.timestamp = 5.5
        image_compare.capture_and_compare(view, 'colormap_default.png')

        view.attributes.colormaps.compartments.setRange(-65, -60)
        image_compare.capture_and_compare(view, 'colormap_default_rescaled.png')

        view.attributes.colormaps.compartments.setPoints({0 : [0, 0, 0, 1],
                                                          1 : [1, 1, 1, 1]})
        view.attributes.colormaps.compartments.setRange(-65, -60)
        image_compare.capture_and_compare(view, 'colormap_2points.png')

        colorMap = ColorMap()
        colorMap.setPoints({0 : [0, 0, 0, 1],
                            0.5 : [0, 1, 0, 1],
                            1.0 : [1, 1, 1, 1]})
        view.attributes.colormaps.compartments = colorMap
        colorMap.setRange(-65, -55)
        image_compare.capture_and_compare(view, 'colormap_3points.png')

class TestSpikeColorMapImages(unittest.TestCase) :

    def _create_spikes_file(self, target) :
        # Generating a spike data file (the file from TestData is too sparse)
        spikefile = tempfile.NamedTemporaryFile(suffix = '.dat')
        spikes = open(spikefile.name, 'w')
        time = 0
        gids = self.simulation.gids(target)
        positions = self.simulation.open_circuit().positions(gids)

        x_gid = [(p[0], g) for p, g in zip(positions, gids)]
        x_gid.sort()
        for dummy, gid in x_gid :
            spikes.write('1 %d\n' % gid)
            spikes.write('%f %d\n' % (time + 2, gid))
            spikes.write('%f %d\n' % (time + 4, gid))
            time += 2.0 / len(gids)
        spikes.flush()
        spikes.close()
        return spikefile

    def setUp(self) :
        target = 'L5CSPC'
        attributes = AttributeMap({'mode' : rtneuron.RepresentationMode.SOMA})
        setup.setup_simple(self, target, target_attributes=attributes)

        view = self.view
        view.attributes.idle_AA_steps = 32
        view.attributes.auto_compute_home_position = False
        view.attributes.background = [0, 0, 0, 1]

        self._spikefile = self._create_spikes_file(target)
        rtneuron.apply_spike_data(self._spikefile.name, view)
        self.engine.player.window = [0, 6]

        # To make sure the scene is ready before the timestamps are modified
        self.engine.frame()
        self.engine.frame()

    def tearDown(self) :
        del self.engine
        self._spikefile.close()

    def test_spike_colormaps(self) :
        view = self.view
        self.engine.player.timestamp = 5
        view.attributes.spike_tail = 2
        view.camera.setView([49, 630, 509], ([0.0, 0.0, 1.0], 0.0))

        image_compare.capture_and_compare(
            view, 'spike_simulation_t5_d2_somas.png')

        view.attributes.colormaps.spikes.setPoints({0 : [1, 0, 0, 1],
                                      1 : [1, 1, 1, 1]})
        image_compare.capture_and_compare(view, 'spike_colormap_2points.png')

        view.attributes.colormaps.spikes.setPoints({0 : [0, 0, 0, 0],
                                                    0.33 : [1, 0, 0, 0.25],
                                                    0.66 : [1, 1, 0, 0.5],
                                                    1.0 : [1, 1, 1, 1.0]})
        image_compare.capture_and_compare(view, 'spike_colormap_4points.png')

class TestCustomStaticColorMapImages(unittest.TestCase) :

    def setUp(self) :
        attributes = AttributeMap()
        attributes.primary_color = [1, 1, 1, 1]
        attributes.secondary_color = [0, 0, 1, 1]
        setup.setup_simple(self, 450, target_attributes=attributes,
                           eq_config=setup.medium_eq_config)
        view = self.view
        view.attributes.auto_compute_home_position = False
        view.attributes.background = [0, 0, 0, 1]
        view.camera.setView([-1124, 1162, -57],
                            ([-0.070, -0.997, -0.0362], 98))

    def tearDown(self) :
        del self.engine
        rtneuron.engine = None

    def test_by_distance_colormaps(self):
        view = self.view
        neuron = view.scene.objects[0]

        neuron.attributes.color_scheme = \
            rtneuron.ColorScheme.BY_DISTANCE_TO_SOMA
        neuron.update()
        image_compare.capture_and_compare(view,
                                          'default_by_distance_coloring.png')

        colormap = ColorMap()
        colormap.setPoints({0: [0, 0, 1, 1], 100: [1, 0, 0, 1]})
        neuron.attributes.colormaps = AttributeMap()
        neuron.attributes.colormaps.by_distance_to_soma = colormap
        neuron.update()
        image_compare.capture_and_compare(view, 'by_distance_colormap1.png')

        colormap.setPoints({0: [0, 0, 1, 1], 100: [1, 0, 0, 1],
                            200: [1, 0.5, 0, 1], 300: [1, 1, 0, 1]})
        neuron.update()
        image_compare.capture_and_compare(view, 'by_distance_colormap2.png')

        colormap.setRange(0, 1000)
        neuron.update()
        image_compare.capture_and_compare(view, 'by_distance_colormap3.png')

        neuron.attributes.colormaps = AttributeMap()
        neuron.update()
        image_compare.capture_and_compare(view,
                                          'default_by_distance_coloring.png')

    def test_by_width_colormaps(self):
        view = self.view
        neuron = view.scene.objects[0]

        neuron.attributes.color_scheme = rtneuron.ColorScheme.BY_WIDTH
        neuron.update()
        image_compare.capture_and_compare(view, 'default_by_width_coloring.png')

        colormap = ColorMap()
        colormap.setPoints({0: [1, 0, 0, 1], 30: [1, 1, 1, 1]})
        neuron.attributes.colormaps = AttributeMap()
        neuron.attributes.colormaps.by_width = colormap
        neuron.update()
        image_compare.capture_and_compare(view, 'by_width_colormap1.png')

        colormap.setPoints({0: [1, 0, 0, 1], 5: [1, 0.5, 0, 1],
                            10: [1, 1, 0, 1], 30: [1, 1, 1, 1]})
        neuron.update()
        image_compare.capture_and_compare(view, 'by_width_colormap2.png')

        colormap.setRange(0, 15)
        neuron.update()
        image_compare.capture_and_compare(view, 'by_width_colormap3.png')

        neuron.attributes.colormaps = AttributeMap()
        neuron.update()
        image_compare.capture_and_compare(view, 'default_by_width_coloring.png')

if __name__ == '__main__':
    unittest.main()
