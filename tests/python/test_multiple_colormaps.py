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

import unittest
import os

GIDS = [394, 400, 406, 412, 423, 429, 443, 451, 456, 463, 468, 479]

class Common:
    def setUp(self, gids=GIDS,
              mode=rtneuron.RepresentationMode.WHOLE_NEURON):

        rtneuron.display_circuit(
            brain.test.blue_config, (gids, {'mode' : mode}),
            eq_config = setup.small_eq_config)

        view = rtneuron.engine.views[0]
        view.attributes.idle_AA_steps = 32
        view.attributes.auto_compute_home_position = False
        view.attributes.background = [0.5, 0.5, 0.5, 1]
        view.attributes.spike_tail = 2
        view.camera.setView([49, 630, 809], ([0.0, 0.0, 1.0], 0.0))
        self.view = view

        # Splitting the loaded target in three
        self.gids = gids
        self.targets = [gids[0::3], gids[1::3], gids[2::3]]
        self.tmp = None

    def tearDown(self):
        del rtneuron.engine
        if self.tmp:
            os.remove(self.tmp)

    def _apply_colormaps(self, spikes=False):
        scene = self.view.scene
        colormaps = []
        subsets = []
        obj = scene.objects[0]
        for i in range(3):
            colormap = ColorMap()
            if spikes:
                colormap.setPoints({0: [0, 0, 0, 0],
                                    1: [i==0, i==1, i==2, 1.0]})
            else:
                colormap.setPoints({-80: [0, 0, 0, 0],
                                     -50: [i==0, i==1, i==2, 0.5],
                                     -10: [1, 1, 1, 1]})
            colormaps.append(colormap)

            subset = obj.query(self.targets[i])
            subset.attributes.colormaps = AttributeMap()
            if spikes:
                subset.attributes.colormaps.spikes = colormap
            else:
                subset.attributes.colormaps.compartments = colormap
            subset.update()
            subsets.append(subset)

        return subsets, colormaps

    def _create_spikes_file(self, gids) :
        # Generating a spike data file (the file from TestData is too sparse)
        spikefile = tempfile.NamedTemporaryFile(suffix = '.dat')
        spikes = open(spikefile.name, 'w')
        time = 0
        positions = rtneuron.simulation.open_circuit().positions(gids)

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

    def _remove_colormaps_test(self, spikes=False):
        if self.tmp:
            os.remove(self.tmp)

        rtneuron.engine.player.timestamp = 5

        # Remove from same handler
        self.tmp = image_compare.capture_temporary(self.view)
        subsets, colormaps = self._apply_colormaps(spikes)
        for subset in subsets:
            subset.attributes.colormaps = AttributeMap()
            subset.update()
        image_compare.capture_and_compare(
            self.view, self.tmp, prepend_sample_path = False)

       # Remove from top handler
        subsets, colormaps = self._apply_colormaps(spikes)
        handler = self.view.scene.objects[0]
        handler.attributes.colormaps = AttributeMap()
        handler.update()
        image_compare.capture_and_compare(
            self.view, self.tmp, prepend_sample_path = False)
        self._clear_colormaps(subsets) # To avoid warning

        # Remove from an overlapping subhandler
        subsets, colormaps = self._apply_colormaps(spikes)
        subset = handler.query(self.gids)
        subset.attributes.colormaps = AttributeMap()
        subset.update()
        image_compare.capture_and_compare(
            self.view, self.tmp, prepend_sample_path = False)
        self._clear_colormaps(subsets) # To avoid warning

    def _set_default_colormap_at_subsets(self):
        # Resetting the default color map using overlapping targets
        obj = self.view.scene.objects[0]
        subsets = []
        for i in range(3):
            subset = obj.query(self.targets[i])
            subset.attributes.colormaps = AttributeMap()
            subset.attributes.colormaps.compartments = \
                self.view.attributes.colormaps.compartments
            subset.update()
            subsets.append(subset)
        return subsets

    def _clear_colormaps(self, handlers):
        for handler in handlers:
            handler.attributes.colormaps = AttributeMap()

class TestCompartmentColormapsOnSomas(Common, unittest.TestCase) :
    def setUp(self) :
        Common.setUp(self, mode=rtneuron.RepresentationMode.SOMA)
        rtneuron.apply_compartment_report(rtneuron.simulation, self.view,
                                          'allCompartments')

    def tearDown(self) :
        Common.tearDown(self)

    def test_apply_colormaps(self):
        rtneuron.engine.player.timestamp = 5.5
        self.tmp = image_compare.capture_temporary(self.view)

        subsets, colormaps = self._apply_colormaps()
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps_on_soma.png')

        colormaps[0].setRange(-80, -60)
        colormaps[1].setRange(-70, -60)
        colormaps[2].setRange(-70, -50)
        for s in subsets:
            s.update()
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps_on_soma_2.png')
        self._clear_colormaps(subsets) # To avoid warning

        subsets = self._set_default_colormap_at_subsets()
        image_compare.capture_and_compare(
            self.view, self.tmp, prepend_sample_path = False)
        self._clear_colormaps(subsets) # To avoid warning

    def test_remove_colormaps(self):
        self._remove_colormaps_test()

    def test_colormaps_with_transparency(self):
        rtneuron.engine.player.timestamp = 5.5
        scene = self.view.scene
        subsets, colormaps = self._apply_colormaps()

        sceneops.enable_transparency(scene)
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps_on_soma_transparent.png')

        sceneops.disable_transparency(scene)
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps_on_soma.png')
        self._clear_colormaps(subsets) # To avoid warning

class TestCompartmentColormaps(Common, unittest.TestCase) :
    def setUp(self) :
        Common.setUp(self)
        rtneuron.apply_compartment_report(rtneuron.simulation, self.view,
                                          'allCompartments')

    def tearDown(self) :
        Common.tearDown(self)

    def test_apply_colormaps(self):
        rtneuron.engine.player.timestamp = 5.5
        self.tmp = image_compare.capture_temporary(self.view)

        subsets, colormaps = self._apply_colormaps()
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps.png')

        colormaps[0].setRange(-80, -60)
        colormaps[1].setRange(-70, -60)
        colormaps[2].setRange(-70, -50)
        for s in subsets:
            s.update()
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps_2.png', threshold=2)
        self._clear_colormaps(subsets) # To avoid warning

        subsets = self._set_default_colormap_at_subsets()
        image_compare.capture_and_compare(
            self.view, self.tmp, prepend_sample_path = False)
        self._clear_colormaps(subsets) # To avoid warning

    def test_remove_colormaps(self):
        self._remove_colormaps_test()

    def test_colormaps_with_transparency(self):
        rtneuron.engine.player.timestamp = 5.5
        scene = self.view.scene
        subsets, colormaps = self._apply_colormaps()

        sceneops.enable_transparency(scene)
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps_transparent.png')

        sceneops.disable_transparency(scene)
        image_compare.capture_and_compare(
            self.view, 'compartments_with_colormaps.png')
        self._clear_colormaps(subsets) # To avoid warning

class TestSpikeColormapsOnSomas(Common, unittest.TestCase) :

    def setUp(self) :
        Common.setUp(self, mode=rtneuron.RepresentationMode.SOMA)

        self._spikefile = self._create_spikes_file(self.gids)
        rtneuron.apply_spike_data(self._spikefile.name, self.view)
        rtneuron.engine.player.window = [0, 6]

        # To make sure the scene is ready before the timestamps are modified
        rtneuron.engine.frame()
        rtneuron.engine.frame()

    def tearDown(self) :
        Common.tearDown(self)

    def test_apply_colormaps(self):
        rtneuron.engine.player.timestamp = 5
        subsets, colormaps = self._apply_colormaps(spikes=True)
        image_compare.capture_and_compare(
            self.view, 'spikes_with_colormaps_on_soma.png')
        self._clear_colormaps(subsets) # To avoid warning

    def test_remove_colormaps(self):
        self._remove_colormaps_test(spikes=True)

class TestSpikeColormaps(Common, unittest.TestCase) :

    def setUp(self) :
        gids = [400, 406, 412]
        Common.setUp(self, gids=gids)

        self.view.attributes.spike_tail = 2

        self._spikefile = self._create_spikes_file(gids)
        rtneuron.apply_spike_data(self._spikefile.name, self.view)
        rtneuron.engine.player.window = [0, 6]

    def tearDown(self) :
        Common.tearDown(self)

    def test_apply_colormaps(self):
        rtneuron.engine.player.timestamp = 5
        subsets, colormaps = self._apply_colormaps(spikes=True)
        image_compare.capture_and_compare(
            self.view, 'spikes_with_colormaps.png')
        self._clear_colormaps(subsets) # To avoid warning

    def test_remove_colormaps(self):
        self._remove_colormaps_test(spikes=True)

if __name__ == '__main__':
    unittest.main()
