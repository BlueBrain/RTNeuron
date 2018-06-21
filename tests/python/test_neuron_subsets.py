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

import unittest

class TestSubsetDisplay(unittest.TestCase) :
    def setUp(self):
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.SOMA
        attributes.color = [1, 1, 1, 1]
        setup.setup_simple(self, 'L5CSPC', target_attributes=attributes)

        self._gids = self.simulation.gids('L5CSPC')

        view = self.view
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        view.attributes.auto_compute_home_position = False
        view.camera.setView([49, 630, 509], ([0.0, 0.0, 1.0], 0.0))

    def tearDown(self):
        self.engine.exit()
        del self.engine

    def test_subset_attributes1(self):
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t1_somas.png")

        subset = self.scene.objects[0].query(self._gids[0::2])
        subset.attributes.color = [1, 0, 0, 1]
        subset.update()
        image_compare.capture_and_compare(
            self.view, "half_red_half_white_somas.png")

    def test_subset_attributes2(self):
        # Forcing first two frames to ensure the scene is created,
        # otherwise the attribute updates will be overriden by the parent
        # ones.
        self.engine.frame()
        self.engine.frame()

        parent = self.scene.objects[0]
        subset1 = parent.query(self._gids[0::2])
        subset2 = parent.query(self._gids[1::2])

        parent.attributes.color = [0, 0, 0, 1]
        parent.update()
        subset1.attributes.color = [1, 0, 0, 1]
        subset1.update()
        subset2.attributes.color = [1, 1, 1, 1]
        subset2.update()
        image_compare.capture_and_compare(
            self.view, "half_red_half_white_somas.png")

        fullset = parent.query(self._gids)
        fullset.attributes.color = [1, 1, 0, 1]
        fullset.update()
        # Now calling update again on subset1 and subset2 must reapply their
        # attributes.
        subset1.update()
        subset2.update()
        image_compare.capture_and_compare(
            self.view, "half_red_half_white_somas.png")

        del subset1
        del subset2
        parent.attributes.color = [1, 1, 1, 1]
        parent.update()
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t1_somas.png")

    def test_create_update_remove_subset(self):
        # Forcing first two frames to ensure the scene is created,
        # otherwise the attribute updates will be overriden by the parent
        # ones.
        self.engine.frame()
        self.engine.frame()

        parent = self.scene.objects[0]
        subset = parent.query(self._gids[0::2])
        subset.attributes.color = [1, 0, 0, 1]
        subset.update()
        del subset

        image_compare.capture_and_compare(
            self.view, "half_red_half_white_somas.png")

    def test_update_child_attributes(self):
        self.engine.frame()
        self.engine.frame()

        parent = self.scene.objects[0]
        subset = self.scene.objects[0].query(self._gids[0::2])

        color1 = [1, 0, 0, 1]
        color2 = [0, 0, 1, 1]

        parent.attributes.color = color1
        parent.update()
        self.view.snapshot("parent_default.png")

        parent.attributes.color = color2
        subset.attributes.color = color2
        parent.update()
        subset.update()
        assert(parent.attributes.color == color2)
        assert(subset.attributes.color == color2)

        parent.attributes.color = color1
        parent.update()
        self.view.snapshot("parent_reset.png")
        assert(parent.attributes.color == color1)
        assert(subset.attributes.color == color1)

        try:
            image_compare.compare("parent_default.png", "parent_reset.png")
        finally:
            import os
            if 'KEEP_IMAGES' not in os.environ:
                os.remove('parent_default.png')
                os.remove('parent_reset.png')


class TestSimpleOperations(unittest.TestCase):
    def setUp(self):
        setup.setup_simple(self, 'MiniColumn_0')
        self._gids = self.simulation.gids('MiniColumn_0')

    def tearDown(self):
        del self.engine

    def test_create_subset(self):
        subset = self.scene.objects[0].query(self._gids[0::2])

        assert(all(subset.object == self._gids[0::2]))

    def test_create_empty_subset(self):
        subset = self.scene.objects[0].query([])
        assert(len(subset.object) == 0)

    def test_invalid_subset(self):
        subset = self.scene.objects[0].query([1000000])
        assert(len(subset.object) == 0)

    def test_create_subset_from_subset(self):
        subset1 = self.scene.objects[0].query(self._gids[0::2])
        subsubset = self._gids[0::2][0::2]
        subset2 = subset1.query(subsubset)

        neurons = subset2.query(self._gids[0::2]).object
        assert(len(neurons) == len(subset2.object))

        neurons = subset2.object
        assert(all(neurons == subsubset))

    def test_update_parent_attributes(self):
        parent = self.scene.objects[0]
        subset1 = parent.query(self._gids[0::2])
        subset2 = subset1.query(self._gids[1::2])

        color = [0.1, 0.2, 0.3, 0.4]
        parent.attributes.color = color
        parent.update()
        assert(subset1.attributes.color == color)
        assert(subset2.attributes.color == color)

        subset1.attributes.color = [0.4, 0.3, 0.2, 0.1]
        subset1.update()
        assert(parent.attributes.color == color)
        assert(subset2.attributes.color == color)

    def test_updates_subset_then_parent(self):
        parent = self.scene.objects[0]
        parent.attributes.color = [1, 1, 1, 1]
        parent.update()

        subset = parent.query(self._gids[0::2])
        assert(subset.attributes.color == [1, 1, 1, 1])

        subset.attributes.color = [0.4, 0.3, 0.2, 0.1]
        subset.update()

        parent.update()
        assert(parent.attributes.color == [1, 1, 1, 1])
        assert(subset.attributes.color == [1, 1, 1, 1])

    def test_subset_invalid_handler(self):
        parent = self.scene.objects[0]
        subset = parent.query(self._gids[0::2])

        # This invalidates the scene and should invalidate all handlers
        del self.engine
        self.engine = RTNeuron([])

        self.assertRaises(RuntimeError, lambda: list(self.scene.update()))
        self.assertRaises(RuntimeError, lambda: list(parent.update()))
        self.assertRaises(RuntimeError, lambda: list(subset.update()))

    def test_subset_invalid_handler2(self):
        parent = self.scene.objects[0]
        subset1 = parent.query(self._gids[0::2])
        subset2 = subset1.query(self._gids[0::2][0::2])

        self.engine.exit()
        del subset1
        del self.scene
        self.assertRaises(RuntimeError, lambda: list(subset2.query([])))
        self.assertRaises(RuntimeError, lambda: list(subset2.update()))

if __name__ == '__main__':
    unittest.main()

