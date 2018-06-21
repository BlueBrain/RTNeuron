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

import six

import setup
import image_compare
from rtneuron import *
import brain

import unittest
import os

class Common:
    def setUp(self, sceneAttr = AttributeMap()):
        self._engine = RTNeuron([])

        self._gids = [310, 320, 330, 340, 350]
        self._scene = self._engine.createScene(sceneAttr)
        self._scene.circuit = brain.Circuit(brain.test.circuit_config)

        self._engine.init(setup.medium_rect_eq_config)
        view = self._engine.views[0]
        self._view = view
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        view.attributes.auto_compute_home_position = False
        view.camera.setView([51, 1414, -416],
                            ([0.00431, 0.99360, 0.11283], 177.39))
        view.scene = self._scene

    def tearDown(self):
        self._engine.exit()
        del self._engine

class TestMode(Common):
    def do_test_mode(self, mode):
        self._scene.addNeurons(self._gids, AttributeMap({'mode': mode}))
        image_compare.capture_and_compare(
            self._view, "repr_mode_%s.png" % str(mode).lower())

class TestModesAtConstruction(unittest.TestCase, TestMode):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def test_display_soma(self):
        self.do_test_mode(RepresentationMode.SOMA)

    def test_display_skeleton(self):
        self.do_test_mode(RepresentationMode.SEGMENT_SKELETON)

    def test_display_whole_neuron(self):
        self.do_test_mode(RepresentationMode.WHOLE_NEURON)

    def test_display_no_axon(self):
        self.do_test_mode(RepresentationMode.NO_AXON)

class TestModesAtConstructionWithUniqueMorphologies(unittest.TestCase,
                                                    TestMode):
    def setUp(self):
        Common.setUp(self, AttributeMap({'unique_morphologies': True}))

    def tearDown(self):
        Common.tearDown(self)

    def test_display_no_axon(self):
        self.do_test_mode(RepresentationMode.NO_AXON)

class TestModeChangesMetaclass(type):
    '''This class creates the test cases for representation mode changes
    programmatically. The alternative to this slighty convoluted code is to
    create a single test case or listing all the combinations by hand. The
    former runs faster but it's a single test and makes it more difficult
    identify which combination failed. The latter is not an option.'''
    def __new__(cls, name, bases, attributes):
        values = list(RepresentationMode.values.values())[:4]
        # Function used to create the test method where the arguments for
        # do_test_mode_change are bound

        def test_closure(first, second):
            def test(self):
                self.do_test_mode_change(
                    first, second, (first, second) in self.bad_changes)
            return test

        import itertools
        for first, second in itertools.product(values, values):
            if first == second:
                continue
            attributes['test_%s_to_%s' % (first, second)] = \
                test_closure(first, second)
        return type.__new__(cls, name, bases, attributes)

class TestModeChanges(six.with_metaclass(TestModeChangesMetaclass,
                                         unittest.TestCase, Common)):
    '''There's no need to test unique morphologies here because the changing
    from NO_AXON to WHOLE_NEURON is not allowed in either case and the rest
    of the combinations don't use code specific for unique morphologies'''

    bad_changes = set([
            (RepresentationMode.NO_AXON, RepresentationMode.WHOLE_NEURON)])

    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def change_mode(self, mode):
        obj = self._scene.objects[0]
        obj.attributes.mode = mode
        obj.update()

    def do_test_mode_change(self, first_mode, second_mode, throws = False):
        self._scene.addNeurons(
            self._gids, AttributeMap({'mode': first_mode}))
        # Making sure the scene is displayed before the mode is changed
        for i in range(2):
            self._engine.frame()
            self._engine.waitFrame()
        if throws:
            self.assertRaises(RuntimeError,
                              lambda: self.change_mode(second_mode))
        else:
            self.change_mode(second_mode)
            image_compare.capture_and_compare(
                self._view, "repr_mode_%s.png" % str(second_mode).lower())

class TestChangeToNoDisplay(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def test_from_soma(self):
        self._test_mode_to_no_display(RepresentationMode.SOMA)

    def test_from_whole_neuron(self):
        self._test_mode_to_no_display(RepresentationMode.WHOLE_NEURON)

    def _test_mode_to_no_display(self, mode):
        tmp = image_compare.capture_temporary(self._view)

        self._scene.addNeurons(self._gids, AttributeMap({'mode': mode}))
        for i in range(2):
            self._engine.frame()
            self._engine.waitFrame()

        obj = self._scene.objects[0]
        obj.attributes.mode = RepresentationMode.NO_DISPLAY
        obj.update()
        try:
            image_compare.capture_and_compare(
                self._view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

if __name__ == '__main__':
    unittest.main()
