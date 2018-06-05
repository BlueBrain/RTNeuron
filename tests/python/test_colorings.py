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

class Common:
    def setUp(self, sceneAttr = AttributeMap()):
        self._engine = RTNeuron([])
        self._simulation = brain.Simulation(brain.test.blue_config)
        self._gids = [406]

        self._scene = self._engine.createScene(sceneAttr)
        self._scene.circuit = self._simulation.open_circuit()

        self._engine.init(setup.medium_rect_eq_config)
        view = self._engine.views[0]
        self._view = view
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        view.attributes.auto_compute_home_position = False
        view.camera.setView([29, 666, 563], ([0.0, 0.0, 1.0], 0.0))
        view.scene = self._scene

    def tearDown(self):
        self._engine.exit()
        del self._engine

class TestMode(Common):

    def add_neurons(self, scheme, primary=[1, 0, 0, 0.1],
                   secondary=[0, 0.2, 1, 1], extra = None):
        attributes = AttributeMap()
        attributes.color_scheme = scheme
        attributes.primary_color = primary
        attributes.secondary_color = secondary
        if extra:
            attributes.extra = extra
        self._handler = self._scene.addNeurons(self._gids, attributes)

    def do_test_mode(self, scheme):
        self.add_neurons(scheme)
        # Adding the neurons to the scene with the requested coloring scheme
        image_compare.capture_and_compare(
            self._view, "coloring_%s.png" % str(scheme).lower())

class TestColorSchemesAtConstruction(unittest.TestCase, TestMode):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def test_solid(self):
        self.do_test_mode(ColorScheme.SOLID)

    def test_by_branch_type_(self):
        self.do_test_mode(ColorScheme.BY_BRANCH_TYPE)

    def test_alpha_by_distance_to_soma(self):
        self.do_test_mode(ColorScheme.BY_DISTANCE_TO_SOMA)

    def test_alpha_by_width(self):
        self.do_test_mode(ColorScheme.BY_WIDTH)


class TestColorSchemeChangesMetaclass(type):
    '''This class creates the test cases for representation mode changes
    programmatically. The alternative to this slighty convoluted code is to
    create a single test case or listing all the combinations by hand. The
    former runs faster but it's a single test and makes it more difficult
    identify which combination failed. The latter is not an option.'''
    def __new__(cls, name, bases, attributes):
        values = [ColorScheme.SOLID, ColorScheme.BY_DISTANCE_TO_SOMA,
                  ColorScheme.BY_WIDTH, ColorScheme.BY_BRANCH_TYPE]
        # Function used to create the test method where the arguments for
        # do_test_mode_change are bound
        def test_closure(first, second):
            def test(self):
                self.do_test_color_scheme_change(first, second)
            return test

        import itertools
        for first, second in itertools.product(values, values):
            if first == second:
                continue
            attributes['test_%s_to_%s' % (first, second)] = \
                test_closure(first, second)
        return type.__new__(cls, name, bases, attributes)


class TestModeChanges(six.with_metaclass(TestColorSchemeChangesMetaclass,
                                         unittest.TestCase, TestMode)):

    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def change_coloring(self, scheme):
        obj = self._scene.objects[0]
        obj.attributes.color_scheme = scheme
        obj.update()

    def do_test_color_scheme_change(self, first, second):
        self.add_neurons(first)
        # Making sure the scene is displayed before the mode is changed
        for i in range(2):
            self._engine.frame()
            self._engine.waitFrame()
        self.change_coloring(second)
        image_compare.capture_and_compare(
            self._view, "coloring_%s.png" % str(second).lower())

class TestParameterChanges(unittest.TestCase, TestMode):

    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def test_attenuation_changes(self):
        extra = AttributeMap()
        extra.attenuation = 2.0
        self.add_neurons(ColorScheme.BY_WIDTH, extra=extra)
        for i in range(2):
            self._engine.frame()
            self._engine.waitFrame()
        old_attenuation = self._handler.attributes.extra.attenuation
        self._handler.attributes.extra.attenuation = 4.0
        self._handler.update()

        def capture():
            image_compare.capture_and_compare(
                self._view, "coloring_by_width.png")

        self.assertRaises(AssertionError, capture)

        self._handler.attributes.extra.attenuation = old_attenuation
        self._handler.update()
        capture()

    def test_color_changes_with_distance_to_soma(self):
        self.add_neurons(ColorScheme.BY_DISTANCE_TO_SOMA)

        for i in range(2):
            self._engine.frame()
            self._engine.waitFrame()
        old_color = self._handler.attributes.primary_color
        self._handler.attributes.primary_color = [0, 1, 0, 1]
        self._handler.update()

        def capture():
            image_compare.capture_and_compare(
                self._view,"coloring_by_distance_to_soma.png")

        self.assertRaises(AssertionError, capture)

        self._handler.attributes.primary_color = old_color
        self._handler.update()
        capture()


if __name__ == '__main__':
    unittest.main()
