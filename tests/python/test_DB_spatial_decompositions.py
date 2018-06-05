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
import sys

import unittest

del os.environ['EQ_WINDOW_IATTR_PLANES_SAMPLES']  # No hw AA

circuit_config = brain.test.blue_config

def merge(target, source):
    for attr in dir(source):
        target.__setattr__(attr, source.__getattr__(attr))

class Common:
    def create_config(self, common_attributes):
        self.engine = RTNeuron([])

        self.scenes = {}
        self.scenes['regular'] = self.engine.createScene(common_attributes)
        attributes = AttributeMap({'mesh_based_partition': True})
        merge(attributes, common_attributes)
        self.scenes['mesh_partition'] = self.engine.createScene(attributes)
        attributes = AttributeMap({'use_meshes': False})
        merge(attributes, common_attributes)
        self.scenes['no_meshes'] = self.engine.createScene(attributes)

        for scene in self.scenes.values():
            scene.addNeurons([394, 395, 396, 407, 410, 412])

        self.engine.init(setup.DB_ranges_config)
        self.engine.resume()

    def run_partition_test(self, layout, scene):
        view = self.engine.views[0]
        view.scene = None
        self.engine.useLayout(layout)
        # The view changes after the layout change
        view = self.engine.views[0]
        view.scene = self.scenes[scene]
        view.attributes.auto_compute_home_position = False
        view.camera.setView([132.875, 611.087, 656.934],
                            ([-0.930, 0.359, 0.0677], 19.478))

        file_name = 'DB_' + scene + '_' + layout + '.png'
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        self.engine.frame()
        image_compare.capture_and_compare(view, file_name, 1)

class TestScenarioMetaclass(type):
    '''This class creates the test cases for the different scenarios
    programmatically. The alternative is a bit of copy paste.'''
    def __new__(cls, name, bases, attributes):

        layouts = {'half': '0-0.5', 'quarter': '0.25-0.5',
                   'eight': '0.25-0.375'}

        # Function used to create the test method where the arguments for
        # do_test_mode_change are bound
        def test_closure(layout, partition_type):
            def test(self):
                self.run_partition_test(layout, partition_type)
            return test

        import itertools
        sizes = ['half', 'quarter', 'eight']
        partitions = ['regular', 'mesh_partition', 'no_meshes']

        for first, second in itertools.product(sizes, partitions):
            attributes['test_%s_%s' % (first, second)] = \
                test_closure(layouts[first], second)
        return type.__new__(cls, name, bases, attributes)

class TestSharedMorphologies(six.with_metaclass(TestScenarioMetaclass,
                                                unittest.TestCase, Common)):

    def setUp(self):
        self.create_config(
            AttributeMap({'circuit': circuit_config,
                          'partitioning': DataBasePartitioning.SPATIAL}))

    def tearDown(self):
        del self.engine
        del self.scenes

class TestUniqueMorphologies(unittest.TestCase, Common):
    __metaclass__ = TestScenarioMetaclass

    def setUp(self):
        self.create_config(
            AttributeMap({'circuit': circuit_config,
                          'partitioning': DataBasePartitioning.SPATIAL,
                          'unique_morphologies': True}))

    def tearDown(self):
        del self.engine
        del self.scenes

class TestConservativeLoading(unittest.TestCase, Common):
    __metaclass__ = TestScenarioMetaclass

    def setUp(self):
        self.create_config(
            AttributeMap({'circuit': circuit_config,
                          'partitioning': DataBasePartitioning.SPATIAL,
                          'unique_morphologies': True,
                          'conservative_loading': True}))

    def tearDown(self):
        del self.engine
        del self.scenes

if __name__ == '__main__':
    unittest.main()
