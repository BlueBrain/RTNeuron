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

class Common :
    def create_config(self, config) :
        attributes = AttributeMap()
        attributes.partitioning = DataBasePartitioning.SPATIAL
        setup.setup_simple(self, [394, 395, 396, 407, 410, 412],
                           scene_attributes=attributes, eq_config=config)

        self.engine.resume()

        self.view.attributes.idle_AA_steps = 32
        # It's important to use a background different from black
        self.view.attributes.background = [0.1, 0.2, 0.3, 1]

        self.cameras = {
            'view1' : ([-11, 797, 209], ([-0.92166, -0.35207, -0.163039], 60)),
            'view2' : ([0, 386, 230],([0.93759, -0.30783, 0.16175], 55))}

    def run_assembly_test(self, camera) :
        self.view.camera.setView(*self.cameras[camera])
        file_name = "DB_assembly_" + camera + '.png'
        image_compare.capture_and_compare(self.view, file_name, 1)

class TestVanillaDB(unittest.TestCase, Common) :
    def setUp(self) :
        self.create_config(setup.vanilla_DB_config)

    def tearDown(self) :
        del self.engine

    def test_view1(self) :
        self.run_assembly_test('view1')

    def test_view2(self) :
        self.run_assembly_test('view2')

class TestDirectSend(unittest.TestCase, Common) :
    def setUp(self) :
        self.create_config(setup.direct_send_DB_config)

    def tearDown(self) :
        del self.engine

    def test_view1(self) :
        self.run_assembly_test('view1')

    def test_view2(self) :
        self.run_assembly_test('view2')


if __name__ == '__main__':
    unittest.main()
