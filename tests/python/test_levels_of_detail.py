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
from rtneuron import AttributeMap

import unittest

attributes = []
no_mesh = AttributeMap({'use_meshes' : False})
no_connect = AttributeMap({'connect_first_order_branches' : False})
# In each LOD to test, the first argument is the test name, the second is
# the LOD name, the third an extra attribute map to use for the scene
# creation. The golden sample file name is the same as the test name with
# the prefix "lod_" and ".png" extension
lods = [("tubelets_no_mesh", "tubelets", no_mesh),
        ("tubelets_no_connect", "tubelets", no_connect),
        ("tubelets", "tubelets", None),
        ("smooth_tubelets", "tubelets",
         AttributeMap({"smooth_tubelets" : True})),
        ("cylinders_no_mesh", "high_detail_cylinders", no_mesh),
        ("cylinders_no_connect", "high_detail_cylinders", no_connect),
        ("cylinders", "high_detail_cylinders", None),
        ("low_cylinders", "low_detail_cylinders", None),
        ("mesh", "mesh", None),
        ("detailed_soma", "detailed_soma", None),
        ("soma", "spherical_soma", None)]

class TestLevelsOfDetail(unittest.TestCase) :
    def setUp(self) :
        setup.setup_lod_scenes(self, lods,
                               AttributeMap({'color' : [0.7, 0.85, 1.0, 1.0]}))

    def tearDown(self) :
        del self.engine
        del self.scenes
        del self.view

    def test_mesh(self) :
        self.run_lod_test('mesh')

    def test_soma(self) :
        self.run_lod_test('soma')

    def test_inflated_soma(self) :
        self.run_lod_test('soma', 1.0)

    def test_detailed_soma(self) :
        self.run_lod_test('detailed_soma')

    def test_cylinders_no_mesh(self) :
        self.run_lod_test('cylinders_no_mesh')

    def test_cylinders_no_connect(self) :
        self.run_lod_test('cylinders_no_connect')

    def test_cylinders(self) :
        self.run_lod_test('cylinders')

    def test_inflated_cylinders(self) :
        self.run_lod_test('cylinders', 1.0)

    def test_low_cylinders(self) :
        self.run_lod_test('low_cylinders')

    # Tubelets are rendered slightly different in GL2 and GL3, so the threshold
    # needs to be increased to use the same ground truth images.

    def test_tubelets_no_mesh(self) :
        self.run_lod_test('tubelets_no_mesh')

    def test_tubelets_no_connect_mesh(self) :
        self.run_lod_test('tubelets_no_connect')

    def test_smooth_tubelets(self) :
        self.run_lod_test('smooth_tubelets')

    def test_inflated_tubelets(self) :
        self.run_lod_test('tubelets')

    def test_tubelets(self) :
        self.run_lod_test('tubelets')

    def test_model_inflation(self) :
        scene = self.scenes["mesh"]
        # Testing 1.0 factor inflation
        self.view.scene = scene
        scene.attributes.inflatable_neurons = True
        self.view.attributes.inflation_factor = 1.0
        image_compare.capture_and_compare(self.view, "lod_mesh_1.0.png")
        # Reset inflation to false
        scene.attributes.inflatable_neurons = False
        image_compare.capture_and_compare(self.view, "lod_mesh.png")
        # Modifying inflation factor. Model must be unaffected because
        # inflation is disabled
        self.view.attributes.inflation_factor = 2.0
        image_compare.capture_and_compare(self.view, "lod_mesh.png")
        # Enabling inflation
        scene.attributes.inflatable_neurons = True
        image_compare.capture_and_compare(self.view, "lod_mesh_2.0.png")
        self.view.attributes.inflation_factor = 1.0
        image_compare.capture_and_compare(self.view, "lod_mesh_1.0.png")


    def run_lod_test(self, scene_name, inflation=0) :
        scene = self.scenes[scene_name]
        self.view.scene = scene
        if inflation != 0 :
            scene.attributes.inflatable_neurons = True
            self.view.attributes.inflation_factor = inflation
            file_name = 'lod_' + scene_name + '_%.1f.png' % inflation
        else :
            scene.attributes.inflatable_neurons = False
            file_name = 'lod_' + scene_name + '.png'
        self.engine.frame()
        image_compare.capture_and_compare(self.view, file_name)

if __name__ == '__main__':
    unittest.main()





