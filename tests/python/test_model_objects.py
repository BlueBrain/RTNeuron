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

import math
import os

import unittest

cube_path = setup.misc_data_path + "/cube.obj"

class TestAddModel(unittest.TestCase):
    def setUp(self):
        setup.setup_empty_scene(self, eq_config=setup.small_eq_config)
        self.view.attributes.background = [0.5, 0.5, 0.5, 1]

    def tearDown(self):
        self.scene = None
        self.engine.exit()
        del self.engine

    def test_add_simple_obj_model(self):
        self.scene.addModel(cube_path)

        image_compare.capture_and_compare(
            self.view, "cube.png")

    def test_add_with_color(self):
        attributes = AttributeMap()
        attributes.color = [1, 0, 0, 1]
        self.scene.addModel(cube_path, attributes=attributes)

        image_compare.capture_and_compare(
            self.view, "red_cube.png")

    def test_add_with_color_flat(self):
        attributes = AttributeMap()
        attributes.color = [1, 0, 0, 1]
        attributes.flat = True
        self.scene.addModel(cube_path, attributes=attributes)

        image_compare.capture_and_compare(
            self.view, "red_cube.png")

    def test_add_and_change_color(self):
        self.scene.addModel(cube_path)

        cube = self.scene.objects[0]
        cube.attributes.color = [1, 0, 0, 1]
        cube.update()

        image_compare.capture_and_compare(
            self.view, "red_cube.png")

    def test_add_models_with_transforms(self):
        for i in range(30):
            angle = math.radians(12 * i)
            p = [350 * math.cos(angle), 350 * math.sin(angle), 0]
            s = 1 + math.cos(angle * 4) * 0.5

            transform = ("t@-50,50,00:" + # To center the cube
                         "s@%f,%f,%f:" % (s, s, s) +
                         "r@0,0,1,%f:" % (12.0 * i) +
                         "t@%f,%f,%f:" % (p[0], p[1], p[2]))
            self.scene.addModel(cube_path, transform)
        self.view.camera.setViewLookAt([0, 0, 2800], [0, 0, 0], [0, 1, 0])

        image_compare.capture_and_compare(
            self.view, "cubes.png")

    def test_add_remove(self):
        handler = self.scene.addModel(cube_path)

        tmp = image_compare.capture_temporary(self.view)
        try:
            self.scene.remove(handler)
            image_compare.capture_and_compare(self.view, 'empty.png', 1, tmp)
        except:
            os.remove(tmp)

class TestAddGeometry(unittest.TestCase):

    def setUp(self):
        setup.setup_empty_scene(self, eq_config=setup.small_eq_config)
        self.view.attributes.background = [0.5, 0.5, 0.5, 1]

    def tearDown(self):
        self.scene = None
        self.engine.exit()
        del self.engine

    def create_vertices(self, radius=10, count=12):
        vertices = []
        for i in range(count):
            angle = math.radians(360 * i / count)
            vertices.append(
                [math.cos(angle) * radius, math.sin(angle) * radius, 0])
        return vertices

    def add_points_to_scene(self, style, size_multiplier=1):
        attributes = AttributeMap()
        attributes.point_style = style
        vertices = [[v[0], v[1], v[2], size_multiplier]
                    for v in self.create_vertices(10)]
        self.scene.addGeometry(vertices, attributes=attributes)
        colors = [[i/11.0, 0, 0, 1] for i in range(12)]
        attributes.point_size = 2 * size_multiplier
        self.scene.addGeometry(self.create_vertices(20), colors=colors,
                               attributes=attributes)
        self.view.camera.setViewLookAt([0, 0, 100], [0, 0, 0], [0, 1, 0])

    def test_add_points_as_spheres(self):
        self.add_points_to_scene("spheres")
        image_compare.capture_and_compare(
            self.view, "spheres.png")

    def test_add_points_as_points(self):
        self.add_points_to_scene("points", 20)
        image_compare.capture_and_compare(
            self.view, "points.png")

    def test_add_points_as_round_points(self):
        self.add_points_to_scene("circles", 20)
        image_compare.capture_and_compare(
            self.view, "circles.png")

    def test_add_prism(self):
        center = [100, 100, 100]
        add_hexagonal_prism(
            self.scene, [100, 100, 100], 80, 40, color=[1, 0, 0, 1])
        apotheme = 40 * math.sqrt(3) * 0.5
        self.view.camera.setViewLookAt(
            [100, 140, -100 - apotheme], [100, 140, 100], [0, 1, 0])
        self.view.camera.setProjectionFrustum(-0.02, 0.02, -0.02, 0.02, 0.1)

        image_compare.capture_and_compare(
            self.view, "hexagonal_prism.png")


if __name__ == '__main__':
    unittest.main()
