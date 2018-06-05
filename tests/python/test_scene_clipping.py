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
import brain

import unittest


class SceneClipping(unittest.TestCase):

    def setUp(self):

        setup.setup_simple(
            self, 200, blue_config=brain.test.circuit_config,
            eq_config=setup.small_rect_eq_config)

        view = self.view
        view.camera.setView([204, 1500, 170], ([0, 1, 0], 45))
        view.attributes.auto_compute_home_position = False

        x = -75 + 34; y = 1450; z = -75 + 32;
        X = x + 150; Y = y + 100; Z = z + 150;
        self.view.scene.addGeometry(
            [[x, y, z], [X, y, z], [X, Y, z], [x, Y, z],
             [x, y, Z], [X, y, Z], [X, Y, Z], [x, Y, Z]],
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]], colors=[1, 1, 1, 1])

    def tearDown(self):
        self.engine.exit()
        del self.engine

    def test_add_clip_plane(self):
        view = self.view
        image_compare.capture_and_compare(view, "scene_clipping1.png")
        view.scene.setClipPlane(0, [0, 0, -1, 75 + 32.00001])
        image_compare.capture_and_compare(view, "scene_clipping2.png")

    def test_get_clip_plane(self):
        view = self.view
        self.assertRaises(RuntimeError, Scene.getClipPlane, view.scene, 1)
        view.scene.setClipPlane(1, [0, 0, -1, 1])
        assert([0, 0, -1, 1] == view.scene.getClipPlane(1))
        self.assertRaises(RuntimeError, Scene.getClipPlane, view.scene, 0)
        self.assertRaises(RuntimeError, Scene.getClipPlane, view.scene, 2)
        view.scene.clearClipPlanes()
        self.assertRaises(RuntimeError, Scene.getClipPlane, view.scene, 1)

    def test_add_clip_planes(self):
        view = self.view
        image_compare.capture_and_compare(view, "scene_clipping1.png")
        view.scene.setClipPlane(0, [0, 0, -1, 75 + 32.00001])
        image_compare.capture_and_compare(view, "scene_clipping2.png")
        # This clipping plane clips one of the faces of the box
        view.scene.setClipPlane(1, [-1, 0, 0, 74 + 34])
        image_compare.capture_and_compare(view, "scene_clipping3.png")

    def test_add_modify_clip_plane(self):
        view = self.view
        view.scene.setClipPlane(0, [0, 0, -1, 0])
        view.scene.setClipPlane(1, [-1, 0, 0, 0])
        self.engine.frame()
        view.scene.setClipPlane(0, [0, 0, -1, 75 + 32.00001])
        view.scene.setClipPlane(1, [-1, 0, 0, 74 + 34])
        image_compare.capture_and_compare(view, "scene_clipping3.png")

    def test_remove_clip_planes(self):
        view = self.view
        scene = view.scene
        scene.setClipPlane(0, [0, 0, -1, 75 + 32.00001])
        self.engine.frame()
        scene.clearClipPlanes()
        image_compare.capture_and_compare(view, "scene_clipping1.png")

        scene.setClipPlane(0, [0, 0, -1, 75 + 32.00001])
        scene.setClipPlane(1, [-1, 0, 0, 74 + 34])
        self.engine.frame()
        scene.clearClipPlanes()
        image_compare.capture_and_compare(view, "scene_clipping1.png")

if __name__ == '__main__':
    unittest.main()
