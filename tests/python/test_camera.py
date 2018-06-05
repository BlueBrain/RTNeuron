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

import sys
import os
import math
from numpy import array

import setup
import image_compare
import unittest

import rtneuron
import brain

os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12' # FBO
del os.environ['EQ_WINDOW_IATTR_PLANES_SAMPLES']    # No hw AA

class Common(object):
    def setUp(self):
        rtneuron.display_circuit(brain.test.blue_config,
            ('L5CSPC', {'mode': rtneuron.RepresentationMode.SOMA}),
            eq_config = setup.small_eq_config)

        self.view = rtneuron.engine.views[0]
        self.view.attributes.background = [0, 0, 0, 1]
        # So results are not GPU/drivers dependant
        self.view.attributes.idle_AA_steps = 32
        # Forcing the model scale to be the one that was badly computed before
        # the fix in 57c463b
        self.view.attributes.model_scale = 390.60089111328125
        # To make sure the scene is ready
        rtneuron.engine.frame()
        rtneuron.engine.waitFrame()

    def tearDown(self):
        del self.view
        del rtneuron.engine

class TestCameraView(Common, unittest.TestCase):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def test_get_initial_view(self):
        assert(self.view.camera.getView() ==
            ([48.436325073242188, 640.196533203125, 1010.5360107421875],
             ([0.0, 0.0, 1.0], 0.0)))

    def test_set_get_view(self):
        position = [30, 600, 300]
        orientation = ([0, 0, 1], 0)
        assert(self.view.camera.getView() != (position, orientation))
        self.view.camera.setView(position, orientation)
        assert(self.view.camera.getView() == (position, orientation))
        image_compare.capture_and_compare( self.view, "camera_setView.png")

    def test_set_get_viewLookAt(self):
        eye = [30, 600, 300]
        up = [0, 1, 0]

        center = [30, 600, 3001]
        self.view.camera.setViewLookAt(eye, center, up)
        assert(self.view.camera.getView() ==
               ([30, 600, 300], ([0, 1, 0], 180.0)))

        center = [30, 600, 299]
        self.view.camera.setViewLookAt(eye, center, up)
        assert(self.view.camera.getView() == ([30, 600, 300], ([0, 0, 1], 0.0)))
        image_compare.capture_and_compare( self.view, "camera_setView.png")

class TestCameraProjections(Common, unittest.TestCase):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)

    def test_set_get_projectionPerspective(self):
        # Pre-computed value to adjust the scene content to the view limits
        fov = 23.13
        self.view.camera.setProjectionPerspective(fov)
        image_compare.capture_and_compare(
            self.view,'camera_projectionPerspective.png', threshold=2.0)

    def test_set_get_projectionFrustum(self):
        left = bottom = -0.5
        right = top = 0.5
        near = 1
        self.view.camera.setProjectionFrustum(left, right, bottom, top, near)
        assert(self.view.camera.getProjectionFrustum() ==
               (left, right, bottom, top, near))
        image_compare.capture_and_compare(
            self.view,'camera_projectionFrustum.png', threshold=2.0)

    def test_set_get_projectionOrtho(self):
        left = bottom = -0.5
        right = top = 0.5
        self.view.camera.setProjectionOrtho(left, right, bottom, top, 1)
        assert(self.view.camera.getProjectionOrtho() ==
               (left, right, bottom, top))
        image_compare.capture_and_compare(
            self.view,'camera_projectionOrtho.png', threshold=2.0)

    def test_make_ortho(self):
        assert(not self.view.camera.isOrtho())
        self.view.camera.makeOrtho()
        assert(self.view.camera.isOrtho())

    # Test switching between perspective and orthographic projection
    def test_projectionSwitch(self):
        left = bottom = -0.5
        right = top = 0.5
        self.view.camera.setProjectionOrtho(left, right, bottom, top, 1)
        image_compare.capture_and_compare(
            self.view,'camera_projectionOrtho.png', threshold=2.0)

        self.view.camera.makePerspective()
        image_compare.capture_and_compare(
            self.view,'camera_projectionFrustum.png', threshold=2.0)

        self.view.camera.makeOrtho()
        image_compare.capture_and_compare(
            self.view,'camera_projectionOrtho.png', threshold=2.0)


class TestPointProjections(unittest.TestCase):

    def setUp(self):
        self.engine = rtneuron.RTNeuron()
        self.engine.init(setup.small_eq_config)
        self.engine.resume()
        self.view = self.engine.views[0]
        self.camera = self.view.camera

    def tearDown(self):
        del self.engine

    def test_coincident(self):
        camera = self.camera
        camera.setView([1.0, 2.0, 3.0], ([0.0, 0.0, 1.0], 0.0))
        assert(camera.projectPoint(camera.getView()[0]) == (0.0, 0.0, 0.0))

    def test_behind(self):
        camera = self.camera
        camera.setView([1.0, 2.0, 3.0], ([0.0, 0.0, 1.0], 0.0))
        assert(camera.projectPoint([1.0, 2.0, 4.0]) == (0.0, 0.0, -1.0))

    def test_perspective_unrotated(self):
        camera = self.camera
        eye = array([123.0, 456.0, 789.0])
        camera.setView(eye, ([0.0, 0.0, 1.0], 0.0))
        camera.setProjectionFrustum(-0.4, 0.7, -0.3, 0.9, 1)
        frustum = camera.getProjectionFrustum()

        # Testing bottom left corner
        p = camera.projectPoint(
            eye + array([frustum[0], frustum[2], -1.0]) * 100)
        self.assertAlmostEqual(p[0], -1, places=6)
        self.assertAlmostEqual(p[1], -1, places=6)
        assert(p[2] == 1.0)
        # Testing top right side
        p = camera.projectPoint(
            eye + array([frustum[1], frustum[3], -1.0]) * 100)
        self.assertAlmostEqual(p[0], 1, places=6)
        self.assertAlmostEqual(p[1], 1, places=6)
        assert(p[2] == 1.0)

    def test_orthographics_unrotated(self):
        camera = self.camera
        eye = array([123.0, 456.0, 789.0])
        scale = 100
        camera.setView(eye, ([0.0, 0.0, 1.0], 0.0))
        camera.setProjectionOrtho(-0.4, 0.7, -0.3, 0.9, 1)
        self.view.attributes.model_scale = scale
        frustum = camera.getProjectionOrtho()

        # Testing bottom left corner
        p = camera.projectPoint(
            eye + array([frustum[0], frustum[2], -1]) * scale)
        self.assertAlmostEqual(p[0], -1, places=6)
        self.assertAlmostEqual(p[1], -1, places=6)
        assert(p[2] == 1.0)
        # Testing right side
        p = camera.projectPoint(
            eye + array([frustum[1], frustum[3], -1]) * scale)
        self.assertAlmostEqual(p[0], 1, places=6)
        self.assertAlmostEqual(p[1], 1, places=6)
        assert(p[2] == 1.0)

    def test_perspective_rotated(self):
        camera = self.view.camera
        frustum = array(camera.getProjectionOrtho())
        self.view.camera.setView([0, 0, 0], ([-1, 0, 1], 45))

        x = array([0.85355339, 0.5, -0.14644661])
        y = array([-0.5, 0.70710678, -0.5])
        z = array([-0.14644661, 0.5, 0.85355339])

        p = camera.projectPoint(x * frustum[0] + y * frustum[2] -z)
        self.assertAlmostEqual(p[0], -1, places=6)
        self.assertAlmostEqual(p[1], -1, places=6)
        assert(p[2] == 1.0)

        p = camera.projectPoint(x * frustum[1] + y * frustum[3] -z)
        self.assertAlmostEqual(p[0], 1, places=6)
        self.assertAlmostEqual(p[1], 1, places=6)
        assert(p[2] == 1.0)

    def test_othographic_rotated(self):
        camera = self.view.camera
        frustum = array(camera.getProjectionOrtho())
        self.view.camera.setView([0, 0, 0], ([-1, 0, 1], 45))

        x = array([0.85355339, 0.5, -0.14644661])
        y = array([-0.5, 0.70710678, -0.5])
        z = array([-0.14644661, 0.5, 0.85355339])

        p = camera.projectPoint(x * frustum[0] + y * frustum[2] -z)
        self.assertAlmostEqual(p[0], -1, places=6)
        self.assertAlmostEqual(p[1], -1, places=6)
        assert(p[2] == 1.0)

        p = camera.projectPoint(x * frustum[1] + y * frustum[3] -z)
        self.assertAlmostEqual(p[0], 1, places=6)
        self.assertAlmostEqual(p[1], 1, places=6)
        assert(p[2] == 1.0)

class TestPointUnprojections(unittest.TestCase):
    def setUp(self):
        self.engine = rtneuron.RTNeuron()
        self.engine.init(setup.small_eq_config)
        self.engine.resume()
        self.view = self.engine.views[0]
        self.camera = self.view.camera

    def tearDown(self):
        del self.engine

    def test_coincident(self):
        camera = self.camera
        p = [1.0, 2.0, 3.0]
        camera.setView(p, ([0.0, 0.0, 1.0], 0.0))
        assert(camera.unprojectPoint([0, 0], 0) == tuple(p))

    def test_centered(self):
        camera = self.camera
        camera.setView([1.0, 2.0, 3.0], ([0.0, 0.0, 1.0], 0.0))
        p = camera.unprojectPoint([0, 0], -1)
        self.assertAlmostEqual(p[0], 1.0)
        self.assertAlmostEqual(p[1], 2.0)
        # z is displaced
        self.assertAlmostEqual(p[2], 2.0)

    def test_perspective_unrotated(self):
        camera = self.camera
        eye = array([123.0, 456.0, 789.0])
        camera.setView(eye, ([0.0, 0.0, 1.0], 0.0))
        camera.setProjectionFrustum(-0.4, 0.7, -0.3, 0.9, 1)
        frustum = camera.getProjectionFrustum()

        # Testing bottom left corner
        p = camera.unprojectPoint([-1, -1], -10)
        self.assertAlmostEqual(p[0], 119)
        self.assertAlmostEqual(p[1], 453)
        self.assertAlmostEqual(p[2], 779)

        # Testing top right corner
        p = camera.unprojectPoint([1, 1], -10)
        self.assertAlmostEqual(p[0], 130)
        self.assertAlmostEqual(p[1], 465)
        self.assertAlmostEqual(p[2], 779)

    def test_perspective_rotated(self):
        camera = self.camera
        eye = array([123.0, 456.0, 789.0])
        camera.setView(eye, ([0.0, 1.0, 0.0], 90.0))
        camera.setProjectionFrustum(-0.4, 0.7, -0.3, 0.9, 1)
        frustum = camera.getProjectionFrustum()

        # Testing bottom left corner
        p = camera.unprojectPoint([-1, -1], -10)
        self.assertAlmostEqual(p[0], 113)
        self.assertAlmostEqual(p[1], 453)
        self.assertAlmostEqual(p[2], 793)

        # Testing top right corner
        p = camera.unprojectPoint([1, 1], -10)
        self.assertAlmostEqual(p[0], 113)
        self.assertAlmostEqual(p[1], 465)
        self.assertAlmostEqual(p[2], 782)

    def test_orthographic_unrotated(self):
        camera = self.camera
        eye = array([123.0, 456.0, 789.0])
        camera.setView(eye, ([0.0, 0.0, 1.0], 0.0))
        camera.setProjectionOrtho(-4, 7, -3, 9, 1)
        self.view.attributes.model_scale = 1

        # Testing bottom left corner
        p = camera.unprojectPoint([-1, -1], -10)
        self.assertAlmostEqual(p[0], 119)
        self.assertAlmostEqual(p[1], 453)
        self.assertAlmostEqual(p[2], 779)
        p = camera.unprojectPoint([-1, -1], -1)
        self.assertAlmostEqual(p[0], 119)
        self.assertAlmostEqual(p[1], 453)
        self.assertAlmostEqual(p[2], 788)

        # Testing top right corner
        p = camera.unprojectPoint([1, 1], -10)
        self.assertAlmostEqual(p[0], 130)
        self.assertAlmostEqual(p[1], 465)
        self.assertAlmostEqual(p[2], 779)
        p = camera.unprojectPoint([1, 1], -1)
        self.assertAlmostEqual(p[0], 130)
        self.assertAlmostEqual(p[1], 465)
        self.assertAlmostEqual(p[2], 788)

    def test_orthographic_rotated(self):
        camera = self.camera
        eye = array([123.0, 456.0, 789.0])
        camera.setView(eye, ([0.0, 1.0, 0.0], 90.0))
        camera.setProjectionOrtho(-4, 7, -3, 9, 1)
        self.view.attributes.model_scale = 1

        # Testing bottom left corner
        p = camera.unprojectPoint([-1, -1], -10)
        self.assertAlmostEqual(p[0], 113)
        self.assertAlmostEqual(p[1], 453)
        self.assertAlmostEqual(p[2], 793)
        p = camera.unprojectPoint([-1, -1], -1)
        self.assertAlmostEqual(p[0], 122)
        self.assertAlmostEqual(p[1], 453)
        self.assertAlmostEqual(p[2], 793)

        # Testing top right corner
        p = camera.unprojectPoint([1, 1], -10)
        self.assertAlmostEqual(p[0], 113)
        self.assertAlmostEqual(p[1], 465)
        self.assertAlmostEqual(p[2], 782)
        p = camera.unprojectPoint([1, 1], -1)
        self.assertAlmostEqual(p[0], 122)
        self.assertAlmostEqual(p[1], 465)
        self.assertAlmostEqual(p[2], 782)

if __name__ == '__main__':
    unittest.main()
