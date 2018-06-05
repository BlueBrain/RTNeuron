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
from rtneuron import *
import rtneuron as _rtneuron
import brain

import unittest

class Common:

    def setUp(self):
        attributes = AttributeMap({'mode' : RepresentationMode.SOMA})
        self._cells = [310, 320, 330, 340, 350]
        scene_attributes = AttributeMap({"load_morphologies": False})
        setup.setup_simple(self, self._cells,
                           target_attributes=attributes,
                           scene_attributes=scene_attributes,
                           # Rectangular selections in orthographics
                           # projections don't work if a configuration file
                           # is used.
                           eq_config=None,
                           blue_config=brain.test.circuit_config,
                           engine_attributes=_rtneuron.global_attributes)
        _rtneuron.engine = self.engine
        _rtneuron.simulation = self.simulation

    def tearDown(self):
        _rtneuron.engine = None
        _rtneuron.simulation = None
        self.engine.exit()
        del self.engine

class TestRayPicking(unittest.TestCase, Common):

    def setUp(self):
        Common.setUp(self)

        self._cell = None
        self._synapse = None

        def cell_selected(gid, section, segment):
            self._cell = gid

        def synapse_selected(synapse):
            self._synapse = synapse.gid()

        self.scene.cellSelected.connect(cell_selected)
        self.scene.synapseSelected.connect(synapse_selected)

    def tearDown(self):
        Common.tearDown(self)

    def test_cell_pick(self):

        origin = [41, 1273, 215]
        self.view.camera.setView(origin, ([0.0, 0.0, 1.0], 0.0))

        self.scene.pick(origin, [-0.11089, 0.016633, -0.998])
        assert(self._cell == 340)

        self._cell = None
        self.scene.pick(origin, [-0.11089, 0.08, -0.998])
        assert(self._cell == None)

        self.scene.pick(origin, [-0.17188, 0.43986, -0.998])
        assert(self._cell == 330)

        origin = [34, 1299, -103]
        self.view.camera.setView(origin,
                                 ([0.688269, 0.7241339, 0.043769221], 188))

        self._cell = None
        self.scene.pick(origin, [-0.0389556, -0.604839, 0.91692])
        assert(self._cell == None)

        self.scene.pick(origin, [-0.0327834, -0.603274, 0.916916])
        assert(self._cell == 350)

    def test_synapse_pick(self):
        display_synapses(350)
        origin = [41, 1273, 215]
        self.view.camera.setView(origin, ([0.0, 0.0, 1.0], 0.0))

        self.scene.pick(origin, [0.69112, 0.36801, -0.998])
        assert(self._synapse == (350, 69))

        self.scene.pick(origin, [-0.11103, -0.25324, -0.998])
        assert(self._synapse == (350, 20))

        self._synapse = None
        self.scene.pick(origin, [-0.10167, -0.25181, -0.99811])
        assert(self._synapse == None)

        origin = [34, 1299, -103]
        self.view.camera.setView(origin,
                                 ([0.688269, 0.7241339, 0.043769221], 188))

        self.scene.pick(origin, [0.40677, -0.42381, 0.92662])
        assert(self._synapse == (350, 50))

        self._synapse = None
        self.scene.pick(origin, [0.40914, -0.42121, 0.92694])
        assert(self._synapse == None)

        self.scene.objects[1].attributes.radius = 5
        self.scene.objects[1].update()
        self.view.attributes.auto_compute_home_position = False
        self.scene.pick(origin, [0.40914, -0.42121, 0.92694])
        assert(self._synapse == (350, 50))

class TestCameraAreaPicking(unittest.TestCase, Common):
    # Several checks are performed in each test to minimize the total
    # execution time.
    def setUp(self):
        Common.setUp(self)

        self._cells = None

        def setSelected(cells):
            self._cells = list(cells)

        # Forcing the model scale to be the one that was badly computed before
        # the fix in 57c463b
        self.view.attributes.model_scale = 139.32611083984375
        self.scene.cellSetSelected.connect(setSelected)

    def tearDown(self):
        Common.tearDown(self)

    def test_pick_ortho_no_rotation(self):
        self.view.camera.makeOrtho()
        self.check_unrotated()

    def test_pick_ortho_rotated(self):

        self.view.camera.makeOrtho()
        self.view.camera.setView([283, 1072, 625],
                                 ([0.4441163, 0.1833558, 0.8770070], 63))
        self.check_rotated()

    def test_pick_perspective_no_rotated(self):
        self.view.camera.setView([41, 1273, 233], ([0.0, 0.0, 1.0], 0.0))
        self.check_unrotated()

    def test_pick_perspective_rotated(self):
        self.view.camera.setView([43, 1243, 157],
                                  ([0.2495182, 0.026356816, 0.96801131], 62))
        self.check_rotated()

    def check_unrotated(self):
        self.scene.pick(self.view, 0, 1, 0, 1)
        assert(self._cells == [310, 320, 330, 340, 350])

        self.scene.pick(self.view, 0.52, 1, 0, 1) # Checking left side
        assert(self._cells == [320])

        self.scene.pick(self.view, 0, 0.52, 0, 1) # Checking right side
        assert(self._cells == [310, 330, 340, 350])

        self.scene.pick(self.view, 0, 1, 0.75, 1) # Checking bottom side
        assert(self._cells == [310, 330])

        self.scene.pick(self.view, 0, 1, 0, 0.75) # Checking top side
        assert(self._cells == [320, 340, 350])

        self.scene.pick(self.view, 0.4, 0.6, 0.4, 0.6) # All sides
        assert(self._cells == [320, 340])

        self._cells = None
        self.scene.pick(self.view, 0.85, 0.90, 0.85, 0.90) # Empty
        assert(self._cells == None)

    def check_rotated(self):
        self.scene.pick(self.view, 0, 1, 0, 1)
        assert(self._cells == [310, 320, 330, 340, 350])

        self.scene.pick(self.view, 0.3, 1, 0, 1) # Checking left side
        assert(self._cells == [310, 320, 330, 340])

        self.scene.pick(self.view, 0, 0.3, 0, 1) # Checking right side
        assert(self._cells == [350])

        self.scene.pick(self.view, 0, 1, 0.4, 1) # Checking bottom side
        assert(self._cells == [310, 330, 340])

        self.scene.pick(self.view, 0, 1, 0, 0.4) # Checking top side
        assert(self._cells == [320, 350])

        self.scene.pick(self.view, 0.4, 0.6, 0.4, 0.6) # All sides
        assert(self._cells == [340])

        self._cells = None
        self.scene.pick(self.view, 0.05, 0.10, 0.85, 0.90) # Empty
        assert(self._cells == None)

class TestMasking(unittest.TestCase, Common):

    def setUp(self):

        Common.setUp(self)

        self._cells = None
        self._cell = None

        def cellSelected(gid, section, segment):
            self._cell = gid

        def setSelected(cells):
            self._cells = [cell for cell in cells]

        self.scene.cellSetSelected.connect(setSelected)
        self.scene.cellSelected.connect(cellSelected)

    def tearDown(self):
        Common.tearDown(self)

    def test_masked_camera_area_pick(self):
        self.scene.pick(self.view, 0, 1, 0, 1)
        assert(self._cells == [310, 320, 330, 340, 350])

        self.scene.neuronSelectionMask = [310]
        self.scene.pick(self.view, 0, 1, 0, 1)
        assert(self._cells == [320, 330, 340, 350])

        self._cells = None
        self.scene.neuronSelectionMask = [310, 320, 330, 340, 350]
        self.scene.pick(self.view, 0, 1, 0, 1)
        assert(self._cells == None)

        self.scene.neuronSelectionMask = []
        self.scene.pick(self.view, 0, 1, 0, 1)
        assert(self._cells == [310, 320, 330, 340, 350])

    def test_masked_ray_pick(self):
        self.view.camera.setView([41, 1273, 215], ([0.0, 0.0, 1.0], 0.0))

        self.scene.pick([41, 1273, 215], [-0.11089, 0.016633, -0.998])
        assert(self._cell == 340)

        self.scene.neuronSelectionMask = [340]
        self._cell = None
        self.scene.pick([41, 1273, 215], [-0.11089, 0.016633, -0.998])
        assert(self._cell == None)

        self.scene.neuronSelectionMask = []
        self._cell = None
        self.scene.pick([41, 1273, 215], [-0.11089, 0.016633, -0.998])
        assert(self._cell == 340)


if __name__ == '__main__':
    unittest.main()
