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

import os

tile_size = (256, 256)
window_size = (600, 600)

# Set a very small framebuffer size to enforce tiling
os.environ['RTNEURON_MAX_FB_WIDTH'] = str(tile_size[0])
os.environ['RTNEURON_MAX_FB_HEIGHT'] = str(tile_size[1])

os.environ['EQ_WINDOW_IATTR_HINT_WIDTH'] = str(window_size[0])
os.environ['EQ_WINDOW_IATTR_HINT_HEIGHT'] = str(window_size[1])

import setup
import image_compare
import unittest

import brain
import rtneuron

import sys

os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12' # FBO
del os.environ['EQ_WINDOW_IATTR_PLANES_SAMPLES']    # No hw AA

class TestTiledSnapshot(unittest.TestCase):
    def setUp(self):
        rtneuron.display_circuit(brain.test.blue_config,
            ('L5CSPC', {'mode' : rtneuron.RepresentationMode.SOMA}))

        self.view = rtneuron.engine.views[0]
        self.view.attributes.background = [0, 0, 0, 1]
        # So results are not GPU/drivers dependant
        self.view.attributes.idle_AA_steps = 32
        # To make sure the scene is ready
        rtneuron.engine.frame()
        rtneuron.engine.waitFrame()

    def tearDown(self):
        rtneuron.engine = None
        del self.view

    # Tiled snapshot
    def test_tiled_snapshot(self):
        ref_image = "snapshot_no_tiling.png"
        tile_prefix = "snapshot_tiling"
        self.view.snapshot(ref_image)
        self.view.snapshot(tile_prefix + ".png", window_size)
        image_compare.compare_tiled_image(ref_image, tile_prefix,
                                          window_size, tile_size)


if __name__ == '__main__':
    unittest.main()
