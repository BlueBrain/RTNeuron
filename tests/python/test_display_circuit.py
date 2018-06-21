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
import rtneuron

import brain
import unittest

config = brain.test.blue_config

class TestDisplayCircuit(unittest.TestCase):

    def tearDown(self):
        del rtneuron.engine

    def test_display_empty_scene(self):
        rtneuron.display_empty_scene()

    def test_display_circuit(self):

        rtneuron.display_circuit(
            config, ('MiniColumn_0',
                      {'mode': rtneuron.RepresentationMode.SOMA}))

    def test_display_circuit_with_regex(self):

        rtneuron.display_circuit(
            config, ('MiniColumn_[0-9]',
                      {'mode': rtneuron.RepresentationMode.SOMA}))

class TestDisplayCircuitImplicitDestroy(unittest.TestCase):

    def test_display_circuit(self):

        rtneuron.display_circuit(
            config, ('MiniColumn_0',
                     {'mode': rtneuron.RepresentationMode.SOMA}))

if __name__ == '__main__':
    unittest.main()

