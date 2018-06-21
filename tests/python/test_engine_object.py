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
import sys
import rtneuron

import unittest

class TestRTNeuronObject(unittest.TestCase):
    """This test case tried tests the basics of starting RTNeuron and a basic
    single node equalizer configuration."""

    def test_creation_without_args(self) :
        engine = rtneuron.RTNeuron()

    def test_creation_with_args(self) :
        args = ['--eq-flags', '']
        engine = rtneuron.RTNeuron(args)

    def test_creation_with_attributes(self) :
        attributes = rtneuron.AttributeMap({'soma_radius' : 10})
        engine = rtneuron.RTNeuron(attributes = attributes)

    def test_creation_with_args_and_attributes(self) :
        attributes = rtneuron.AttributeMap({'soma_radius' : 10})
        engine = rtneuron.RTNeuron(sys.argv, attributes)

    def test_configuration_instantiation_implicit_exit(self) :
        engine = rtneuron.RTNeuron(sys.argv)
        engine.init()
        del engine

    def test_configuration_instantiation_explicit_exit(self) :
        engine = rtneuron.RTNeuron()
        engine.init()
        engine.exit()
        del engine

    def test_configuration_instantiation_with_config(self) :
        engine = rtneuron.RTNeuron()
        engine.init(setup.medium_rect_eq_config)
        del engine

    def test_configuration_reinstantiation(self) :
        # Trying several instantiations one after the other. This test doesn't
        # only test the instantiation, but also reinstantiation after cleanup

        # With implicit exit
        engine = rtneuron.RTNeuron(sys.argv)
        engine.init()

        # Without deleting the object
        # This throws because the new instance is created before the old
        # one is destroyed.
        self.assertRaises(RuntimeError, rtneuron.RTNeuron)

        del engine

        engine = rtneuron.RTNeuron(sys.argv)
        engine.init()
        del engine

if __name__ == '__main__':
    unittest.main()
