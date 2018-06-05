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

import unittest

class TestAttributeMap(unittest.TestCase):

    def test_attributes(self):
        attributes = rtneuron.AttributeMap()
        attributes.foo = False
        assert(attributes.foo == False)
        attributes.foo = 10
        assert(attributes.foo == 10)
        attributes.foo = 10.0
        assert(attributes.foo == 10.0)
        attributes.foo = 'string'
        assert(attributes.foo == 'string')
        attributes.blah = [0, 1, 2]
        assert(attributes.blah == [0, 1, 2])
        attributes == rtneuron.AttributeMap({'foo': 'string',
                                             'blah': [0, 1, 2]})
        assert (attributes.__str__() == "blah: 0 1 2\nfoo: string\n")
        assert(dir(attributes) == ['blah', 'foo'])
        attributes.help()

    def test_color_map(self):
        attributes = rtneuron.AttributeMap()
        attributes.colormap = rtneuron.ColorMap()
        points = {0: (0.0, 0.0, 0.0, 0.0), 1: (1.0, 1.0, 1.0, 1.0)}
        attributes.colormap.setPoints(points)
        assert(attributes.colormap.getPoints() == points)

    def test_attribute_changed_signal(self):
        attributes = rtneuron.AttributeMap()

        def callback(name):
            self._name = name
            self._value = attributes.__getattr__(name)

        attributes.attributeChanged.connect(callback)
        attributes.test = 10
        assert(self._name == "test")
        assert(self._value == 10)

if __name__ == '__main__':
    unittest.main()

