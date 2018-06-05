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
from rtneuron.sceneops import NeuronClipping

import brain
import os

import unittest

class Common:
    def setUp(self, sceneAttr = AttributeMap(), neuronAttr = AttributeMap(),
              make_ready=True):

        setup.setup_simple(self, 406, sceneAttr, neuronAttr)

        circuit = self.scene.circuit
        self._morphology = \
            circuit.load_morphologies([406], circuit.Coordinates.local)[0]
        self._handler = self.scene.objects[0]

        view = self.view
        # Disabling idle AA to speed up testing.
        view.attributes.idle_AA_steps = 0
        view.camera.setView(
            [272.80291748046875, 764.23828125, 393.82244873046875],
            ([-0.08497956395149231, 0.9948055148124695, 0.05603933706879616],
             38.1639518737793))
        view.attributes.background = [0, 0, 0, 1]
        view.attributes.auto_compute_home_position = False

        if make_ready:
            # Displaying the first frame to ensure that clipping operations
            # have the intended effect
            for i in range(2):
                self.engine.frame()
                self.engine.waitFrame()

    def tearDown(self):
        self.engine.exit()
        del self.engine

def create_low_lod_attributes():
    attributes = AttributeMap()
    attributes.lod = AttributeMap()
    attributes.lod.neurons = AttributeMap({'low_detail_cylinders' : [0.0, 1.0],
                                           'spherical_soma' : [0.0, 1.0]})
    return attributes

def create_medium_lod_attributes():
    attributes = AttributeMap()
    attributes.lod = AttributeMap()
    attributes.lod.neurons = AttributeMap({'high_detail_cylinders' : [0.0, 1.0],
                                           'detailed_soma' : [0.0, 1.0]})
    return attributes

def create_high_lod_attributes():
    attributes = AttributeMap()
    attributes.lod = AttributeMap()
    attributes.lod.neurons = AttributeMap({'mesh' : [0.0, 1.0]})
    return attributes

class ClipErrors(unittest.TestCase):
    def test_badlengths(self):
        clipping = NeuronClipping()
        self.assertRaises(ValueError, NeuronClipping.clip, clipping,
                          [0], [0], [0, 1])
        self.assertRaises(ValueError, NeuronClipping.clip, clipping,
                          [0], [0, 1], [0])
        self.assertRaises(ValueError, NeuronClipping.clip, clipping,
                          [0, 1], [0], [0])
        self.assertRaises(ValueError, NeuronClipping.unclip, clipping,
                          [0], [0], [0, 1])
        self.assertRaises(ValueError, NeuronClipping.unclip, clipping,
                          [0], [0, 1], [0])
        self.assertRaises(ValueError, NeuronClipping.unclip, clipping,
                          [0, 1], [0], [0])

class SimpleClipUnclip(Common):

    def test_clip(self):
        self._handler.apply(NeuronClipping().clip(
                range(500), [0.0] * 500, [0.5] * 500))


        # Displaying the first frame to ensure that clipping operations
        # have the intended effect
        for i in range(2):
            self.engine.frame()
            self.engine.waitFrame()

        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_clip_axon(self):
        sections = [int(x) for x in self._morphology.section_ids(
            [brain.neuron.SectionType.axon])]
        length = len(sections)
        self._handler.apply(NeuronClipping().clip(
                sections, [0] * length, [1] * length))
        tmp = image_compare.capture_temporary(self.view)

        self._handler.attributes.mode = RepresentationMode.NO_AXON
        self._handler.update()
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

    def test_unclip(self):
        self._handler.apply(NeuronClipping().clipAll(True))
        self._handler.apply(NeuronClipping().unclip(
                range(1, 501), [0.5] * 500, [1.0] * 500))
        # This result must match the result from test_clip
        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_clip_all(self):
        self._handler.apply(NeuronClipping().clipAll(True))
        tmp = image_compare.capture_temporary(self.view)

        self._handler.attributes.mode = RepresentationMode.NO_DISPLAY
        self._handler.update()
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

    def test_unclip_all(self):
        tmp = image_compare.capture_temporary(self.view)

        self._handler.apply(NeuronClipping().clipAll(True))
        self._handler.apply(NeuronClipping().unclipAll())
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

class TestLowLODSimpleClipUnclip(unittest.TestCase, SimpleClipUnclip):
    def setUp(self):
        self._prefix = "neuron_clip_low_lod_"
        Common.setUp(self, create_low_lod_attributes())

    def tearDown(self):
        Common.tearDown(self)

    def test_clip_all_but_soma(self):
        self._handler.apply(NeuronClipping().clipAll())
        tmp = image_compare.capture_temporary(self.view)

        self._handler.attributes.mode = RepresentationMode.SOMA
        self._handler.update()
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

    def test_clip_soma_interval(self):
        # Soma is clipping with any range
        self._handler.apply(NeuronClipping().clipAll())
        self._handler.apply(NeuronClipping().clip([0], [0], [0.1]))
        tmp = image_compare.capture_temporary(self.view)

        self._handler.attributes.mode = RepresentationMode.NO_DISPLAY
        self._handler.update()
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

class TestMediumLODSimpleClipUnclip(unittest.TestCase, SimpleClipUnclip):
    def setUp(self):
        self._prefix = "neuron_clip_medium_lod_"
        Common.setUp(self, create_medium_lod_attributes())

    def tearDown(self):
        Common.tearDown(self)

class TestHighLODSimpleClipUnclip(unittest.TestCase, SimpleClipUnclip):
    def setUp(self):
        self._prefix = "neuron_clip_high_lod_"
        Common.setUp(self, create_high_lod_attributes())

    def tearDown(self):
        Common.tearDown(self)

class TestClippingRestrictions(unittest.TestCase, Common):
    def setUp(self):
        self._prefix = "neuron_clip_medium_lod_"
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.NO_AXON
        Common.setUp(self, neuronAttr = attributes)

    def tearDown(self):
        Common.tearDown(self)

    def test_soma_mode(self):
        self._handler.attributes.mode = RepresentationMode.SOMA
        self._handler.update()
        self.engine.frame()
        tmp = image_compare.capture_temporary(self.view)

        self._handler.apply(NeuronClipping().clipAll(False))
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

        self._handler.attributes.mode = RepresentationMode.NO_DISPLAY
        self._handler.update()
        self.engine.frame()
        tmp = image_compare.capture_temporary(self.view)

        self._handler.apply(NeuronClipping().clipAll(True))
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)


    def test_axon_mode_protected(self):
        tmp = image_compare.capture_temporary(self.view)

        self._handler.apply(NeuronClipping().unclipAll())
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

class TestCombinedClipUnclip(unittest.TestCase, Common):
    def setUp(self):
        self._prefix = "neuron_clip_high_lod_"
        Common.setUp(self, create_high_lod_attributes())

    def tearDown(self):
        Common.tearDown(self)

    def test_multiple_clip_1(self):
        clipping = NeuronClipping()
        for i in range(500):
            clipping.clip([i], [0.0], [0.5])
        self._handler.apply(clipping)
        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_multiple_clip_2(self):
        for i in range(500):
            self._handler.apply(NeuronClipping().clip([i], [0.0], [0.5]))
        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_multiple_unclip_1(self):
        self._handler.apply(NeuronClipping().clipAll(True))
        clipping = NeuronClipping()
        for i in range(1, 501):
            clipping.unclip([i], [0.5], [1.0])
        self._handler.apply(clipping)
        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_multiple_unclip_2(self):
        self._handler.apply(NeuronClipping().clipAll(True))
        clipping = NeuronClipping()
        for i in range(1, 501):
            self._handler.apply(NeuronClipping().unclip([i], [0.5], [1.0]))
        self._handler.apply(clipping)
        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_clip_chain_1(self):
        tmp = image_compare.capture_temporary(self.view)
        self._handler.apply(
            NeuronClipping().clipAll()
                            .unclip(range(500), [0] * 500, [1.0] * 500))
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

    def test_clip_chain_2(self):
        self._handler.apply(
            NeuronClipping().clip(range(500), [0] * 500, [0.75] * 500)
                            .unclip(range(1, 501), [0.5] * 500, [0.75] * 500))
        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")

    def test_clip_and_unclip_are_perfect_opposites(self):
        from random import random
        starts = [random() for i in range(500)]
        ends = [s + random() * (1 - s) for s in starts]

        tmp = image_compare.capture_temporary(self.view)
        self._handler.apply(NeuronClipping().clip(range(500), starts, ends))
        self._handler.apply(NeuronClipping().unclip(range(500), starts, ends))
        try:
            image_compare.capture_and_compare(
                self.view, tmp, prepend_sample_path = False)
        finally:
            os.remove(tmp)

class TestClipBeforeSceneReady(unittest.TestCase, Common):

    def setUp(self):
        self._prefix = "neuron_clip_high_lod_"
        Common.setUp(self, create_high_lod_attributes(), make_ready=False)

    def test_clip(self):
        self._handler.apply(NeuronClipping().clip(
                range(500), [0.0] * 500, [0.5] * 500))

        # Rendering at least one frame to make sure the scene is not empty for
        # the snapshot.
        self.engine.frame()
        self.engine.waitFrame()

        image_compare.capture_and_compare(
            self.view, self._prefix + "simple_clip.png")


if __name__ == '__main__':
    unittest.main()
