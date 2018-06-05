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
import tempfile
import threading

import brain

from rtneuron import *

import unittest

class Common:
    def setUp(self, target_attributes=AttributeMap(),
              scene_attributes=AttributeMap(), target='L5CSPC'):
        setup.setup_simple(self, target, target_attributes=target_attributes,
                           scene_attributes=scene_attributes)

        view = self.view
        view.attributes.idle_AA_steps = 32
        view.attributes.background = [0, 0, 0, 1]
        view.auto_compute_home_position = False
        view.scene = self.scene
        # To make sure the scene is ready before the timestamps are modified
        self.engine.frame()
        self.engine.frame()

    def tearDown(self):
        del self.engine

    def switch_mode(self, mode):
        neurons = self.view.scene.objects[0]
        neurons.attributes.mode = mode
        neurons.update()

    def create_spikes_file(self):
        # Generating a spike data file (the file from TestData is too sparse)
        spikefile = tempfile.NamedTemporaryFile(suffix='.dat')
        spikes = open(spikefile.name, 'w')
        time = 0
        positions = self.simulation.open_circuit().positions(self.gids)

        x_gid = [(p[0], g) for p, g in zip(positions, self.gids)]
        x_gid.sort()
        for dummy, gid in x_gid:
            spikes.write('1 %d\n' % gid)
            spikes.write('%f %d\n' % (time + 2, gid))
            spikes.write('%f %d\n' % (time + 4, gid))
            time += 2.0 / len(self.gids)
        spikes.close()
        return spikefile

class TestBadTimestamp(unittest.TestCase):
    def setUp(self):
        self.engine = RTNeuron([])
        self.simulation = brain.Simulation(brain.test.blue_config)
        scene = self.engine.createScene()
        self.engine.init(setup.medium_rect_eq_config)
        self.view = self.engine.views[0]
        self.view = scene

    def tearDown(self):
        del self.engine

    def test_bad_set_timestamp(self):
        player = self.engine.player
        player.window = [10, 20]

        player.timestamp = player.window[0]
        self.assertRaises(
            RuntimeError, setattr, player, "timestamp", player.window[0] - 1)

        player.timestamp = player.window[1]
        self.assertRaises(
            RuntimeError, setattr, player, "timestamp", player.window[1] + 1)

class TestSimulationWindowAdjustments(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)
        self._spikefile = self.create_spikes_file()
        apply_spike_data(self._spikefile.name, self.view)

    def tearDown(self):
        Common.tearDown(self)
        self._spikefile.close()

    def test_no_auto_adjust(self):
        self.engine.attributes.auto_adjust_simulation_window = False
        player = self.engine.player
        self.assertRaises(
            RuntimeError, setattr, player, "timestamp", 1 )
        player.play()
        # The window stays with the default nan, nan
        assert(player.window != player.window)

    def test_manual_update(self):
        player = self.engine.player
        player.adjustWindow()
        assert(player.window == (0.0, 5.942856788635254))

    def test_set_window_turns_off_auto_adjust(self):
        player = self.engine.player
        # This window is choosen to be contained within the spike report window
        manual_window = (2, 3)
        player.window = manual_window
        assert(not self.engine.attributes.auto_adjust_simulation_window)

        player.timestamp = 2.5 # This would trigger autoadjust, but mustn't
        assert(player.window == manual_window)

        self.engine.attributes.auto_adjust_simulation_window = True
        # Trying again
        player.timestamp = 1.0 # Deliberately outside the manual window but
                               # but inside the report window.
        assert(player.window == (0.0, 5.942856788635254))

    def test_auto_adjust_with_set_timestamp(self):
        player = self.engine.player
        player.timestamp = 1.0
        assert(player.window == (0.0, 5.942856788635254))

    def test_auto_adjust_with_play(self):
        player = self.engine.player
        player.play()
        assert(player.window == (0.0, 5.942856788635254))

    def test_merge_windows(self):
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        player = self.engine.player
        player.adjustWindow()
        assert(player.window == (0.0, 10.0))

    def test_adjust_with_no_report_fails(self):
        self.view.scene.clear()
        player = self.engine.player
        self.assertRaises(RuntimeError, SimulationPlayer.adjustWindow, player)

class TestCompartmentReports(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)

    def tearDown(self):
        Common.tearDown(self)
        self._spikefile.close()

    def tearDown(self):
        Common.tearDown(self)

    def test_simulation_display(self):
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        self.engine.player.window = [0, 10]

        self.view.attributes.lod_bias = 1
        self.engine.player.timestamp = 0
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t0.png")

        self.engine.player.timestamp = 5
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5.png")

class TestCompartmentReportsWithLODs(unittest.TestCase, Common):
    def setUp(self):
        attributes = AttributeMap()
        attributes.lod = AttributeMap()
        attributes.lod.neurons = AttributeMap()
        attributes.lod.neurons.mesh = [0.5, 1.0]
        attributes.lod.neurons.spherical_soma = [0, 0.5]
        attributes.lod.neurons.high_detail_cylinders = [0, 0.5]
        Common.setUp(self, scene_attributes = attributes)

    def tearDown(self):
        Common.tearDown(self)

    def test_lod_update(self):
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        self.engine.player.window = [0, 10]

        self.view.attributes.lod_bias = 0
        self.engine.player.timestamp = 5
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5_low_lod.png")

        self.view.attributes.lod_bias = 1
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5.png")

        self.switch_mode(RepresentationMode.SOMA)
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5_somas.png")

class TestCompartmentReportsWithModeUpgrade(unittest.TestCase, Common):
    def setUp(self):
        attributes = AttributeMap()
        attributes.mode = RepresentationMode.SOMA
        Common.setUp(self, target_attributes=attributes)

    def tearDown(self):
        Common.tearDown(self)

    def test_soma_to_detailed(self):
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        self.engine.player.window = [0, 10]

        self.engine.player.timestamp = 5
        self.engine.frame()

        self.switch_mode(RepresentationMode.WHOLE_NEURON)
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5.png")

class TestSomaSpikeReports(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)
        self._spikefile = self.create_spikes_file()
        apply_spike_data(self._spikefile.name, self.view)
        self.engine.player.window = [0, 6]

    def tearDown(self):
        Common.tearDown(self)
        self._spikefile.close()

    def test_soma_spikes(self):
        self.switch_mode(RepresentationMode.SOMA)
        self.view.camera.setView([49, 630, 509], ([0.0, 0.0, 1.0], 0.0))

        self.engine.player.timestamp = 0
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t0_somas.png")
        self.engine.player.timestamp = 1
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t1_somas.png")
        self.engine.player.timestamp = 5
        self.view.attributes.spike_tail = 1
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t5_d1_somas.png")
        self.view.attributes.spike_tail = 2
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t5_d2_somas.png")

class TestCombinedReportsOnSoma(unittest.TestCase, Common):
    def setUp(self):
        Common.setUp(self)
        self._spikefile = self.create_spikes_file()

    def tearDown(self):
        del self.engine

    def test_compartment_report_rules1(self):
        apply_spike_data(self._spikefile.name, self.view)
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        self.engine.player.window = [0, 6]
        self.switch_mode(RepresentationMode.SOMA)
        self.engine.player.timestamp = 5
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5_somas.png")

    def test_compartment_report_rules2(self):
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        apply_spike_data(self._spikefile.name, self.view)
        self.engine.player.window = [0, 6]
        self.switch_mode(RepresentationMode.SOMA)
        self.engine.player.timestamp = 5
        image_compare.capture_and_compare(
            self.view, "compartment_simulation_t5_somas.png")

class TestSpikeReportsWithAxon(unittest.TestCase, Common):

    def setUp(self):
        Common.setUp(self, target=[396, 410])
        self._spikefile = self.create_spikes_file()

    def tearDown(self):
        Common.tearDown(self)
        self._spikefile.close()

    def test_axon_spikes(self):
        apply_spike_data(self._spikefile.name, self.view)
        self.engine.player.window = [0, 10]

        self.view.camera.setView([-187, 481, 880], ([0.0, 0.0, 1.0], 0.0))
        self.engine.player.timestamp = 4.5
        self.view.attributes.spike_tail = 1
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t4.5_d1_axons.png")
        self.view.attributes.spike_tail = 2
        image_compare.capture_and_compare(
            self.view, "spike_simulation_t4.5_d2_axons.png")

    def test_combined_reports_1(self):
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        apply_spike_data(self.simulation, self.view)
        self.engine.player.window = [0, 10]
        self.engine.player.timestamp = 9.999999
        image_compare.capture_and_compare(self.view,
                                          "combined_simulation.png")

    def test_combined_reports_2(self):
        apply_spike_data(self.simulation, self.view)
        apply_compartment_report(self.simulation, self.view,
                                 "allCompartments")
        self.engine.player.window = [0, 10]
        self.engine.player.timestamp = 9.999999
        image_compare.capture_and_compare(self.view,
                                          "combined_simulation.png")

class TestSimpleSignals(unittest.TestCase):

    def setUp(self):
        self.engine = RTNeuron()
        self.engine.init()
        self._delta = None
        self._window = None

    def tearDown(self):
        del self.engine

    def test_simulation_delta_changed(self):
        player = self.engine.player

        def callback(delta) :
            self._delta = delta
        player.simulationDeltaChanged.connect(callback)
        delta = 0.5
        player.simulationDelta = delta
        assert(self._delta == delta)

    def test_simulation_window_changed(self):
        player = self.engine.player

        def callback(begin, end) :
            self._window = (begin, end)
        player.windowChanged.connect(callback)
        window = (0.25, 0.75)
        player.window = window
        assert(self._window == window)


class TestSimulationStateSignals(unittest.TestCase, Common):

    def setUp(self):
        Common.setUp(self, target=[396])
        self._spikefile = self.create_spikes_file()
        self._state = SimulationPlayer.PAUSED

    def tearDown(self):
        Common.tearDown(self)
        self._spikefile.close()

    def test_play(self):
        apply_spike_data(self._spikefile.name, self.view)
        player = self.engine.player
        player.window = [0, 10]

        def callback(state) :
            self._state = state
        player.playbackStateChanged.connect(callback)
        player.play()
        assert(self._state == SimulationPlayer.PLAYING)

    def test_pause(self):
        apply_spike_data(self._spikefile.name, self.view)
        player = self.engine.player
        player.window = [0, 10]
        player.play()
        self._state = SimulationPlayer.PLAYING

        def callback(state) :
            self._state = state
        player.playbackStateChanged.connect(callback)
        player.pause()
        assert(self._state == SimulationPlayer.PAUSED)

    def test_finished(self):
        apply_spike_data(self._spikefile.name, self.view)
        player = self.engine.player
        player.window = [0, 10]
        player.simulationDelta = 1
        self._state = SimulationPlayer.PLAYING
        player.play()

        self._condition = threading.Condition()
        def callback(state) :
            with self._condition:
                self._state = state
                self._condition.notify()
        player.playbackStateChanged.connect(callback)

        self.engine.resume()
        with self._condition:
            if self._state != SimulationPlayer.FINISHED:
                self._condition.wait(2)
        assert(self._state == SimulationPlayer.FINISHED)

    def test_restart(self):
        apply_spike_data(self._spikefile.name, self.view)
        player = self.engine.player
        start = 0
        end = 4
        player.window = [start, end]
        player.simulationDelta = 1
        player.play()
        self._state = SimulationPlayer.PLAYING
        self._transitions = {}
        self._condition = threading.Condition()

        # Callback to annotate state transitions
        def callback(state) :
            with self._condition:
                transition = (self._state, state)
                if transition in self._transitions:
                    self._transitions[transition] += 1
                else:
                    self._transitions[transition] = 1
                self._state = state
                self._condition.notify()
        player.playbackStateChanged.connect(callback)

        self.engine.resume()

        def wait_state(state):
            with self._condition:
                if self._state != state:
                    self._condition.wait(2)
            assert(self._state == state)

        self.engine.resume()
        wait_state(SimulationPlayer.FINISHED)
        assert(player.timestamp >= end - player.simulationDelta)

        # Playback should resume going backwards and finish again at the
        # simulation window start when setting a negative delta
        player.simulationDelta = -1.0
        wait_state(SimulationPlayer.FINISHED)
        assert(self._transitions[(SimulationPlayer.FINISHED,
                                  SimulationPlayer.PLAYING)] == 1)
        assert(player.timestamp <= start - player.simulationDelta)

        # Playback also resumes if a new timestamp within the window is set.
        player.timestamp = 3.8
        wait_state(SimulationPlayer.FINISHED)
        assert(self._transitions[(SimulationPlayer.PLAYING,
                                  SimulationPlayer.FINISHED)] == 3)

if __name__ == '__main__':
    unittest.main()
