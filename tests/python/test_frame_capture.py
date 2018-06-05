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
import image_compare

import math
import os
os.environ['RTNEURON_FORCE_LINEAR_PATH_INTERPOLATION'] = '1'

import unittest

bluecfg = brain.test.blue_config

test_file_prefix = "record_test"
test_file_pattern = "^record_test_[0-9]{6}.png$"
IMAGE_SIZE = (840, 525)

def count_files(pattern) :
    """Count the number of files in the current working directory that
    match a given pattern.
    """
    import re
    regex = re.compile(pattern)
    count = 0
    for name in os.listdir(os.getcwd()) :
        if regex.match(name) :
            count += 1
    return count

def delete_files(pattern) :
    """Delete files in the current working directory that match a given
    pattern.
    """
    import re
    regex = re.compile(pattern)
    count = 0
    for name in os.listdir(os.getcwd()) :
        if regex.match(name) :
            os.unlink(name)

def display_circuit() :
    rtneuron.display_circuit(
        bluecfg, ('MiniColumn_0', {'mode' : rtneuron.RepresentationMode.SOMA}),
        eq_config = setup.medium_rect_eq_config)
    # Disabling idle AA and pausing to ensure that the background change and
    # idle AA don't disturb frame accounting. In principle the record function
    # also calls pause, but there's a race condition between redraw requests
    # and pause that makes frame accounting fail randomly otherwise
    # (BBPRTN-332).
    rtneuron.engine.pause()
    view = rtneuron.engine.views[0]
    view.attributes.idle_AA_steps = 0
    rtneuron.engine.waitFrame()
    view.attributes.background = [0, 0, 0, 1]

def record_and_check_frames(frames = 0, camera_path = None,
                            path_delta_secs = 0.1, use_waitRecord = True) :
    """Record, count frames and check the number of frames is the expected one.

    Record a some frames using a fixed number of frames or a camera path to
    determine the stop condition.
    Return the number of frames recorded.
    """
    params = rtneuron.RecordingParams()
    params.filePrefix = test_file_prefix
    if frames != 0 :
        params.frameCount = frames
    if camera_path :
        params.cameraPath = camera_path
        params.cameraPathDelta = path_delta_secs * 1000.0
        params.stopAtCameraPathEnd = True
    assert(frames != 0 or camera_path)

    rtneuron.engine.record(params)
    if use_waitRecord :
        rtneuron.engine.waitRecord()
        # waitRecord only waits for the frame to have been issued, waitFrame
        # waits for it to have been finished.
        rtneuron.engine.waitFrame()
    else :
        rtneuron.engine.waitFrames(frames)

    output_frames = count_files(test_file_pattern)
    # expected_frames is assigned something different from output_frame and
    # supposedly higher than the expected.
    expected_frames = output_frames + 1
    # The number final number of frames is determine by the first stop
    # condition met
    if frames :
        expected_frames = frames
    if camera_path :
        path_length = camera_path.stopTime - camera_path.startTime
        expected_frames = min(math.ceil(path_length / path_delta_secs),
                              expected_frames)

    assert(expected_frames == output_frames)

    return output_frames

def create_camera_path(keyframes) :
    path = rtneuron.CameraPath()
    for time, position, orientation in keyframes :
        key = rtneuron.CameraPath.KeyFrame()
        key.position = position
        key.orientation = orientation
        key.stereo_correction = 1
        path.addKeyFrame(time, key)
    return path

class TestRecordingParams(unittest.TestCase) :
    # Since assigning into non-existing attributes doesn't fail, let's
    # try that at least the known attributes haven't been changed inadvertly
    def test_attribute_names(self) :
        params = rtneuron.RecordingParams()
        hasattr(params, "simulationStart")
        hasattr(params, "simulationEnd")
        hasattr(params, "simulationDelta")
        hasattr(params, "cameraPathDelta")
        hasattr(params, "cameraPath")
        hasattr(params, "filePrefix")
        hasattr(params, "fileFormat")
        hasattr(params, "outputPath")
        hasattr(params, "stopAtCameraPathEnd")
        hasattr(params, "frameCount")

class TestFrameGrabbingWithFrameCount(unittest.TestCase):

    def setUp(self) :
        display_circuit()
        view = rtneuron.engine.views[0]
        view.attributes.idle_AA_steps = 0

    def tearDown(self) :
        delete_files(test_file_pattern)
        rtneuron.engine.exit()
        del rtneuron.engine

    def test_single_output_file(self) :
        record_and_check_frames(1)

    def test_wait_record(self) :
        record_and_check_frames(2)
        delete_files(test_file_pattern)
        record_and_check_frames(4)

    def test_wait_frames(self) :
        record_and_check_frames(2, use_waitRecord = False)
        delete_files(test_file_pattern)
        record_and_check_frames(4, use_waitRecord = False)

class TestFrameGrabbingWithCameraPath(unittest.TestCase):

    initial_position = [44, 732, 1487], ([0.0, 0.0, 1.0], 0.0)
    final_position = [44, 732, 156], ([0.0, 0.0, 1.0], 0.0)

    def setUp(self) :
        display_circuit()
        view = rtneuron.engine.views[0]
        view.attributes.idle_AA_steps = 0
        view.camera.setView(*self.initial_position)
        view.snapshot('record_test_first.png')
        view.camera.setView(*self.final_position)
        view.snapshot('record_test_last.png')

    def tearDown(self) :
        import os
        os.remove('record_test_first.png')
        os.remove('record_test_last.png')
        delete_files(test_file_pattern)
        rtneuron.engine.exit()
        del rtneuron.engine

    def test_path_deltas(self) :
        def test_delta(delta, extra_key_frame = True) :
            keyframes = [
                (0, self.initial_position[0], self.initial_position[1]),
                (1, self.final_position[0], self.final_position[1])]
            if extra_key_frame :
                # Since the camera path interval is open on the right, an extra
                # key frame is added to make sure that the recording stops at
                # the camera position given in the previous key frame
                keyframes.append(
                    (1 + delta, self.final_position[0], self.final_position[1]))

            path = create_camera_path(keyframes)

            frames = record_and_check_frames(camera_path = path,
                                             path_delta_secs = delta)

            image_compare.compare('record_test_000000.png',
                                  'record_test_first.png')
            if extra_key_frame :
                image_compare.compare('record_test_0000%.2d.png' % (frames - 1),
                                      'record_test_last.png')

            delete_files(test_file_pattern)
        test_delta(0.1)
        test_delta(0.33)
        test_delta(0.07)
        test_delta(1.5, extra_key_frame = False)

class TestStopConditions(unittest.TestCase):

    def setUp(self) :
        display_circuit()

    def tearDown(self) :
        delete_files(test_file_pattern)
        rtneuron.engine.exit()
        del rtneuron.engine

    def test_camera_path_against_frame_count(self) :
        keyframes = [
            (0, [44, 732, 1487], ([0.0, 0.0, 1.0], 0.0)),
            (1, [44, 731, 156], ([0.0, 0.0, 1.0], 0.0))]
        path = create_camera_path(keyframes)

        for delta in [0.1, 0.2, 0.4, 0.5] :
            record_and_check_frames(
                frames = 5, camera_path = path, path_delta_secs = delta)
            delete_files(test_file_pattern)

class TestBackBakgrounds(unittest.TestCase):
    def setUp(self) :
        self._engine = rtneuron.RTNeuron()
        self._engine.init(setup.medium_rect_eq_config)
        self._view = self._engine.views[0]

    def tearDown(self) :
        self._engine.exit()
        self._engine = None

    def capture_background(self, color) :
        name = image_compare._get_tmp_file_name()
        try :
            self._view.attributes.background = color
            self._view.snapshot(name)
            from PIL import Image
            return Image.open(name)
        finally :
            import os
            os.remove(name)

    def test_opaque(self) :
        image = self.capture_background([1, 1, 1, 1])
        assert(image.getbands() == ('R', 'G', 'B'))
        hist = image.histogram()
        pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        hist[255] == pixels
        hist[511] == pixels
        hist[767] == pixels

    def test_tranparent(self) :
        image = self.capture_background([1, 1, 1, 0])
        assert(image.getbands() == ('R', 'G', 'B', 'A'))
        hist = image.histogram()
        pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        hist[255] == pixels
        hist[511] == pixels
        hist[767] == pixels
        hist[768] == pixels

if __name__ == '__main__':
    unittest.main()

