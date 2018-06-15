# -*- coding: utf8 -*-
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

class CameraPathRecorder :
    '''Utility class to record the camera position and orientation of a view
    as a CameraPath in real-time

    version 2.4'''

    def __init__(self, engine, view) :
        self._engine = engine
        self._view = view
        self._cameraPath = None
        self._referenceTime = None

    def __del__(self) :
        self.stopRecording()

    def startRecording(self) :
        '''Start recording the camera position.

        For each frame issued by the engine the current camera
        position will be added to a camera path using a wall clock timestamp.
        The reference time is set when the first key-frame is registered'''
        from . import CameraPath
        self._referenceTime = None
        self._cameraPath = CameraPath()
        self._engine.frameIssued.connect(self)

    def stopRecording(self) :
        '''Stop registering the camera path.'''
        self._engine.frameIssued.disconnect(self)

    def replay(self) :
        '''Create a camera path manipulator with the recorded path and assign
        it to the view.'''
        if not self._cameraPath :
            return
        from . import CameraPathManipulator
        manipulator = CameraPathManipulator()
        manipulator.setPath(self._cameraPath)
        self._view.cameraManipulator = manipulator

    def getCameraPath(self) :
        '''Return the registered CameraPath'''
        return self._cameraPath

    def __call__(self) :
        # Internal method
        import time
        if not self._referenceTime :
            self._referenceTime = time.time()
            elapsed = 0
        else :
            elapsed = time.time() - self._referenceTime
        self._cameraPath.addKeyFrame(elapsed, self._view)


