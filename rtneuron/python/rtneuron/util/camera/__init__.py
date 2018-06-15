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

import math as _math
import numpy as _np

import rtneuron as _rtneuron
import brain as _brain

def soma_positions(gids, circuit):
    """Return the position of the soma of a given neuron."""
    return circuit.positions(gids)

def dendrite_endpoint_positions(gids, circuit):

    morphologies = circuit.load_morphologies(gids, circuit.Coordinates.global_)

    positions = circuit.positions(gids)
    for morphology in morphologies:
        S = _brain.neuron.SectionType
        neurites = [S.dendrite, S.apical_dendrite]
        # Finding the terminal point of each section without children
        terminals = _np.array(
            [s[-1][:3] for s in morphology.sections(neurites) if
             len(s.children()) == 0])
        # If this list of points is too long for later, we could try to reduce
        # it to only those points in the convex hull
        positions = _np.vstack((positions, terminals))
    return positions

class _CommonOptions(object):

    keyword_doc = """
    The keyword arguments used to fit the viewpoint are:

    - air_pixels: a list of two floats with the desired fraction of empty
      horizontal and vertical space. This applies to the method used to
      frame the cell somas, i.e, branches are not considered.
      The default value is [0.1, 0.1].
    - fit_point_generator: the function used to determine the position
      of a neuron.
      Possible values:

      - rtneuron.util.camera.soma_positions              (default, fastest)
      - rtneuron.util.camera.dendrite_endpoint_positions (more precise, slower)

    - point_radius: If provided, the points to be bound will be considered
      spheres of the given radius. If ommited and the fit point list has
      a single point, it will be considered as a sphere of radius 200.
    - aspect_ratio: ratio of width/height of the image.
      The default corresponds to the default frustum
    - vertical_fov: vertical field of view angle in degrees of the camera
      to be used.
      The default corresponds to the default frustum.
    """

    vertical_fov = _math.atan(0.5) * 2
    aspect_ratio = 0.8 / 0.5
    point_radius = None

    air_pixels = [0.1, 0.1] # Relative measures in screen spaces
                            # These are minimum values, actual values depend
                            # on the screen and model aspect ratios.

    fit_point_generator = staticmethod(soma_positions)

    def __init__(self, **kwargs):
        try:
            self.aspect_ratio = kwargs['aspect_ratio']
        except KeyError:
            pass
        try:
            self.point_radius = kwargs['point_radius']
        except KeyError:
            pass
        try:
            self.vertical_fov = _math.radians(kwargs['vertical_fov'])
        except KeyError:
            pass
        try:
            self.air_pixels = kwargs['air_pixels']
        except KeyError:
            pass
        try:
            self.fit_point_generator = kwargs['fit_point_generator']
        except KeyError:
            pass

def _compute_eye_position(positions, options):
    """Internal

    Compute the eye position for a list of points with given aspect_ratio and
    vertical_fov options. First the center of the AABB face with max z is
    computed, then the eye position is adjusted to include all points in the
    field of view.
    """
    top = positions.max(0)
    bottom = positions.min(0)
    center = (top + bottom) * 0.5
    eye = [center[0], center[1], top[2]]

    tanVFOV = _math.tan(options.vertical_fov / 2.0)
    tanHFOV = tanVFOV * options.aspect_ratio
    # Adjusting the effective field of view based on the desired empty space
    tanVFOV *= 1.0 - options.air_pixels[1]
    tanHFOV *= 1.0 - options.air_pixels[0]
    z_by_y = 0
    z_by_x = 0
    # This calculation is not perfect because it doesn't adjust the center
    # position to fit the scene tighly on all the screen borders, but the
    # correct right solution seems to be complicated.
    for x, y, z in positions:
        z_by_x = max(z_by_x, z - bottom[2] + abs(x - center[0]) / tanHFOV)
        z_by_y = max(z_by_y, z - bottom[2] + abs(y - center[1]) / tanVFOV)

    # Adjuting the final position to consider spheres with radius instead
    # of points if needed.
    radius = options.point_radius
    if len(positions) == 1 and radius == None:
        radius = 200
    if radius:
        assert(radius > 0)
        sinVFOV = tanVFOV / _math.sqrt(1 + tanVFOV**2)
        sinHFOV = tanHFOV / _math.sqrt(1 + tanHFOV**2)
        z_by_y += radius / sinVFOV
        z_by_x += radius / sinHFOV

    eye[2] = eye[2] + max(z_by_x, z_by_y)

    return _np.array(eye)

def _neurons_front_view(gids, circuit, **kwargs):
    options = _CommonOptions(**kwargs)

    points = options.fit_point_generator(gids, circuit)
    eye = _compute_eye_position(points, options)

    path = _rtneuron.CameraPath()
    path.addKeyFrame(0, path.KeyFrame(eye, ([1, 0, 0], 0), 1))

    return path

def _neurons_top_view(gids, circuit, **kwargs):
    options = _CommonOptions(**kwargs)

    points = options.fit_point_generator(gids, circuit)
    points = points[:,[0, 2, 1]]

    eye = _compute_eye_position(points, options)
    eye[1], eye[2] = eye[2], eye[1]

    path = _rtneuron.CameraPath()
    path.addKeyFrame(0, path.KeyFrame(eye, ([1, 0, 0], -90), 1))

    return path

def set_manipulator_home_position(view, target, **kwargs):
    """Sets the home positions of the camera manipulator of the given view
    to a front view of the input cell target.

    The input target can be one of:
    - single cell GID as integer
    - a numpy array of u4, i4 or u8
    - a string with a target label (regular expressions included)
    - an iterable object, each element being a cell identifier

    """
    try:
        view.cameraManipulator.setHomePosition
    except AttributeError:
        raise ValueError(
            "This view's camera manipulator doesn't have a home position")

    options = _CommonOptions(**kwargs)

    target = _rtneuron.util.key_to_gids(target, _rtneuron.simulation)
    circuit = _rtneuron.simulation.open_circuit()

    center = circuit.positions(target).mean(axis=0)
    eye = _compute_eye_position(positions, options)

    view.cameraManipulator.setHomePosition(eye, center, [0, 1, 0])

set_manipulator_home_position.__doc__ += _CommonOptions.keyword_doc

from . import Ortho, Paths, PathRecorder
