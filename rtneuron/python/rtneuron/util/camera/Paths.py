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

import math as _math
import numpy as _numpy
import brain as _brain
import rtneuron as _rtneuron

from rtneuron.util.camera import \
    _CommonOptions, _compute_eye_position, \
    _neurons_front_view, _neurons_top_view

def _compute_orientation(z, up):
    """Return the axis, angle rotation corresponding to a camera orientation
    defined by the given z (front) and up normalized axis.

    The camera will look down the negative z axis. The x axis (right) is
    implicitly defined by the cross product up ^ z. The real up vector of
    the camera is then computed as z ^ x
    """

    # These column vectors define the base change matrix that defines the
    # rotation matrix. The base is created following OpenGL conventions
    # (looking down negative z, y is up, x is right).
    x = _numpy.cross(up, z)
    x /= _numpy.linalg.norm(x)
    y = _numpy.cross(z, x)

    # The orientation axis and angle are extracted from the rotation
    # in matrix form:
    #
    # c(a) + x^2(1-c(a))   xy(1-c(a)) - z s(a)  xz(1-c(a)) + y s(a)
    # xy(1-c(a)) + z s(a)  c(a) + y^2(1-c(a))   yz(1-c(a)) - x s(a)
    # xz(1-c(a)) - y s(a)  yz(1-c(a)) + x s(a)  c(a) + z^2(1-c(a))
    #
    # Where each column is look_x, look_y, look_z, c(a) = cos(angle) and
    # s(a) = sin(angle) and [x, y, z] is the (normalized) rotation axis.
    trace = x[0] + y[1] + z[2]
    cos_a = (trace - 1) * 0.5
    if cos_a == 1:
        # No rotation
        return ([1, 0, 0], 0)
    elif cos_a == -1:
        angle = _math.acos(cos_a) / _math.pi * 180
        axis = _numpy.sqrt((_numpy.array([x[0], y[1], z[2]]) + 1) * 0.5)
        return (axis, angle)
    else :
        # u = axis * sin(angle)
        u = _numpy.array([y[2] - z[1], z[0] - x[2], x[1] - y[0]]) * 0.5
        # u dot u = sin^2(angle)
        sin_a = _math.sqrt(_numpy.dot(u, u))
        # A rotation can by described as either axis, angle or
        # -axis, -angle. The chosen convention is that axis[0] > 0 or
        # axis[0] == 0 and axis[1] > 0 or axis[0] == 0 and axis[1] == 0 and
        # axis[2] > 0.
        # The rationale is to avoid potential interpolation problems around
        # critical points where axis and angles could be inverted.
        if (u[0] < 0 or (u[0] == 0 and u[1] < 0) or
            (u[0] == 0 and u[1] == 0 and u[2] < 0)):
            sin_a = -sin_a
        axis = u / sin_a
        angle = _math.acos(cos_a) / _math.pi * 180
        if sin_a < 0:
            angle = 360 - angle
        return (axis, angle)

def circular_ease_in_out(t) :
    t *= 2
    if t < 1:
        return 0.5 - 0.5 * _math.sqrt(1 - t**2)
    t -= 2
    return 0.5 + 0.5 * _math.sqrt(1 - t**2)

def quadratic_ease_in_out(t) :
    t *= 2
    if t < 1:
        return 0.5 * t**2
    t = 2 - t
    return 0.5 + 0.5 * t**2

def front_view(simulation, targets, **kwargs):
    """Return a camera path with a front view a cell target.

    This function will load the blue config file given and all the neurons
    associated with the target specification. The camera position is
    computed based only on the soma positions.

    The target parameter can be:

    - A cell GID (as integer)
    - An iterable of GIDs
    - A target label (as a string)
    - A list of any of the above
    """
    return _neurons_front_view(_load_targets(simulation, targets), **kwargs)

front_view.__doc__ += _CommonOptions.keyword_doc

def make_front_view(view, **kwargs):
    """Setup the camera position of the given view to look from the front
    at the neurons on the view's scene.
    """
    scene = view.scene
    circuit = scene.circuit

    gids = _numpy.zeros((0), dtype="u4")
    for object in scene.objects :
        if _rtneuron.sceneops.is_neuron_handler(object):
            gids = _numpy.append(gids, object.object)

    path = _neurons_front_view(gids, circuit, **kwargs)
    keyframe = path.getKeyFrames()[0][1]
    view.camera.setView(keyframe.position, keyframe.orientation)

make_front_view.__doc__ += _CommonOptions.keyword_doc

def top_view(simulation, targets, **kwargs):
    """Return a camera path with a top view of a cell target.

    This function will load the simulation config file given and all the neurons
    associated with the target specification. The camera position is
    computed based only on the soma positions.

    The target parameter can be:

    - A cell GID (as integer)
    - An iterable of GIDs
    - A target labels (as a string)
    - A list of any of the above
    """
    return _neurons_top_view(_load_targets(simulation, targets), **kwargs)

top_view.__doc__ += _CommonOptions.keyword_doc

def make_top_view(view, **kwargs):
    """Setup the camera position of the given view to look from the top
    at the neurons on the view's scene.
    """
    scene = view.scene
    circuit = scene.circuit

    gids = _numpy.zeros((0), dtype="u4")
    for object in scene.objects :
        if _rtneuron.sceneops.is_neuron_handler(object):
            gids = _numpy.append(gids, object.object)

    path = _neurons_top_view(gids, circuit, **kwargs)
    keyframe = path.getKeyFrames()[0][1]
    view.camera.setView(keyframe.position, keyframe.orientation)

make_top_view.__doc__ += _CommonOptions.keyword_doc

def rotation(look_at, axis, start, angle, up=[0, 1, 0], duration=10, **kwargs):
    """Return a camera path of a rotation around an arbitrary axis with a
    fixation point.

    The parameters are:

    - look_at: The point at which the camera will look. The rotation axis
      is also placed at this point.
    - axis: A normalized vector used as rotation axis. The rotation sense
      is defined applying the right-hand rule to this vector.
    - start: The initial camera position. The distance of this point to center
      is preserved.
    - angle: The rotation angle of the final position in radians.
    - up: A normalized vector or one of the strings "axis" or "tangent".
      This parameter defines the vector to which the y axis of the camera is
      aligned. If a normalized vector is given, that direction is used.  For
      "axis", the axis direction is used. For "tangent", the up direction lies
      on the rotation plane and is tangent to the circular trajectory.

    The optional keyword arguments are:

    - samples: Number of keyframes to generate
    - timing: A [0..1]->[0..1] function used to map sample timestamps
      from a uniform distribution to any user given distribution.
    """

    path = _rtneuron.CameraPath()

    try:
        samples = kwargs['samples']
        assert(samples > 1)
    except KeyError:
        samples = 100
    try:
        timing = kwargs['timing']
    except KeyError:
        timing = lambda x : x

    look_at = _numpy.array(look_at)
    start = _numpy.array(start)
    axis = _numpy.array(axis)
    if type(up) is str and up == "axis":
        up = axis
    else:
        up = _numpy.array(up)

    center = look_at + axis * _numpy.dot(axis, start - look_at)
    x = start - center
    distance = _numpy.linalg.norm(x)
    if distance == 0:
        print("Warning: start point lies on rotation axis. "
              "No rotation performed")
        return
    x /= distance
    y = _numpy.cross(axis, x)

    for i in range(samples):
        # Computing the normalized sample position
        i = 1 / (samples - 1.0) * i
        # Camera position
        phi = angle * timing(i)
        position = center + x * _math.cos(phi) * distance + \
                            y * _math.sin(phi) * distance

        camera_z = position - look_at
        camera_z /= _numpy.linalg.norm(camera_z)
        if type(up) is str and up == "tangent":
            orientation = _compute_orientation(
                camera_z, _numpy.cross(position - center, axis))
        else:
            orientation = _compute_orientation(camera_z, up)

        time = duration * i
        path.addKeyFrame(time, path.KeyFrame(position, orientation, 1))

    return path

def front_to_top_rotation(simulation, targets, duration=10, **kwargs):
    """Return a camera path of a rotation from front to top view of
    a set of circuit targets.

    This function will load the simulation config file given and all the neurons
    associated with the target specification. The front and top camera
    positions are  computed based on the soma positions.
    The path duration is in seconds.

    The targets parameter can be:

    - A cell GID (as integer)
    - An iterable of GIDs
    - A target labels (as a string)
    - A list of any of the above

    The optional keyword arguments are:

    - timing: A [0..1]->[0..1] function used to map sample timestamps
      from a uniform distribution to any user given distribution.
    - samples: Number of keyframes to generate
    """
    options = _CommonOptions(**kwargs)

    simulation = _brain.Simulation(simulation)
    circuit = simulation.open_circuit()
    gids = _rtneuron.util.targets_to_gids(targets, simulation)
    positions = options.fit_point_generator(gids, circuit)

    center = (positions.max(0) + positions.min(0)) * 0.5

    # Computing the front position ensuring that after the rotation the circuit
    # will be correctly frames in the top view.
    front_eye = _compute_eye_position(positions, options)
    positions = positions[:,[0, 2, 1]]
    top_eye = _compute_eye_position(positions, options)
    top_eye[1], top_eye[2] = top_eye[2], top_eye[1]

    eye = front_eye
    if abs(eye[2] - center[2]) < abs(top_eye[1] - center[1]):
        eye[2] += (top_eye[1] - center[1]) - (eye[2] - center[2])

    return rotation(center, [1, 0, 0], eye, -_math.pi/2, up="tangent",
                    duration=duration, **kwargs)

front_to_top_rotation.__doc__ += _CommonOptions.keyword_doc

def rotate_around(simulation, targets, duration=10, **kwargs):
    """Return a camera path of a front view rotation around a circuit target.

    This function will load the simulation config file given and all the neurons
    associated with the target specification. The start position is computed
    based on the soma positions. The path duration is in seconds.

    The target parameter can be:

    - A cell GID (as integer)
    - An iterable of GIDs
    - A target labels (as a string)
    - A list of any of the above
    """
    options = _CommonOptions(**kwargs)

    simulation = _brain.Simulation(simulation)
    circuit = simulation.open_circuit()
    gids = _rtneuron.util.targets_to_gids(targets, simulation)
    positions = options.fit_point_generator(gids, circuit)

    center = (positions.max(0) + positions.min(0)) * 0.5
    eye = _compute_eye_position(positions, options)
    return rotation(center, [0, 1, 0], eye, 2 * _math.pi, duration=duration)

rotate_around.__doc__ += _CommonOptions.keyword_doc

def flythrough(simulation, targets, duration=10, **kwargs):
    """Return a camera path of a flythrough of a cell target.

    This function will load the simulation config file given and all the neurons
    associated with the target specification. The camera position is
    computed based only on the soma positions and corresponds to a front
    view of the circuit. The path duration must be in seconds.

    The targets parameter can be:

    - A cell GID (as integer)
    - An iterable of GIDs
    - A target labels (as a string)
    - A list of any of the above

    The optional keyword arguments are:

    - samples: Number of keyframes to generate
    - speedup: From 1 to inf, this parameter specifies a speed up for
      the initial camera speed. Use 1 for a linear camera path, if higher
      that one the camera will start faster and will decrease its speed
      non-linearly and monotonically. Recommended values are between 1
      and 3.
      The default value is 1.
    """
    options = _CommonOptions(**kwargs)
    try:
        samples = kwargs['samples']
        assert(samples > 1)
    except KeyError:
        samples = 100
    try:
        norm = float(kwargs['speedup'])
        assert(norm >= 1)
    except KeyError:
        norm = 1

    simulation = _brain.Simulation(simulation)
    circuit = simulation.open_circuit()
    positions = circuit.positions(
        _rtneuron.util.targets_to_gids(targets, simulation))

    top = positions.max(0)
    bottom = positions.min(0)
    center = (top + bottom) * 0.5
    start = _compute_eye_position(positions, options)
    end = center
    end[2] = bottom[2] - (top[2] - bottom[2]) / 5.0

    path = _rtneuron.CameraPath()

    for i in range(samples):
        i = 1 / (samples - 1.0) * i
        time = duration * i
        a = (1 - (1 - i) ** norm) ** (1 / norm)
        position = start * (1 - a) + end * a
        path.addKeyFrame(time, path.KeyFrame(position, ([1, 0, 0], 0), 1))

    return path

flythrough.__doc__ += _CommonOptions.keyword_doc

def apply_camera_path(path, view):
    """Apply a camera path to a view using a  camera path manipulator."""
    manipulator = _rtneuron.CameraPathManipulator()
    manipulator.setPath(path)
    view.cameraManipulator = manipulator
