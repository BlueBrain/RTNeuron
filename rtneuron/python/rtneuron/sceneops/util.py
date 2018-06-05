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

import numpy as _np
import re as _re

from .._rtneuron import AttributeMap as _AttributeMap
from ..util import key_to_gids as _key_to_gids
from . import NeuronClipping as _NeuronClipping

def _create_random_colors(count, rgb_ranges):

    def map_to_range(x, minmax):
        return x * (minmax[1] - minmax[0]) + minmax[0]

    import random
    # I don't find a way to get a generator object from random, so at least
    # I'll save the state to restore it later
    state = random.getstate()
    random.seed(0)
    colors = []
    for i in range(count):
        colors.append((map_to_range(random.random(), rgb_ranges[0]),
                       map_to_range(random.random(), rgb_ranges[1]),
                       map_to_range(random.random(), rgb_ranges[2])))
    random.setstate(state)
    return colors

def _create_colors(count, rgb_ranges):

    def map_to_range(x, minmax):
        return x * (minmax[1] - minmax[0]) + minmax[0]

    if (count > 30):
        return _create_random_colors(count, rgb_ranges)

    # For just one color return the midpoint of the range of each channel
    if count == 1:
        return [(map_to_range(0.5, rgb_ranges[0]),
                 map_to_range(0.5, rgb_ranges[1]),
                 map_to_range(0.5, rgb_ranges[2]))]

    try:
        import scipy.optimize
        import scipy.spatial
    except ImportError:
        print("Scipy not found, generating random colors")
        return _create_random_colors(count, rgb_range)

    dims = 3
    bounds = []
    for i in range(count):
        bounds.extend(rgb_ranges[0:dims])

    def obj_func(values):
        points = values.reshape(int(len(values)/dims), dims)
        distances = scipy.spatial.distance.pdist(points)
        return -min(distances)

    best = 0
    for n in range(max(int(200 / count), 1)):
        x = _np.random.random(count * dims)
        r = 0
        for i in range(len(x)):
            x[i] = map_to_range(x[i], rgb_ranges[r])
            r = (r + 1) % dims

        result = scipy.optimize.minimize(
            obj_func, x, bounds=bounds, method="SLSQP")
        if result.fun < best:
            best = result.fun
            solution = result
    return solution.x.reshape(count, dims)

def _create_exc_inh_colors(num_exc, num_inh):
    return (_create_colors(num_exc, [(0.6, 1.0), (0, 1.0), (0, 0.2)]),
            _create_colors(num_inh, [(0.0, 0.4), (0, 0.8), (0.4, 1.0)]))

def create_mtype_colors(gids, circuit):
    """Create three list containing the labels, gids, and generated
    colors of each of the morphological types found in a neuron set."""

    regex = _re.compile("(.*PC.*)|(.*_SS$)")

    mtype_names = circuit.morphology_type_names()
    excitatory = list(map(lambda x: regex.match(x) is not None, mtype_names))
    num_excitatory = _np.sum(excitatory)
    mtypes = circuit.morphology_types(gids)

    exc_colors, inh_colors = \
        _create_exc_inh_colors(num_excitatory, len(mtypes) - num_excitatory)

    e = 0
    i = 0
    targets = []
    colors = []
    for k in range(len(mtype_names)):
        target = gids[mtypes == k]
        targets.append(target)
        if excitatory[k]:
            r, g, b = exc_colors[e]
            e += 1
        else:
            r, g, b = inh_colors[i]
            i += 1
        colors.append((r, g, b, 1))

    return (mtype_names, targets, colors)

def create_metype_colors(gids, circuit):
    """Create three list containing the labels, Cell_Targets, and generated
    colors of each of the ME type combinations found in a neuron set."""

    regex = _re.compile("(.*PC.*)|(.*_SS$)")

    mtype_names = circuit.morphology_type_names()
    etype_names = circuit.electrophysiology_type_names()
    excitatory = list(map(lambda x: regex.match(x) is not None, mtype_names))
    mtypes = circuit.morphology_types(gids)
    etypes = circuit.electrophysiology_types(gids)

    # The first iteration creates the targets and counts the number of
    # excitatory ones because not all combinations exist.
    e = 0
    i = 0
    targets = []
    empty = set()
    num_excitatory = 0
    for j in range(len(mtype_names)):
        for k in range(len(etype_names)):
            target = gids[_np.logical_and(mtypes == j, etypes == k)]
            if target.size == 0:
                empty.add((j, k))
                continue
            targets.append(target)
            if excitatory[j]:
                num_excitatory += 1

    exc_colors, inh_colors = \
        _create_exc_inh_colors(num_excitatory, len(targets) - num_excitatory)

    colors = []
    names = []
    for j in range(len(mtype_names)):
        for k in range(len(etype_names)):
            if (j, k) in empty:
                continue
            if excitatory[j]:
                r, g, b = exc_colors[e]
                e += 1
            else:
                r, g, b = inh_colors[i]
                i += 1
            colors.append((r, g, b, 1))
            names.append(etype_names[k] + ' ' + mtype_names[j])

    return (names, targets, colors)

def enable_transparency(scene, algorithm="auto", **options):
    """Set up an alpha-blending rendering algorithm to use with a scene

    Parameters:
    - algorithm str: The name of the algorithm to use, one of:
      - 'auto': Let the implementation choose the best option automatically.
      - 'multilayer_depth_peeling': The preferred algorithm
      - 'fragment_linked_list': A buffer implementation only available if
        compiled with GL>=3 support.
      - 'depth_peeling': Simple depth peeling implementation, only for
         debugging purposes.

    Keyword parameters for multilayer_depth_peeling:
    - slices int: The number of slices to be used for the depth partition.

    Keyword parameters for fragment_linked_list:
    - alpha_cutoff_threshold: A float value in [0, 1] with the accumulated
      opacity at which fragments at the back can be considered occluded. The
      algorithm will try to discard these fragments as soon as possible.
    """
    attributes = _AttributeMap()
    attributes.mode = algorithm

    for key, value in options.items():
        # Preparing the code for different parameters and algorithms
        if algorithm == "multilayer_depth_peeling":
            if key == "slices":
                attributes.slices = int(value)
                continue
        elif algorithm == "fragment_linked_list":
            if key == "alpha_cutoff_threshold":
                attributes.alpha_cutoff_threshold = double(value)

        print("Warning: invalid parameter '%s' for alpha-blending "
              "algorithm %s" % (key, algorithm))

    scene.attributes.alpha_blending = attributes


def disable_transparency(scene):
    """Disable any alpha-blending rendering algorithm active in the given
    scene."""
    scene.attributes.alpha_blending = _AttributeMap()

def is_neuron_handler(handler):
    obj = handler.object
    return (type(obj) is _np.ndarray and obj.dtype == "u4" and
            len(obj.shape) == 1)

def get_neuron_subset(target, object=None, simulation=None):
    """Return a handler to the scene object representing the subset of the
    neurons given by target.

    If the origin object is not specified, the first neuron object of the
    scene of the first existing view is taken.
    If the simulation is not given, the simulation is taken from the
    rtneuron.simulation module variable.

    The parameter target can be a:
    - String with the target name.
    - String with the a regular expresion of the target name.
    - Single GID given as in integer.
    - A numpy array of type u4, u8 or i4
    - Iterable of GIDs.
    """

    if not simulation:
        simulation = _rtneuron.simulation
    if not object:
        for obj in _rtneuron.app.views[0].scene.objects:
            if is_neuron_handler(obj):
                break
        raise ValueError("No neuron object handler found in scene")
    return object.query(key_to_gids(target, simulation))

def get_neurons(scene):
    """Extract all the neurons and neuron object handlers from a scene.
    Return a tuple with a neuron container and the list of neuron object
    handlers or (None, []) if there are no neurons"""
    gids = _np.zeros((0), dtype="u4")
    neuron_objects = []
    for handler in scene.objects:
        if not is_neuron_handler(handler):
            continue
        gids = _np.union1d(gids, handler.object)
        neuron_objects.append(handler)
    return gids, neuron_objects

def colorize_scene_with_targets(scene, simulation, targets, colors):
    """Colorizes the neuron objects of a scene using the lists of target
    identifiers and colors given.

    The parameter targets is a list of target identifiers. A target identifier
    can be:
    - String with the target name.
    - String with the a regular expresion of the target name.
    - Single GID given as in integer.
    - List of GIDs.

    The parameters colors is a list of RGBA tuples.

    The code uses auxiliary subsets objects queried from the neuron objects.
    This means that if any of the neuron object handlers from the scene is
    updated, the color changes will be lost.
    """

    handlers = []
    for handler in scene.objects:
        if is_neuron_handler(handler):
            handlers.append((handler, handler.object))

    for key, color in zip(targets, colors):
        gids = _key_to_gids(key, simulation)

        for handler, ids in handlers:
            handler = handler.query(_np.intersect1d(ids, target))
            handler.attributes.color = color
            handler.update()

def colorize_neurons_with_targets(handler, simulation, targets, colors):
    """Colorizes the neurons of a set of handlers using a list  of per target
    colors.

    This is the same as colorsize_scene_with_targets but it takes a neuron
    object handler instead of a scene."""
    ids = handler.object
    for key, color in zip(targets, colors):
        target = _key_to_cell_target(key, simulation)
        subset = handler.query(_np.intersect1d(ids, target))
        subset.attributes.color = color
        subset.update()
