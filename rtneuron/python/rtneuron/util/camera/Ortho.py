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

import numpy as _numpy
import brain as _brain

from rtneuron.util.camera import \
    _CommonOptions, _neurons_front_view, _neurons_top_view
import rtneuron as _rtneuron

def _ortho_frustum(positions, **kwargs) :
    options = _CommonOptions(**kwargs)
    from rtneuron import CameraPath

    maxX, maxY = positions.max(0)
    minX, minY = positions.min(0)
    height = maxY - minY
    width = maxX - minX
    width /= 1 - options.air_pixels[0]
    height /= 1 - options.air_pixels[1]

    halfWidth = max(width, height * options.aspect_ratio) * 0.5
    halfHeight = max(height, width / options.aspect_ratio) * 0.5

    return [-halfWidth, halfWidth, -halfHeight, halfHeight]

def front_ortho(simulation, targets, **kwargs) :
    """Return the frustum parameters for an orthogonal front view of a cell
    target.

    This function will load a simulation configuration file to get the soma
    position of the neurons given their gids. The frustum size is
    computed based only on these positions. In order to frame the
    scene correctly an additional camera path has to be set up.

    The target parameter can be:

    - A cell GID (as integer)
    - A numpy array of u4, u8 or i4
    - A target labels (as a string)
    - A list of any of the above

    """
    options = _CommonOptions(**kwargs)

    simulation = _brain.Simulation(simulation)
    circuit = simulation.open_circuit()
    gids = _rtneuron.util.targets_to_gids(targets, simulation)
    xys = options.fit_point_generator(gids, circuit)[:, [0, 1]]

    return _ortho_frustum(xys, **kwargs)

front_ortho.__doc__ += _CommonOptions.keyword_doc

def make_front_ortho(view, **kwargs) :
    """Setup the camera projection and position of the given view to do
    an orthographic front projection of the neurons on its scene.

    """
    options = _CommonOptions(**kwargs)

    scene = view.scene
    circuit = scene.circuit

    gids = _numpy.zeros((0), dtype="u4")
    for object in scene.objects :
        if _rtneuron.sceneops.is_neuron_handler(object):
            gids = _numpy.append(gids, object.object)

    view.camera.makeOrtho()
    ortho = view.camera.getProjectionOrtho()
    aspect_ratio = (ortho[1] - ortho[0]) / (ortho[3] - ortho[2])

    xys = options.fit_point_generator(gids, circuit)[:, [0, 1]]
    ortho = _numpy.array(
        _ortho_frustum(xys, aspect_ratio=aspect_ratio, **kwargs))
    view.camera.setProjectionOrtho(*(ortho / view.attributes.model_scale),
                                   near=1)

    path = _neurons_front_view(gids, circuit, aspect_ratio=aspect_ratio,
                               **kwargs)
    keyframe = path.getKeyFrames()[0][1]
    view.camera.setView(keyframe.position, keyframe.orientation)

make_front_ortho.__doc__ += _CommonOptions.keyword_doc

def top_ortho(simulation, targets, **kwargs) :
    """Return the frustum parameters for an orthogonal top view of a cell
    target.

    This function will load the blueconfig file given and all the neurons
    associated with the target specification. The frustum size is
    computed based only on the soma positions. In order to frame the
    scene correctly an additional camera path has to be set up.

    The target parameter can be:

    - A cell GID (as integer)
    - A numpy array of u4, u8 or i4
    - A target labels (as a string)
    - A list of any of the above

    """
    options = _CommonOptions(**kwargs)

    simulation = _brain.Simulation(simulation)
    circuit = simulation.open_circuit()
    gids = _rtneuron.util.targets_to_gids(targets, simulation)
    xzs = options.fit_point_generator(gids, circuit)[:, [0, 2]]

    return _ortho_frustum(xys, **kwargs)

top_ortho.__doc__ += _CommonOptions.keyword_doc


def make_top_ortho(view, **kwargs) :
    """Setup a the camera projection and position of the given view to do
    an orthographic top projection of the neurons on its scene.

    """
    from rtneuron import CameraPathManipulator

    options = _CommonOptions(**kwargs)

    scene = view.scene
    circuit = scene.circuit

    gids = _numpy.zeros((0), dtype="u4")
    for object in scene.objects :
        if _rtneuron.sceneops.is_neuron_handler(object):
            gids = _numpy.append(gids, object.object)

    view.camera.makeOrtho()
    ortho = view.camera.getProjectionOrtho()
    aspect_ratio = (ortho[1] - ortho[0]) / (ortho[3] - ortho[2])

    xzs = options.fit_point_generator(gids, circuit)[:, [0, 2]]
    ortho = _numpy.array(
        _ortho_frustum(xzs, aspect_ratio=aspect_ratio, **kwargs))
    view.camera.setProjectionOrtho(
        *(ortho / view.attributes.model_scale), near=1);

    path = _neurons_top_view(gids, circuit, aspect_ratio=aspect_ratio,
                             **kwargs)
    keyframe = path.getKeyFrames()[0][1]
    view.camera.setView(keyframe.position, keyframe.orientation)

make_top_ortho.__doc__ += _CommonOptions.keyword_doc
