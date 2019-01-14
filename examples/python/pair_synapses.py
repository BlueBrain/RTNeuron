#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
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

from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass

from rtneuron import *
import brain

import argparse
import math
import numpy
import os

os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12'
os.environ['EQ_LOG_LEVEL'] = 'ERROR'

presynaptic_color = [1, 0.8, 0.2]
efferent_position_color = [1, 0.8, 0.4]
postsynaptic_color = [0.2, 0.4, 1.0]
afferent_position_color = [1.0, 0.2, 0.2]
resolution = [800, 1000]

def render_image(circuit_config, presynaptic_gid, postsynaptic_gid) :
    # Creating the engine instance with the desired resolution
    attributes = AttributeMap()
    attributes.window_width = resolution[0]
    attributes.window_height = resolution[1]
    engine = RTNeuron([], attributes)

    # Loading the cell data
    circuit = brain.Circuit(circuit_config)

    # Creating the scene
    scene_attributes = AttributeMap()
    if not args.use_meshes :
        # If it was requested not to use meshes, the tubelet mode is
        # configured
        lod = AttributeMap()
        lod.spherical_soma = [0, 1]
        lod.tubelets = [0, 1]
        scene_attributes.use_meshes = False
        scene_attributes.lod = AttributeMap()
        scene_attributes.lod.neurons = lod
    else :
        scene_attributes.em_shading = True

    scene = engine.createScene(scene_attributes)
    scene.circuit = circuit

    # Adding neurons
    def add_neuron(gid, color) :
        scene.addNeurons([gid], AttributeMap({"color": color}))

    add_neuron(presynaptic_gid, presynaptic_color)
    add_neuron(postsynaptic_gid, postsynaptic_color)

    # Adding synapses
    synapses = circuit.projected_synapses([presynaptic_gid], [postsynaptic_gid],
                                          brain.SynapsePrefetch.all)
    attributes = AttributeMap()
    attributes.radius = 4
    attributes.color = afferent_position_color
    scene.addAfferentSynapses(synapses, attributes)

    positions = numpy.column_stack((synapses.post_center_x_positions(),
                                    synapses.post_center_y_positions(),
                                    synapses.post_center_z_positions()))
    positions = numpy.vstack((positions,
                              circuit.positions([presynaptic_gid,
                                                 postsynaptic_gid])))

    engine.init()
    view = engine.views[0]
    view.scene = scene
    engine.frame()

    top = positions.max(0)
    bottom = positions.min(0)
    center = (top + bottom) * 0.5
    eye = numpy.array(center)
    vertical_fov = math.radians(view.camera.getProjectionPerspective()[0])
    tanVFOV = math.tan(vertical_fov * 0.5)
    tanHFOV = tanVFOV * resolution[0] / resolution[1]
    for x, y, z in positions :
        eye[2] = max(eye[2], z + (abs(y - center[1]) + 100) / tanVFOV)
        eye[2] = max(eye[2], z + (abs(x - center[0]) + 100) / tanHFOV)

    view.camera.setView(list(map(float, eye)), ([0.0, 0.0, 1.0], 0.0))
    view.attributes.auto_compute_home_position = False
    view.attributes.background = [1, 1, 1]
    view.scene = scene
    view.attributes.idle_AA_steps = 32
    view.snapshot('%d_%d.png' % (presynaptic_gid, postsynaptic_gid))

parser = argparse.ArgumentParser()
parser.add_argument(
    'blue_config', metavar = 'blueconfig', type = str,
    help = 'The circuit specification file')
parser.add_argument('infile', nargs = '?', type = argparse.FileType('r'),
                    default = None)
parser.add_argument(
    '--no-meshes', dest = 'use_meshes', action = 'store_false', default = True,
    help = 'Do not use meshes. EM shading is also disabled by this option')
args = parser.parse_args();

if not args.infile :
    pre = None
    post = None
    try :
        pre = int(input("Insert a presynaptic gid: "))
        post = int(input("Insert a postsynaptic gid: "))
    except Exception as e:
        print("Error reading cell GID:", e)
        exit(-1)
    render_image(args.blue_config, pre, post)
else :
    for pair in args.infile :
        pair = map(int, pair.split())
        render_image(args.blue_config, pair[0], pair[1])




