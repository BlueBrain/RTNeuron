#!/usr/bin/env python
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

from rtneuron import *
import bbp
import brain
from random import random, seed
import sys
seed(0)

# Environmental variables affecting the behaviour of Equalizer
import os
os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12' # offscreen
os.environ['EQ_LOG_LEVEL'] = 'ERROR'

# Creating the RTNeuron engine instance.
attributes = AttributeMap()
# Setting the default window size (this can be overriden with an Equalizer
# configuration file)
attributes.window_width = 600
attributes.window_height = 800
# The first argument is the command line arguments which are passed to the
# Equalizer initialization (and can be empty).
engine = RTNeuron([], attributes)

# Extracting the simulation config and target name from the command line
# If no arguments are provided the BBPTestData simulation config and the target
# MiniColumn_0 will be used.
if len(sys.argv) > 1 :
    simulation_config = sys.argv[1]
else :
    simulation_config= brain.test.blue_config
if not simulation_config :
    print("No simulation config found, please provide one")
if len(sys.argv) > 2 :
    target = sys.argv[2]
else :
    target = 'MiniColumn_0'

# Loading the experiment and the target. The function load_targets is
# provided by the RTNeuron module as a helper for that. Type
# "help(load_targets)" on the RTNeuron Python console for further details.
circuit = brain.Circuit(simulation_config)
target = circuit.gids(target)

# Creating a Neurons container for each morphology type. The containers are
# divided in two sets, one for excitatory and one for inhibitory cells.
excitatory = []
inhibitory = []

mtypes = circuit.morphology_types(target)

for i, mtype in enumerate(circuit.morphology_type_names()):
    container = excitatory if "PC" in mtype else inhibitory
    container.append(target[mtypes==i])

# Scene parameters.
scene_attributes = AttributeMap()
scene_attributes.use_meshes = False
# Enabling alpha-blending using the multi-layer depth peeling algorithm
alpha_blending = AttributeMap()
alpha_blending.mode = 'multilayer_depth_peeling'
alpha_blending.slices = 1
scene_attributes.alpha_blending = alpha_blending

# Creating the scene.
# This must be done before initializing any Equalizer configuration.
scene = engine.createScene(scene_attributes)
scene.circuit = circuit

# Adding the neuron containers to the scene. All neurons are added using
# the alpha-by-width color scheme. This mode mimics a translucency effect
# in which the transprency of the branches depends on their thickness.

# Excitatory cells are given color in the red-orange-yellow gamut.
for group in excitatory :
    attributes = AttributeMap()
    attributes.color_scheme = ColorScheme.BY_WIDTH
    red = random() * 0.4 + 0.6
    green = min(red, random() * 0.8)
    attributes.color = [red, green, 0, 0]
    attributes.secondary_color = [red, green, 0, 0.6]
    scene.addNeurons(group, attributes)
# Inhibitory cells are given color in the blue-cyan gamut
for group in inhibitory :
    attributes = AttributeMap()
    attributes.color_scheme = ColorScheme.BY_WIDTH
    blue = 0.5 + random() * 0.5
    green = min(blue, random())
    attributes.color = [0, green, blue, 0]
    attributes.secondary_color = [0, green, blue, 0.6]
    scene.addNeurons(group, attributes)

# Starting the Equalizer configuration. Since no configuration file is
# provided the default configuration will be used.
engine.init()

# Assigning the scene to the view. If not done nothing will be rendered.
view = engine.views[0] # By default there's only one view
view.scene = scene

# Configuring some view parameters
view.attributes.background = [0, 0, 0]
view.attributes.idle_AA_steps = 32 # Accumulation based anti-aliasing

# Creating a 2 second flythrough camera path. Note that the helper function
# reloads the blue config and the targets.
path = util.camera.Paths.flythrough(simulation_config, target, 2,
                                    air_pixels = [0, 0.2], norm = 1.8)
manipulator = CameraPathManipulator()
manipulator.setPath(path)

# Triggering one frame to make sure that the next frame rendered is already
# displaying the scene
engine.frame()

# Rendering 100 frames of the camera path. A custom loop is preferred over
# RTNeuron.record because the latter does not perform idle AA.
for i in range(100) :
    # Assigning the camera position for this frame
    view.camera.setView(*manipulator.getKeyFrame(i / 50.0)[:2])
    view.snapshot("frame_%.6d.png" % i)
