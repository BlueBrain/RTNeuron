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

from rtneuron import *
import brain

from random import seed
import sys
seed(0)

# Environmental variables affecting the behaviour of Equalizer
import os
os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12' # offscreen
os.environ['EQ_LOG_LEVEL'] = 'ERROR'

# Creating the RTNeuron application instance.
attributes = AttributeMap()
# Setting the default window size (this can be overriden with an Equalizer
# configuration file)
attributes.window_width = 600
attributes.window_height = 800
# The first argument is the command line arguments which are passed to the
# Equalizer initialization (and can be empty).
engine = RTNeuron([], attributes)

# Extracting the simulation config, the target and the report name from the
# command line. If no arguments are provided the default values are to use the
# BBPTestData simulation config, Layer1 as the target and voltage as the
# report
if len(sys.argv) > 1 :
    simulation_config = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'MiniColumn_0'
    report_name = sys.argv[3] if len(sys.argv) > 3 else 'voltage'
else :
    simulation_config= brain.test.blue_config
    target = 'L5CSPC'
    report_name = 'allCompartments'

if not simulation_config :
    print("No simulation config found, please provide one")
    exit(1)

simulation = brain.Simulation(simulation_config)

# Scene parameters.
scene_attributes = AttributeMap()
scene_attributes.use_meshes = False
scene_attributes.circuit = simulation_config
# Enabling alpha-blending using the multi-layer depth peeling algorithm
alpha_blending = AttributeMap()
alpha_blending.mode = 'multilayer_depth_peeling'
alpha_blending.slices = 1
scene_attributes.alpha_blending = alpha_blending

# Creating the scene.
# This must be done before initializing any Equalizer configuration.
scene = engine.createScene(scene_attributes)
scene.circuit = simulation.open_circuit()

# Adding the neuron target to the scene. Alpha-by-width is used as the
# coloring scheme. Combinaed with simulation display, only the alpha value
# of the primary color is actually used, because the RGB channels are taken
# from the simulation color map.
attributes = AttributeMap()
attributes.color_scheme = ColorScheme.BY_WIDTH
attributes.color = [1, 1, 1, 0.2]
scene.addNeurons(simulation.gids(target), attributes)

# Creating the compartment report object
report = apply_compartment_report(simulation, scene, report_name)

# Starting the Equalizer configuration. Since no configuration file is
# provided the default configuration will be used.
engine.init()

# Assigning the scene to the view. If not done nothing will be rendered.
view = engine.views[0] # By default there's only one view
view.scene = scene

# Configuring some view parameters
view.attributes.background = [1, 1, 1]
view.attributes.idle_AA_steps = 32 # Accumulation based anti-aliasing
view.attributes.display_simulation = True # Enabling simulation rendering

# Creating a color map for the simulation. This is a view property
colorMap = ColorMap()
points = {}
points[-10] = [1, 1, 0, 1]
points[-50] = [1, 0, 0, 0.5]
points[-65] = [0.5, 0.5, 0.5, 0]
points[-77] = [0, 0.5, 1, 0.5]
points[-80] = [0, 0, 1, 1]
colorMap.setPoints(points)
view.colorMap = colorMap

# Triggering one frame to make sure that the next frame rendered is already
# displaying the scene
engine.frame()

# Rendering loop
# A custom loop is preferred over RTNeuron.record because the latter does
# not perform idle AA.
metadata = report.metadata
start = (metadata["start_time"] + metadata["end_time"]) * 0.5
for i in range(10) :
    # Setting the timestamp to display.
    # Since the rendering loop hasn't been resumed, this assigment will not
    # triggering the rendering of a frame.
    engine.player.timestamp = start + i * metadata["time_step"] * 10
    view.snapshot('frame_%.2d.png' % i)
