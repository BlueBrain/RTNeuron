#!/usr/bin/python
# -*- coding: utf8 -*-
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

import bbp
import brain
from rtneuron import *
from random import random
import sys

attributes = AttributeMap()
attributes.window_width = 1200
attributes.window_height = 1600
import os
os.environ['EQ_WINDOW_IATTR_HINT_DRAWABLE'] = '-12'
os.environ['EQ_LOG_LEVEL'] = 'ERROR'
engine = RTNeuron([], attributes)

if len(sys.argv) > 1 :
    blue_config = sys.argv[1]
else :
    blue_config= brain.test.blue_config
if not blue_config :
    print("No blue config found, please provide one")
if len(sys.argv) > 2 :
    target = sys.argv[2]
else :
    target = None

simulation = brain.Simulation(blue_config)

gids = simulation.gids(target) if target else simulation.gids()
circuit = simulation.open_circuit()

inhibitory_gids = circuit.gids("Inhibitory")
excitatory_gids = circuit.gids("Excitatory")

inhibitory = circuit.projected_synapses(inhibitory_gids, gids)
excitatory = circuit.projected_synapses(excitatory_gids, gids)

scene_attributes = AttributeMap()
alpha_blending = AttributeMap()
alpha_blending.mode = 'multilayer_depth_peeling'
alpha_blending.slices = 1
scene_attributes.alpha_blending = alpha_blending
scene = engine.createScene(scene_attributes)
scene.circuit = circuit
progress = ProgressBar()

scene.addAfferentSynapses(excitatory, AttributeMap({'color' : [1, 0, 0, 0.5],
                                                    'radius' : 1}))
scene.addAfferentSynapses(inhibitory, AttributeMap({'color' : [0, 0, 1, 0.5],
                                                    'radius' : 1}))

engine.init()

view = engine.views[0]
view.attributes.background = [1, 1, 1]
view.scene = scene
view.snapshot('synapses.png')



