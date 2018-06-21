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

# Example script to show anterograde and retrograde selections in bbp
# microcircuit using the selection callback feature in RTNeuron.

import brain
import random
import argparse
import rtneuron
from rtneuron import *
from rtneuron.sceneops import SynapticProjections

circuit_config_default = \
    '/gpfs/bbp.cscs.ch/project/proj3/KaustCircuit/BlueConfig'
target_default = 'MiniColumn_0'

width = 1920
height = 1200

def anterograde_retrograde(circuit_config, target_name, shell):

   # Loading the target from the requested blueconfig
   circuit = brain.Circuit(circuit_config)

   # Creating the RTNeuron application instance.
   engine_attributes = AttributeMap()
   engine_attributes.window_width = width
   engine_attributes.window_height = height
   engine = RTNeuron([], engine_attributes)
   rtneuron.engine = engine

   # Creating the scene.
   scene_attributes = AttributeMap()
   alpha_blending = AttributeMap()
   alpha_blending.mode = 'multilayer_depth_peeling'
   alpha_blending.slices = 2
   scene_attributes.alpha_blending = alpha_blending
   scene_attributes.load_morphologies = False
   scene = engine.createScene(scene_attributes)
   scene.circuit = circuit

   # Creating the selection handler and connecting it to the scene.
   projections = SynapticProjections(scene,
                                     presynaptic_color = [0, 0, 1, 1],
                                     postsynaptic_color = [1, 0, 0, 0.5],
                                     unselected_color = [0, 0, 0, 0.2],
                                     target_mode = RepresentationMode.SOMA,
                                     clip_branches = False)

   # Adding the neuron target to the scene.
   neurons_attributes = AttributeMap()
   neurons_attributes.color_scheme = ColorScheme.SOLID
   neurons_attributes.color = [0.5, 0.5, 1.0, 0.2]
   neurons_attributes.mode = RepresentationMode.WHOLE_NEURON

   scene.addNeurons(circuit.gids(target_name), neurons_attributes)

   # Initializing the Equalizer configuration.
   engine.init()

   # Setting up the view attributes and attaching the scene to it.
   view = engine.views[0]
   view.scene = scene

   view.attributes.background = [1, 1, 1]
   view.attributes.auto_compute_home_position = False

   rtneuron.view = view

   # Releasing the rendering thread and starting the shell if requested.
   engine.resume()
   if shell:
      start_shell()
   engine.wait()

   del engine

if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument('--circuit-config', '-c', metavar="file",
                  default = circuit_config_default, help='CircuitConfig file')
   parser.add_argument('--target', metavar="name",
                       default = target_default, help='Target')
   parser.add_argument('--shell', action='store_true',
                       help='Enable interactive shell')
   args = parser.parse_args()
   anterograde_retrograde(args.circuit_config, args.target, args.shell)




