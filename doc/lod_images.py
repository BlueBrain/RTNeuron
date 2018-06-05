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

import brain

attributes = []
no_mesh = AttributeMap({'use_meshes' : False})

def merge(a, b) :
    for k in dir(b) :
        a.__setattr__(k, b.__getattr__(k))

camera1 = ([87.08049011230469, 676.59033203125, 148.541748046875],
           ([-0.16840891540050507, 0.9842529892921448, 0.053707364946603775],
            40.37215805053711))
camera2 = ([22.241785049438477, 724.3768310546875, 21.460968017578125],
           ([0.03039763867855072, 0.992068886756897, 0.12196458876132965],
            184.392822265625))
lods = [("cylinders_no_mesh", ["high_detail_cylinders", "spherical_soma"],
                              no_mesh, camera1),
        ("mesh", ["mesh"], None, camera1),
        ("tubelets_no_mesh", ["tubelets", "spherical_soma"], no_mesh, camera1),
        ("tubelets", ["tubelets", "detailed_soma"], None, camera1),
        ("smooth_tubelets", ["tubelets", "detailed_soma"],
                            AttributeMap({"smooth_tubelets" : True}), camera2),
        ("sharp_tubelets", ["tubelets", "detailed_soma"], None, camera2),
        ("cylinders", ["high_detail_cylinders", "detailed_soma"],
                      None, camera1),
        ("low_cylinders", ["low_detail_cylinders", "spherical_soma"],
                          None, camera1),
        ("detailed_soma", ["detailed_soma"], None, camera1),
        ("soma", ["spherical_soma"], None, camera1)]

app = RTNeuron(sys.argv)

scenes = {}
cameras = {}
for name, lods, extra_attr, camera in lods :
    attributes = AttributeMap()
    attributes.use_cuda = False
    if extra_attr :
        merge(attributes, extra_attr)
    attributes.lod = AttributeMap()
    attributes.lod.neurons = \
        AttributeMap({lod : [0.0, float("+inf")] for lod in lods})
    scene = app.createScene(attributes)
    scene.circuit = brain.Circuit(brain.test.blue_config)
    scenes[name] = scene
    cameras[name] = camera

app.init()
app.resume()

view = app.views[0]
view.attributes.idle_AA_steps = 32
view.attributes.background = [1, 1, 1]
view.attributes.auto_compute_home_position = False

for name, scene in scenes.items() :
    # Reloading the circuit to clean up meshes
    scene.addNeurons(406, AttributeMap({'color' : [0.7, 0.85, 1.0, 1.0]}))

    view.scene = scenes[name]
    camera = cameras[name]
    view.camera.setView(camera[0], camera[1])
    view.snapshot('lod_' + name + '.png')
