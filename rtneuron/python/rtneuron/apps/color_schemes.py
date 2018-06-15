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
import os

import rtneuron as _rtneuron
from rtneuron.gui import LoaderGUI, display_empty_scene_with_GUI

class GUI(LoaderGUI):
    def __init__(self, *args, **kwargs):
        super(GUI, self).__init__(
            os.path.dirname(os.path.realpath(__file__)) + '/ColorSchemes.qml',
            *args, **kwargs)

        qml = self._overlay.rootObject()
        qml.updateScene.connect(self._on_update)

    def _init_implementation(self):
        pass

    def _on_update(self, scheme, primary, secondary, attenuation):
        scene = _rtneuron.engine.views[0].scene

        handler = scene.objects[0]
        if scheme == "0":
            scheme = _rtneuron.ColorScheme.SOLID
        elif scheme == "1":
            scheme = _rtneuron.ColorScheme.BY_WIDTH
        elif scheme == "2":
            scheme = _rtneuron.ColorScheme.BY_DISTANCE_TO_SOMA
        elif scheme == "3":
            scheme = _rtneuron.ColorScheme.BY_BRANCH_TYPE
        else:
            print("Unknown color scheme requested")
            return
        handler.attributes.color_scheme = scheme

        if not hasattr(handler.attributes, "extra"):
            handler.attributes.extra = _rtneuron.AttributeMap()

        handler.attributes.extra.attenuation = attenuation

        handler.attributes.primary_color = [
            primary.redF(), primary.greenF(), primary.blueF(), primary.alphaF()]
        handler.attributes.secondary_color = [
            secondary.redF(), secondary.greenF(), secondary.blueF(),
            secondary.alphaF()]

        if primary.alphaF() != 1.0 or secondary.alphaF() != 1.0:
            _rtneuron.sceneops.enable_transparency(scene)
        else:
            _rtneuron.sceneops.disable_transparency(scene)

        handler.update()

class App(object):
    """Color schemes demo app."""

    def __init__(self):
        attributes = _rtneuron.default_scene_attributes
        # Don't generate meshes because the default scene lacks plenty of them
        attributes.generate_meshes = False

        self._gui = display_empty_scene_with_GUI(GUI, attributes)

        view = _rtneuron.engine.views[0]
        view.attributes.auto_compute_home_position = False

def start(*args, **kwargs):
    """Startup the color schemes demo.
    """
    return App()
