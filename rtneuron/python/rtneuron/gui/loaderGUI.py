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

import rtneuron as _rtneuron

from . import BaseGUI
from .dialogs import LoadDialog, Progress

class LoaderGUI(BaseGUI):
    """ Creates a GUI that will initially show a modal dialog to choose
    a BlueConfig and target to load and then will show the loading progress.

    The RTNeuron engine is supposed to be initialized externally and set
    to the rtneuron.engine variable. The first view of the application object
    must have an scene already attached, presumably empty. This scene will be
    used to add the targets loaded. The brain.Simulation will be assigned
    to the module global variable rtneuron.simulation.

    Derived classes may reimplement the following methods:
    - scene_created(scene): Called when the neuron model creation has been
      completed.
    """

    def __init__(self, qml_file, simulation_config=None, target=None,
                 *args, **kwargs):
        super(LoaderGUI, self).__init__(qml_file, *args, **kwargs)

        self.loader = LoadDialog(self._overlay,
                                 default_config=simulation_config,
                                 default_target=target)
        self.loader.done.connect(self._on_load_dialog_done)

    def scene_created(self, scene):
        pass

    def _on_load_dialog_done(self, simulation, gids, display_mode):
        _rtneuron.simulation = simulation

        # Closing the dialog
        self.loader.close()
        self._simulation = simulation

        # Adding the targets to the scene
        view = _rtneuron.engine.views[0]
        scene = view.scene
        scene.circuit = simulation.open_circuit()

        self.progress = Progress(self._overlay)
        self.progress.done.connect(self._on_progress_done)
        scene.progress.connect(self.progress.step)
        self.progress.set_message("Loading data")

        attributes = _rtneuron.AttributeMap()
        if display_mode == "Soma":
            attributes.mode = _rtneuron.RepresentationMode.SOMA
        elif display_mode == "No axon":
            attributes.mode = _rtneuron.RepresentationMode.NO_AXON
        elif display_mode == "Detailed":
            attributes.mode = _rtneuron.RepresentationMode.WHOLE_NEURON

        scene.addNeurons(gids, attributes)

        # If something was typed in the loader dialog the base overlay has
        # lost the focus. Returning the focus because otherwise key presses
        # won't be captured.
        self._overlay.rootObject().setProperty("focus", True)

    def _on_progress_done(self, pass_number):

        if pass_number == 2:

            view = _rtneuron.engine.views[0]

            view.computeHomePosition() # We need to do this because the auto
                                       # home position is disabled

            scene = view.scene
            scene.progress.disconnect(self.progress.step)
            self.progress.close()
            del self.progress
            # The dialog can't be deleted just after closing it in
            # _on_load_dialog_done because Qt will crash internally (without
            # apparent reason). Deleting it from here is a workaround (leaving
            # it alive despite it won't be used again would be another option).
            del self.loader

            self.scene_created(scene)
