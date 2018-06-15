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

import rtneuron as _rtneuron
import numpy as _numpy

class SelectionHandler(object):

    HIGHLIGHT = 0
    UNHIGHLIGHT = 1

    def __init__(self, view, background, broker=None):
        self._selection_action = None

        self._scene = view.scene

        if _rtneuron.options.sync_selections:
            self._init_synched_selections(view, background, broker)
        else:
            self._init_local_selections(view, background)

        def on_rectangular_selection(rect, select):
            self._selection_action = \
                self.HIGHLIGHT if select else self.UNHIGHLIGHT
            w = float(background.rect().width())
            h = float(background.rect().height())
            view.scene.pick(view, rect.left()/w, rect.right()/w,
                                  1 - rect.bottom()/h, 1 - rect.top()/h)

        background.rectangular_selection.connect(on_rectangular_selection)

    def _init_synched_selections(self, view, background, broker):
        session = _rtneuron.options.sync_selections

        self._broker = broker
        if not self._broker:
            if session == True:
                self._broker = _rtneuron.net.SceneEventBroker()
            else:
                self._broker = _rtneuron.net.SceneEventBroker(session)
            self._broker.trackState = _rtneuron.options.track_selections

        def on_cell_selected(gid, section, segment):
            self._broker.sendToggleRequest([gid])

        def on_cells_selected(target):
            # XXX

            # Depending on the action we want to toggle a different cell set.
            # For HIGHLIGHT, we request the toggle of the difference between
            # the selected cells and currently highlighted ones.
            # For UNHIGHLIGHT, we request the toggle of the intersection of
            # both sets.
            if self._selection_action == self.HIGHLIGHT:
                target = target - self._scene.highlightedNeurons
            else:
                target = self._scene.highlightedNeurons & target
            self._broker.sendToggleRequest(target)

        # We don't use the trackScene because it automatically toggles
        # at selection disregarding the current highlighting state. This
        # doesn't work for rectangular selections. The scene signals are
        # handled directly from here instead.
        self._scene.cellSelected.connect(on_cell_selected)
        self._scene.cellSetSelected.connect(on_cells_selected)

        def updateSelection(target):
            # XXX

            current = self._scene.highlightedNeurons
                # Unhighlighting deselected
            self._scene.highlight(current - target, False)
                # Highlighting newly selected
            self._scene.highlight(target - current, True)

        self._broker.cellsSelected.connect(updateSelection)

    def _init_local_selections(self, view, background):

        def on_cell_selected(gid, section, segment):
            # XXX

            highlighted = self._scene.highlightedNeurons
            self._scene.highlight(gid, not highlighted.contains(gid))

        def on_cells_selected(target):
            # XXX

            self._scene.highlight(
                target, self._selection_action == self.HIGHLIGHT)

        self._scene.cellSelected.connect(on_cell_selected)
        self._scene.cellSetSelected.connect(on_cells_selected)
