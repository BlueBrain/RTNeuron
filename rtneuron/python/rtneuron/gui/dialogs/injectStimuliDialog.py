# -*- coding: utf-8 -*-
## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Ahmet Bilgili <ahmet.bilgili@epfl.ch>
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

from PyQt5 import QtCore

from rtneuron import nest

import rtneuron.sceneops.util as sceneops

from .QMLDialog import *

__all__ = ['InjectStimuliDialog']

class GeneratorListModel(QtCore.QAbstractListModel):

    update_model_signal = QtCore.pyqtSignal(name='updateModel')

    def __init__(self, parent=None, *args):

        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self._data = []
        for generator in nest.Models():
             if "generator" in generator:
                 self._data.append(generator)

        self.update_model_signal.connect(self._update_model)
        self._roles = {QtCore.Qt.DisplayRole: "generatorName"}

    def roleNames(self):
        return self._roles

    def _update_model(self):
        self.layoutChanged.emit()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def data(self, index, role):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self._data[index.row()])
        else:
            return QtCore.QVariant()


class GeneratorDetailModel(QtCore.QAbstractListModel):

    update_item_signal = QtCore.pyqtSignal('int', 'QVariant', name='updateItem')
    update_model_signal = QtCore.pyqtSignal(name='updateModel')
    valid = QtCore.pyqtSignal('bool')

    def __init__(self, parent=None, *args):

        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self._roles = {QtCore.Qt.UserRole + 0: b'generatorDetailKey',
                       QtCore.Qt.UserRole + 1: b'generatorDetailValue',
                       QtCore.Qt.UserRole + 2: b'generatorDetailValid'}

        self.update_item_signal.connect(self._update_item)
        self.update_model_signal.connect(self._update_model)

        self._editable_types = set()
        self._editable_types.add(int.__name__)
        self._editable_types.add(float.__name__)

        self._generator_keys = []
        self._generator_values = []
        self._generator_valid = []
        self._generator_types = []
        self._generator_name = ""

    def roleNames(self):
        return self._roles

    def get_generator_parameters(self):
        ret = {"model": nest.SLILiteral(self._generator_name)}
        for i in range(len(self._generator_keys)):
            ret[self._generator_keys[i]] = self._generator_values[i]

        return ret

    def set_generator(self, generator_name):

        self.beginResetModel()
        self._generator_name = generator_name
        self._generator_keys = []
        self._generator_values = []
        generator_dict = nest.GetDefaults(generator_name)

        keys = list(generator_dict.keys())
        keys.sort()

        for key in keys:

            # Trying to create a generator applying the default value on
            # the key, it NEST complains, we assume that the property is
            # read-only.
            generator = nest.Create(generator_name)
            try:
                parameters = {key: generator_dict[key]}
                nest.SetStatus(generator, parameters)
            except nest.pynestkernel.NESTError as e:
                if "Unused" in str(e):
                    nest.ResetKernel()
                    continue
            nest.ResetKernel()

            value = generator_dict[key]
            test_set = set()
            test_set.add(type(value).__name__)
            if len(self._editable_types.intersection(test_set)) > 0:
                self._generator_keys.append(key)
                self._generator_values.append(value)
                self._generator_valid.append(True)
                self._generator_types.append(type(value))

        self.valid.emit(self.isValid())
        self.endResetModel()

    def _update_item(self, index, value):
        try:
            self._generator_values[index] = self._generator_types[index](value)
            self._generator_valid[index] = True
        except (ValueError, TypeError):
            self._generator_values[index] = value
            self._generator_valid[index] = False

        modelIndex = self.index(index)
        self.dataChanged.emit(modelIndex, modelIndex)
        self.valid.emit(self.isValid())

    def _update_model(self):
        self.layoutChanged.emit()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._generator_keys)

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()

        if role == QtCore.Qt.UserRole:
            return QtCore.QVariant(self._generator_keys[index.row()])
        elif role == QtCore.Qt.UserRole + 1:
            return QtCore.QVariant(self._generator_values[index.row()])
        elif role == QtCore.Qt.UserRole + 2:
            return QtCore.QVariant(self._generator_valid[index.row()])

    def isValid(self):
        return all(self._generator_valid)

class InjectStimuliDialog(QMLDialog):
    """Create the dialog for editing and injecting stimulus.

    The injectStimuli dialog provides the list of generators and their
    parameters as an editable list. Users can inject different generators to
    the neurons in the scene.
    """
    inject_stimuli_done = QtCore.pyqtSignal()

    def __init__(self, parent):

        self._generator_list_model = GeneratorListModel()
        self._generator_detail_model = GeneratorDetailModel()
        context = parent.engine().rootContext()

        context.setContextProperty("generatorDetailModel",
                                   self._generator_detail_model)
        context.setContextProperty("generatorListModel",
                                   self._generator_list_model)

        super(InjectStimuliDialog, self).__init__(
            parent, "dialogs/InjectStimuliDialog.qml")

        # Connect the signals
        self.dialog.generatorSelected.connect(self._on_generator_selected)
        self.dialog.injectStimuli.connect(self._on_inject_stimuli)
        self._generator_detail_model.valid.connect(self.dialog.onGeneratorValid)
        self.simulator = None

    def _on_generator_selected(self, generator):
        self._generator_detail_model.set_generator(str(generator))

    def _on_inject_stimuli(self, allCells):
        if not self._generator_detail_model.isValid():
            return

        parameters = self._generator_detail_model.get_generator_parameters()
        if parameters["model"] == "":
            return

        if not self.simulator:
            print("Error: Cannot inject stimulus, no simulator available")
            return

        if len(self.view.scene.highlightedNeurons) == 0:
            gids, dummy = sceneops.get_neurons(self.view.scene)
        else:
            gids = self.view.scene.highlightedNeurons

        if allCells:
            self.simulator.injectMultipleStimuli(parameters, gids)
        else:
            self.simulator.injectStimulus(parameters, gids)
        self._done()



