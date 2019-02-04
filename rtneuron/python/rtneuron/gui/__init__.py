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

import sys

from PyQt5 import QtWidgets, QtCore

from .gui import BaseGUI
from .loaderGUI import LoaderGUI
from .selectionHandler import SelectionHandler
from . import enums

import rtneuron as rtneuron

__all__ = ['create_qt_app', 'display_empty_scene_with_GUI']

_XInitThreads_done = False

def create_qt_app():
    """Creates a Qt application object to be used together with IPython.

    The application event loop mustn't be invoked directly, it will be called
    from the IPython shell instead.

    Return None when the GUI is not supported on the current platform or the
    Qt application object otherwise."""

    if not sys.platform.startswith('linux'):
        return None

    global _XInitThreads_done
    if not _XInitThreads_done:
        import ctypes
        x11 = ctypes.cdll.LoadLibrary('libX11.so.6')
        x11.XInitThreads()
        _XInitThreads_done = True

    application = QtWidgets.QApplication([])
    application.setAttribute(QtCore.Qt.AA_DontCreateNativeWidgetSiblings)

    enums.register()

    return application


def display_empty_scene_with_GUI(
        GUI, scene_attributes=rtneuron.AttributeMap(), *args, **kwargs):
    """Initializes an RTNeuron instance with a GUI overlay and an empty scene.

    The gui parameter is a class that must inherit from rtneuron.gui.BaseGUI.
    An object of this class will be instantiated to create the overlay what
    will appear on top of the 3D rendering.

    The basic sequence of operations is as follows:
    - The QApplication is created.
    - The GUI is instantiated.
    - The RTNeuron engine object is created and linked to the GUI.
    - An empty scene is displayed.

    The return value is the instance of the GUI class. The RTNeuron
    engine is set into the rtneuron.engine global variable.

    The *args and **kwargs arguments are passed to the GUI constructor.
    """

    qt_app = create_qt_app() # lgtm [py/unused-local-variable]
                             # We need the QtApplication object in a local
                             # variable until the scoped is finished
    gui = GUI(*args, **kwargs)

    rtneuron.display_empty_scene(
        scene_attributes=scene_attributes,
        opengl_share_context=gui.get_background_context())

    gui.connect_engine(rtneuron.engine)

    return gui
