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

import re as _re

from PyQt5 import QtGui as _Qt

from OpenGL import GLU as _GLU
from OpenGL import GL as _GL

_renderer = None
_app = None

def _get_renderer():
    global _renderer
    global _app
    if _renderer != None:
        return _renderer

    app = _Qt.QGuiApplication([])
    surface = _Qt.QOffscreenSurface()
    surface.create()
    context = _Qt.QOpenGLContext()
    assert(context.create())
    assert(context.makeCurrent(surface))
    renderer = _GL.glGetString(_GL.GL_RENDERER)
    context.doneCurrent()
    del context
    del surface
    _renderer = str(renderer)
    if "Tesla" in _renderer:
        # We cannot delete the app until the CUDA bug that makes
        # cudaGLGetDevices fail after a XCloseDisplay is solved. Not deleting
        # the app causing XCB to issue a warning *after* the function returns,
        # but I have no explanation for it.
        _app = app
    return _renderer

def get_GPU_families():
    renderer = _get_renderer()
    if _re.search('Quadro', renderer):
        return ["quadro"]
    elif _re.search('Tesla', renderer):
        return ["tesla", "quadro"]
    elif _re.search('GeForce', renderer):
        return ["geforce"]
    return ["geforce"]
