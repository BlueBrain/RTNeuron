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

set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README" )

# Debian packaging dependencies
set(RTNEURON_PACKAGE_DEB_DEPENDS bbpsdk libopenscenegraph-dev
  libosgtransparency equalizer collage lunchbox brion qml-module-qtquick-layouts
  qml-module-qtquick-dialogs qml-module-qtquick-controls)

if(USE_PYTHON3)
  set(PYTHON_SUFFIX 3)
endif()
list(APPEND RTNEURON_PACKAGE_DEB_DEPENDS python${PYTHON_SUFFIX}-numpy
  python${PYTHON_SUFFIX}-dev python${PYTHON_SUFFIX}-decorator
  python${PYTHON_SUFFIX}-pyqt5 python${PYTHON_SUFFIX}-pyqt5.qtquick
  python${PYTHON_SUFFIX}-opengl python${PYTHON_SUFFIX}-six
  ipython${PYTHON_SUFFIX})
if(RTNEURON_USE_CUDA)
  list(APPEND RTNEURON_PACKAGE_DEB_DEPENDS nvidia-cuda-toolkit)
endif()

include(CommonCPack)
