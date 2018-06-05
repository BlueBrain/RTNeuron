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

try_run(TEST_WINDOW_CREATE_RESULT _dummy
  ${PROJECT_BINARY_DIR}/tests/test_window_create
  SOURCES ${PROJECT_SOURCE_DIR}/CMake/test_window_create.cpp
  LINK_LIBRARIES Qt5::Gui
)

if(NOT TEST_WINDOW_CREATE_RESULT MATCHES 0)
  # Since all tests require a GPU, we skip them completely
  return()
endif()

set(GPUTests_FOUND TRUE)