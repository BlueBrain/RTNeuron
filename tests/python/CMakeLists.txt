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
## FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along
## with this library; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Change this number when adding .py test files to force a
# CMake run: 1

include(CheckCSourceRuns)

if(NOT GPUTests_FOUND)
  return()
endif()

set(PYTHON_TEST_EXCLUDE image_compare.py renderer.py)
set(PYTHON_TEST_DEPENDENCIES rtneuron_python)
include(CommonPythonCTest)

configure_file(setup.py.in ${PYTHON_TEST_OUTPUT_PATH}/setup.py)
configure_file(image_compare.py ${PYTHON_TEST_OUTPUT_PATH}/image_compare.py)
configure_file(renderer.py ${PYTHON_TEST_OUTPUT_PATH}/renderer.py)

add_dependencies(${PROJECT_NAME}-nightlytests ${PROJECT_NAME}-pythontests)
if(COMMON_ENABLE_COVERAGE)
  add_coverage_targets(${PROJECT_NAME}-pythontests)
endif()
