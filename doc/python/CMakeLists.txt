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

if(NOT SPHINX_FOUND)
  return()
endif()

configure_file(${PROJECT_SOURCE_DIR}/doc/python/conf.py.in
               ${PROJECT_BINARY_DIR}/doc/python/conf.py)

file(COPY index.rst api.rst gallery.rst example.py
          getting_started.rst getting_started_files
          colorings_and_modes.rst colorings_and_modes_files
          meshes_and_models.rst meshes_and_models_files
          simulation_playback.rst simulation_playback_files
          synaptic_afferent_projections.rst synaptic_afferent_projections_files
     DESTINATION ${PROJECT_BINARY_DIR}/doc/python)

add_custom_target(${PROJECT_NAME}-python-doc-html
  ${PYTHON_EXECUTABLE} ${SPHINX_EXECUTABLE} -b html . ${PROJECT_BINARY_DIR}/doc/html/python
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc/python
  COMMENT "Generating Python documentation using Sphinx" VERBATIM)
set_target_properties(${PROJECT_NAME}-python-doc-html PROPERTIES
  EXCLUDE_FROM_DEFAULT_BUILD ON FOLDER "doxygen")

add_dependencies(${PROJECT_NAME}-python-doc-html rtneuron_python)
add_dependencies(${PROJECT_NAME}-doxygen ${PROJECT_NAME}-python-doc-html)
