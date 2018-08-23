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

common_find_package(Doxygen)
if(NOT DOXYGEN_FOUND)
  return()
endif()

if (NOT PYTHONINTERP_FOUND)
  message(SEND_ERROR "Python interpreter not found. Python needed "
    "to generate doxygen tables")
  return()
endif()

# We don't need to copy this to the build directory because it's build
# independent
configure_file(${PROJECT_SOURCE_DIR}/doc/doxygen.cfg.in
               ${PROJECT_BINARY_DIR}/doc/doxygen.cfg)
set(DOXYGEN_CONFIG_FILE ${PROJECT_BINARY_DIR}/doc/doxygen.cfg)

configure_file(${PROJECT_SOURCE_DIR}/doc/reference_manual.dox.in
               ${PROJECT_BINARY_DIR}/doc/reference_manual.dox)

macro(create_tables OUTPUT_TARGETS)
  foreach(INPUT_FILE ${ARGN})
    string(REPLACE ".py" ".dox" OUTPUT_FILE ${INPUT_FILE})
    add_custom_command(
      OUTPUT ${PROJECT_BINARY_DIR}/doc/${OUTPUT_FILE}
      COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/${INPUT_FILE}
      DEPENDS ${PROJECT_SOURCE_DIR}/doc/${INPUT_FILE}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc/)
    string(REPLACE ".py" "" OUTPUT_TARGET ${INPUT_FILE})
    add_custom_target(${OUTPUT_TARGET}
      DEPENDS ${PROJECT_BINARY_DIR}/doc/${OUTPUT_FILE})
    set_target_properties(${OUTPUT_TARGET} PROPERTIES
      EXCLUDE_FROM_DEFAULT_BUILD ON FOLDER "doxygen")
    list(APPEND ${OUTPUT_TARGETS} ${OUTPUT_TARGET})
  endforeach()
endmacro()

create_tables(TABLES environmental_variables_table.py command_line_options.py)

# Release notes generation
configure_file(${PROJECT_SOURCE_DIR}/doc/release_notes.py.in
               ${PROJECT_BINARY_DIR}/doc/release_notes.py)
file(GLOB RELEASE_NOTE_FILES ${PROJECT_SOURCE_DIR}/doc/release_notes/*)
add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/doc/release_notes.dox
  COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/release_notes.py
  DEPENDS ${RELEASE_NOTE_FILES} ${PROJECT_BINARY_DIR}/doc/release_notes.py
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc/)
add_custom_target(release_notes
  DEPENDS ${PROJECT_BINARY_DIR}/doc/release_notes.dox)
set_target_properties(release_notes PROPERTIES
  EXCLUDE_FROM_DEFAULT_BUILD ON FOLDER "doxygen")

include(DoxygenRule)
add_dependencies(${PROJECT_NAME}-doxygen ${TABLES} release_notes)

add_subdirectory(python)
