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

include(PythonDocstrings)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIR}
  ${PYTHON_INCLUDE_PATH}
  ${PYTHON_NUMPY_INCLUDE_DIR})

set(RTNEURON_PYTHON_SOURCES
  RawArray.cpp
  helpers.cpp
  submodules.cpp
  rtneuron_python.cpp
  sceneops/export.cpp
)

# This sources must have absolute path. Otherwise the script that creates
# the docstrings fails
set(RTNEURON_PYTHON_SOURCES_WITH_DOCSTRINGS
  AttributeMap.cpp
  Camera.cpp
  ColorMap.cpp
  CameraManipulators.cpp
  Pointer.cpp
  RTNeuron.cpp
  Scene.cpp
  SimulationPlayer.cpp
  View.cpp
  types.cpp
  sceneops/NeuronClipping.cpp
)

if(RTNEURON_WITH_ZEROEQ)
  list(APPEND RTNEURON_PYTHON_SOURCES
    net/export.cpp)
  list(APPEND RTNEURON_PYTHON_SOURCES_WITH_DOCSTRINGS
    net/CameraBroker.cpp
    net/SceneEventBroker.cpp
    net/RestInterface.cpp)
endif()

docstrings(RTNEURON_PYTHON_SOURCES_WITH_DOCSTRINGS PUBLIC_HEADERS
  ${PROJECT_BINARY_DIR}/include)

if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
  # Copying the fallback docstrings to the build directory.
  configure_file(default_docstrings.cpp
                 ${PROJECT_BINARY_DIR}/docstrings/cpp/init_docstrings.cpp)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
set_source_files_properties(
  RawArray.cpp PROPERTIES COMPILE_FLAGS "-Wno-error")
endif()

# Workaround for the GIL acquisition issue in shared_ptr destructors
add_library(shared_ptr_fix MODULE shared_ptr_deallocation.cpp)
common_compile_options(shared_ptr_fix)
target_link_libraries(shared_ptr_fix ${Boost_PYTHON${USE_BOOST_PYTHON_VERSION}_LIBRARY}
                                     ${PYTHON_LIBRARIES})
set_target_properties(shared_ptr_fix PROPERTIES
  PREFIX "" OUTPUT_NAME "_shared_ptr_fix"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron)
add_dependencies(RTNeuron-all shared_ptr_fix)

configure_file(../rtneuron/shared_ptr_fix.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/shared_ptr_fix.py)
install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/shared_ptr_fix.py
  DESTINATION ${PYTHON_LIBRARY_SUFFIX}/rtneuron)

# The library headers are put first in the include path to avoid conflicts
# with CMake generated files (and in particular RTNeuron.h vs rtneuron.h in
# case insensitive filesystems).
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/rtneuron/
                           ${PROJECT_BINARY_DIR}/docstrings/cpp)

add_library(rtneuron_python MODULE
  ${RTNEURON_PYTHON_SOURCES}
  ${RTNEURON_PYTHON_SOURCES_WITH_DOCSTRINGS}
  ${PROJECT_BINARY_DIR}/docstrings/cpp/docstrings.cpp)
if(DOXYGEN_FOUND AND SPHINX_FOUND)
  add_dependencies(rtneuron_python RTNeuron-docstrings)
endif()
add_dependencies(rtneuron_python shared_ptr_fix)

if(RTNEURON_WITH_ZEROEQ)
  set(RTNEURON_PYTHON_ZEROEQ_LIBRARY Lexis ZeroEQ)
endif()

target_link_libraries(rtneuron_python
  rtneuron_core BrainPythonHelpers ${RTNEURON_PYTHON_ZEROEQ_LIBRARY}
  ${Boost_PYTHON${USE_BOOST_PYTHON_VERSION}_LIBRARY}
  ${PYTHON_LIBRARIES})
set_target_properties(rtneuron_python PROPERTIES
  PREFIX "" OUTPUT_NAME "_rtneuron"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron)
add_dependencies(RTNeuron-all rtneuron_python)

set_property(GLOBAL APPEND PROPERTY RTNeuron_COVERAGE_INPUT_DIRS ${CMAKE_CURRENT_BINARY_DIR})

# Make sure that the Python wrappings of Brain and Monsteer are built when
# the dependencies are included as subprojects.
if(TARGET brain_python)
  add_dependencies(rtneuron_python brain_python)
endif()
if(TARGET monsteer_python)
  add_dependencies(rtneuron_python monsteer_python)
endif()

install(TARGETS rtneuron_python shared_ptr_fix
  LIBRARY DESTINATION ${PYTHON_LIBRARY_SUFFIX}/rtneuron)
