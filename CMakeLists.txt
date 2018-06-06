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

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(RTNeuron VERSION 3.0.0)
set(RTNeuron_VERSION_ABI 33)

get_filename_component(RTNEURON_PATH ${PROJECT_SOURCE_DIR} ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake
                              ${PROJECT_SOURCE_DIR}/Pydoxine/CMake
                              ${CMAKE_SOURCE_DIR}/CMake/common)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/CMake/common/Common.cmake)
  message(FATAL_ERROR "CMake/common missing, run: git submodule update --init")
endif()

set(RTNEURON_DESCRIPTION
  "Scalable rendering tool for the visualization of Neuron simulation data.")
set(RTNEURON_MAINTAINER_NAME "Blue Brain Project")
set(RTNEURON_MAINTAINER_EMAIL "<bbp-open-source@googlegroups.com>")
set(RTNEURON_MAINTAINER_MAINTAINER "${RTNEURON_MAINTAINER_NAME} ${RTNEURON_MAINTAINER_EMAIL}")
set(RTNEURON_LICENSE GPLv3)
set(RTNEURON_DEB_DEPENDS ipython3 python3-numpy python3-dev python3-decorator
  python3-pyqt5 python3-pyqt5.qtquick python3-opengl qml-module-qtquick-controls
  qml-module-qtquick-layouts qml-module-qtquick-dialogs
  nvidia-cuda-toolkit nvidia-cuda-dev)

set(COMMON_PROJECT_DOMAIN ch.epfl.bluebrain)

option(RTNEURON_OPTIONAL_DEPS "Include optional dependencies" "OFF")

include(Common)

# External dependencies

common_find_package(Boost 1.54 REQUIRED COMPONENTS filesystem system serialization program_options date_time regex python${USE_BOOST_PYTHON_VERSION} unit_test_framework)
common_find_package(Brion REQUIRED)
common_find_package(BBPTestData)
common_find_package(Collage REQUIRED)
common_find_package(CUDA)
common_find_package(Equalizer REQUIRED)
common_find_package(Monsteer)
common_find_package(Lexis)
common_find_package(LibJpegTurbo 1.2.1)
common_find_package(Lunchbox REQUIRED)
common_find_package(NumPy REQUIRED)
if(NOT APPLE)
  common_find_package(NeuMesh)
endif()
common_find_package(OpenGL REQUIRED)
common_find_package(OpenMP)
common_find_package(OpenSceneGraph 3.2 SYSTEM REQUIRED COMPONENTS osgGA osgText osgUtil osgDB osgViewer)
common_find_package(OsgGL3)
common_find_package(osgTransparency REQUIRED)
common_find_package(PythonInterp REQUIRED)
common_find_package(PythonLibs REQUIRED)
common_find_package(Qt5Core REQUIRED)
common_find_package(Qt5OpenGL REQUIRED)
common_find_package(Qt5Gui REQUIRED)
common_find_package(Qt5Widgets REQUIRED)
common_find_package(Sphinx 1.3)
common_find_package(vmmlib REQUIRED)
common_find_package(VRPN 07.30)
common_find_package(Wiiuse 0.14)
common_find_package(ZeroEQ)
# Fake subproject to check if there's a GPU available. Checking it with
# common_find_package will print out the test result in the configuration log.
common_find_package(GPUTests)

common_find_package_post()

# Library dependency names for pkgconfig (for those libraries that provide it)
set(RTNEURON_DEPENDENT_LIBRARIES Brain Lunchbox vmmlib OpenSceneGraph)

# Suppress cppcheck warnings
list(APPEND CPPCHECK_EXTRA_ARGS
  --suppress=uselessAssignmentPtrArg:${PROJECT_SOURCE_DIR}/lib/rtneuron/AttributeMap.cpp
  --suppress=uselessAssignmentArg:${PROJECT_SOURCE_DIR}/lib/rtneuron/AttributeMap.cpp)

# Compile flags

list(APPEND RTNEURON_LIBRARIES
  PUBLIC Brain Lunchbox vmmlib ${OPENSCENEGRAPH_LIBRARIES}
  PRIVATE ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} osgTransparency Equalizer
          EqualizerAdmin
)

# We need to find Boost.Python in FindPackages, but we don't want the core
# libraries to depend on it
list(REMOVE_ITEM RTNEURON_LIBRARIES
  ${Boost_PYTHON_LIBRARY}
  ${Boost_PYTHON${USE_BOOST_PYTHON_VERSION}_LIBRARY})

add_definitions(-DEQ_SYSTEM_INCLUDES) # get Equalizer's GL(EW) headers


if(NEUMESH_FOUND)
  # For unknown reasons, the transient dependency to vtkCommon (declared by NeuMesh
  # target) is not exported, hence find it to resolve include paths.
  common_find_package(VTK COMPONENTS vtkCommon)
  list(APPEND RTNEURON_LIBRARIES NeuMesh)
endif()

if(TARGET Lexis AND TARGET ZeroEQ AND TARGET ZeroEQHTTP)
  set(RTNEURON_WITH_ZEROEQ ON)
endif()

if(VRPN_FOUND)
  list(APPEND RTNEURON_LIBRARIES ${VRPN_LIBRARIES} pthread)
endif()

add_subdirectory(rtneuron) # RTNeuron C++/Python library
add_subdirectory(app) # Python based application
add_subdirectory(doc)
add_subdirectory(tests)
# This must come after all the install commands
include(CPackConfig)
