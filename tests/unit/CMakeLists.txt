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

# Change this number when adding .cpp or .cmake test files to force a
# CMake run: 6

set(TEST_LIBRARIES ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

# If a test requires a mock up of any class which is the include
# directories, it may need to be disabled.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../rtneuron
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../rtneuron/rtneuron)

file(GLOB_RECURSE TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(FILE ${TEST_FILES})
  # a per test config file to fine tune its compilation.
  # This can be to create per-tests mock-ups if needed.
  string(REGEX REPLACE ".cpp" "" NAME ${FILE})
  string(REGEX REPLACE "[./]" "_" NAME ${NAME})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cmake)
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/defaults.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/defaults.cmake)
  endif()
endforeach()

if(NOT RTNEURON_ENABLE_NET)
  set(EXCLUDE_FROM_TESTS camera_broker.cpp scene_event_broker.cpp)
endif()

include(CommonCTest)
