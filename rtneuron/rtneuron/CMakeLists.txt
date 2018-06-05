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

# CUDA culling library
add_subdirectory(cuda)
if(RTNEURON_USE_CUDA)
  add_definitions(-DUSE_CUDA)
  include_directories(SYSTEM ${RTNEURON_CUDA_INCLUDE_DIRS})

  find_library(NVTX_LIBRARY nvToolsExt
    PATHS $ENV{CUDA_ROOT}/lib $ENV{CUDA_ROOT}/lib64)
endif()

include(files.cmake)

include_directories(${PROJECT_SOURCE_DIR}/rtneuron/rtneuron)
add_definitions(-DBOOST_ARCHIVE_SOURCE)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(util/splines/view.cpp
                              PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
endif()

# All these libraries are in already in a PRIVATE linkage section
set(RTNEURON_CORE_LINK_LIBRARIES ${RTNEURON_LIBRARIES} GL Qt5::Core)
if(LIBJPEGTURBO_FOUND)
  list(APPEND RTNEURON_CORE_LINK_LIBRARIES ${LibJpegTurbo_LIBRARIES})
endif()
if(RTNEURON_WITH_ZEROEQ)
  list(APPEND RTNEURON_CORE_LINK_LIBRARIES Lexis ZeroEQ ZeroEQHTTP)
endif()
if(OSG_GL3_AVAILABLE AND RTNEURON_USE_CUDA AND CUDA_FOUND AND NVTX_LIBRARY)
  list(APPEND RTNEURON_CORE_LINK_LIBRARIES ${NVTX_LIBRARY})
  set_source_files_properties(viewer/osgEq/MultiFragmentCompositor.cpp
                              PROPERTIES COMPILE_FLAGS -DUSE_NVTX)
endif()

set(RTNEURON_CORE_OMIT_LIBRARY_HEADER ON)
common_library(rtneuron_core)
set_target_properties(rtneuron_core PROPERTIES OUTPUT_NAME rtneuron)

# Compilation and installation for shaders
install(DIRECTORY render/shaders
        DESTINATION share/RTNeuron/${VERSION}
        COMPONENT COMMON)
