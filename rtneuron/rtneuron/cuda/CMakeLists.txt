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

include(CMakeDependentOption)

cmake_dependent_option(RTNEURON_USE_CUDA "Enable CUDA based features" ON
                       "CUDA_FOUND" OFF)

if(NOT RTNEURON_USE_CUDA OR NOT CUDA_FOUND)
  message(STATUS "Disable cuda, used ${RTNEURON_USE_CUDA}, found ${CUDA_FOUND}")
  return()
endif()
if(CUDA_VERSION_MAJOR VERSION_LESS 4)
  message(WARNING "CUDA 4.0 or later needed, CUDA support disabled")
  return()
endif()

add_definitions(-DUSE_CUDA)

set(RTNEURON_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${RTNEURON_PATH}/src/cuda
    PARENT_SCOPE)
cuda_include_directories(${RTNEURON_PATH}/lib/src/cuda)

# Find out the architectures supported by nvcc, starting with latest first
set(GENCODE_FLAGS)
foreach(ARCH 30 35 50 60)
  execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} --dryrun
                          -gencode arch=compute_${ARCH},code=sm_${ARCH}
                          ${CMAKE_CURRENT_SOURCE_DIR}/cull.cu
                  RESULT_VARIABLE RESULT
                  OUTPUT_VARIABLE dummy
                  ERROR_VARIABLE dummy)
  if(${RESULT} MATCHES 0)
    list(APPEND GENCODE_FLAGS "-gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endif()
endforeach()

# Strangely, only relative paths work correctly for .cu files with
# cuda_add_library
set(CUDA_SOURCES cull.cu fragments.cu CUDAContext.cpp)

# -D_FORCE_INLINES is a workaround for a very strange compilation error in
# Release builds with Ubuntu 16.04.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -DCUDA_API_PER_THREAD_DEFAULT_STREAM ${GENCODE_FLAGS}")
cuda_add_library(rtneuron_cuda SHARED ${CUDA_SOURCES})
target_link_libraries(rtneuron_cuda Lunchbox ${Boost_THREAD_LIBRARY})
list(APPEND RTNEURON_LIBRARIES rtneuron_cuda)
set(RTNEURON_LIBRARIES "${RTNEURON_LIBRARIES}" PARENT_SCOPE)

install(TARGETS rtneuron_cuda EXPORT ${PROJECT_NAME}Targets LIBRARY DESTINATION lib COMPONENT COMMON)

# Add a special target to clean nvcc generated files.
if(NOT TARGET clean_cuda_depends)
  cuda_build_clean_target()
endif()
