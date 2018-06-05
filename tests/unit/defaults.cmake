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

# The main CMake script in tests clears the include directories for each
# test target. As default value we take the include directories of the
# rtneuron_lib target
get_target_property(
  rtneuron_include_directories rtneuron_core INCLUDE_DIRECTORIES)

set(${NAME}_INCLUDE_DIRECTORIES "${rtneuron_include_directories}"
                                ${PROJECT_SOURCE_DIR}/rtneuron)

# By default all RTNeuron library dependencies are included here.
# RTNeuron classes are *not* included on purpose because mockup classes may
# cause link conflicts (this is not the case with ELF object resolution, but
# I don't know about other systems).
set(${NAME}_LINK_LIBRARIES
  ${RTNEURON_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
