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

# When OpenSceneGraph is available, either OSG_GL2_AVAILABLE or
# OSG_GL3_AVAILABLE are set to indicate the GL version for which it has been
# built.
#
# This script can be used to make common_find_package_post indicate if a GL3
# build of OpenSceneGraph has been found.

if(NOT OpenSceneGraph_FOUND)
  return()
endif()

# Detecting whether OSG is compiled with GL 3 support or not
file(READ ${OSG_INCLUDE_DIR}/osg/GL OPENSCENEGRAPH_CONFIG)
string(REGEX MATCH "define OSG_GL3_AVAILABLE" OSG_GL3_AVAILABLE
  ${OPENSCENEGRAPH_CONFIG})
string(REGEX MATCH "define OSG_GL2_AVAILABLE" OSG_GL2_AVAILABLE
  ${OPENSCENEGRAPH_CONFIG})

if(OSG_GL3_AVAILABLE)
  set(OsgGL3_FOUND TRUE)
endif()