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

set(RTNEURON_CORE_PUBLIC_HEADERS_RELATIVE
  detail/Configurable.h
  detail/attributeMapTypeRegistration.h
  AttributeMap.h
  AttributeMap.ipp
  Camera.h
  CameraManipulator.h
  ColorMap.h
  RTNeuron.h
  Scene.h
  SimulationPlayer.h
  enums.h
  types.h
  View.h
  ui/CameraPath.h
  ui/CameraPathManipulator.h
  ui/Pointer.h
  ui/TrackballManipulator.h
  ui/VRPNManipulator.h
  ui/WiimotePointer.h
  sceneops/NeuronClipping.h
)

if(RTNEURON_WITH_ZEROEQ)
  list(APPEND RTNEURON_CORE_PUBLIC_HEADERS_RELATIVE
    net/CameraBroker.h
    net/RestInterface.h
    net/SceneEventBroker.h)
endif()

set(PUBLIC_HEADERS)
foreach(HEADER ${RTNEURON_CORE_PUBLIC_HEADERS_RELATIVE})
  list(APPEND PUBLIC_HEADERS ${PROJECT_SOURCE_DIR}/rtneuron/rtneuron/${HEADER})
endforeach()

