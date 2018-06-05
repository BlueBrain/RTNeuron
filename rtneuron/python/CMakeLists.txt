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

add_subdirectory(binding)

configure_file(rtneuron/__init__.py.in
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/__init__.py)
configure_file(rtneuron/nest.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/nest.py)
configure_file(rtneuron/options.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/options.py)
configure_file(rtneuron/path_setup.py.in
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/path_setup.py)
configure_file(rtneuron/vizcluster.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/vizcluster.py)

macro(COPY_AND_INSTALL_PACKAGE_DIRECTORY DIRECTORY)
    file(COPY rtneuron/${DIRECTORY}
      DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron)

    install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/${DIRECTORY}
      DESTINATION ${PYTHON_LIBRARY_SUFFIX}/rtneuron
      COMPONENT COMMON)
endmacro()

copy_and_install_package_directory(apps)
copy_and_install_package_directory(gui)
copy_and_install_package_directory(util)
copy_and_install_package_directory(sceneops)
if(RTNEURON_WITH_ZEROEQ)
  copy_and_install_package_directory(net)
endif()

install(FILES
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/__init__.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/nest.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/options.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/path_setup.py
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rtneuron/vizcluster.py
  DESTINATION ${PYTHON_LIBRARY_SUFFIX}/rtneuron)
