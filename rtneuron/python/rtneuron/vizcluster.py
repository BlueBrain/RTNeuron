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

import os as _os
import re as _re
import socket as _socket
import sys as _sys

def _use_virtualGL():
    is_ssh = 'SSH_CLIENT' in _os.environ
    is_vglconnect = is_ssh and ('VGL_DISPLAY' in _os.environ or
                                'VGL_CLIENT' in _os.environ)
    try:
        is_ssh_x = is_ssh and 'localhost:' in _os.environ['DISPLAY']
    except KeyError:
        is_ssh_x = False
    is_vnc = 'VNCDESKTOP' in _os.environ
    return (is_vglconnect or is_vnc or is_ssh_x) and \
        not '--eq-client' in _sys.argv

def _vizcluster_setup():

    if 'CUDA_VISIBLE_DEVICES' in _os.environ:
        # This is a workaround, the real problem is that the rendering clients
        # should be launched with srun instead of ssh. With ssh this
        # environmental variable is not setup by SLURM
        del _os.environ['CUDA_VISIBLE_DEVICES']

    if _use_virtualGL():
        # Running within a vglconnect connection established by vizsh
        sshIP = _os.environ['SSH_CLIENT'].split()
        if len(sshIP) == 0: # This shouldn't happen but checking it anyway
            return
        sshIP = sshIP[0]

        # The following code mimics the part of vglrun does that we need
        if 'VGL_CLIENT' not in _os.environ and 'RRCLIENT' not in _os.environ:
            vgl_client = sshIP + ":0.0"
            _os.environ['VGL_CLIENT'] = vgl_client
            print("[VGL] NOTICE: Automatically setting VGL_CLIENT environment" \
                  "variable to")
            print("[VGL]    %s, the IP address of your SSh client." %
                  vgl_client)

        # vglconnect -s exports VGL_CLIENT=localhost, this seems to deceive
        # VirtualGL into thinking that the most appropriate transport is using
        # Xlib. If an ssh tunnel is detected, jpeg compression is enforced.
        if '__VGL_SSHTUNNEL' in _os.environ:
            # A corner case is when we connect to the cluster using vizsh -4
            # to start a VNC server and then connect to it. In that case the
            # VNC desktop has __VGL_SSHTUNNEL present but jpeg compression
            # mustn't be used (that actually fails). If VNCDESKTOP is in the
            # environemnt we will assume that the final display is the VNC
            # display
            if 'VNCDESKTOP' not in _os.environ:
                _os.environ['VGL_COMPRESS'] = 'jpeg'

        # Loading the VGL interposing library must be done before rtneuron
        # is imported.
        # RTLD_GLOBAL ensures that resolution of GL symbols when rtneuron
        # is loaded will look first in the loaded library.
        import ctypes
        try:
            # The latest VirtualGL version has renamed this library
            ctypes.CDLL('libvglfaker.so', ctypes.RTLD_GLOBAL)
        except:
            ctypes.CDLL('librrfaker.so', ctypes.RTLD_GLOBAL)
        ctypes.CDLL('libdlfaker.so', ctypes.RTLD_GLOBAL)

if (_re.match('^bbplxviz[0-9]{2}.epfl.ch$', _socket.gethostname()) or
    _re.match('^bbpviz[0-9]{3}.cscs.ch$', _socket.gethostname()) or
    _re.match('^bbpviz[0-9]{3}.bbp.epfl.ch$', _socket.gethostname())):
    _vizcluster_setup()


