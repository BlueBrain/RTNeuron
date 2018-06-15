# -*- coding: utf8 -*-
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

import numpy as _np
import re as _re
import rtneuron as _rtneuron

def target_string_to_gids(targets, simulation):
    """Take a QString with a target description for an simulation and convert
    it to a numpy array of GIDs (u4).

    The input string can be a target name, a gid range (two ints separated by
    a hyphen), a single gid or a comma separated list of any of those.
    """
    def target(s):
        s = s.lstrip().rstrip()
        if _re.match('^[1-9][0-9]*$', s):
            key = int(s)
        elif _re.match('^[1-9][0-9]*-[1-9][0-9]*$', s):
            first, second = s.split('-')
            key = range(int(first), int(second) + 1)
        else:
            key = s
        return _rtneuron.util.key_to_gids(key, simulation)

    gids = _np.array([], dtype="u4")
    for s in targets.lstrip().rstrip().split(','):
        gids = _np.append(gids, target(s))

    return gids
