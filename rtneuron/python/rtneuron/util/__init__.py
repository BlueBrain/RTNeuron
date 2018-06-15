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
import random as _random
import re as _re

import rtneuron as _rtneuron

from . import camera

def label_to_gids(label, simulation):
    # This is a target name or regular expression
    try:
        label, fraction = label.split("%")
        fraction = float(fraction) / 100.0
    except:
        fraction = 1

    try:
        gids = simulation.gids(label)
    except Exception as e:
        # Trying as a regular expression
        prog = _re.compile(label)
        names = simulation.target_names()
        gids = _np.array((), dtype="u4")
        for name in names:
            match = prog.match(name)
            if match and match.group(0) == name:
                gids = _np.append(gids, simulation.gids(name))

        if len(gids) == 0:
            raise

    if fraction != 1:
        _np.random.shuffle(gids)
        gids = gids[0:int(round(len(gids) * fraction))]
        if len(gids) == 0:
            return None

    return gids

def key_to_gids(key, simulation):
    """Convert a target key to a GID array

    A key can be:
    - An integer
    - An str with a target name or regex
    - A numpy array of type u4, u8 or i4
    - An iterable of integers.
    """
    if type(key) == str:
        gids = label_to_gids(key, simulation)
    elif type(key) == int or type(key) == _np.uint32:
        # This is a single GID target
        gids = _np.array([key], dtype="u4")
    elif type(key) == _np.ndarray and key.dtype in ["u4, u8, i4"]:
        return key
    else:
        # Assume key is an iterable of something convertible to integers
        gids = _np.zeros((len(key)), dtype="u4")
        try:
            for i, c in enumerate(key):
                gids[i] = int(c)
        except:
            raise ValueError("Cannot convert key %s to a GID list." % key)

    # Now gids contains a numpy of GIDs or None if the target was not found
    return gids

def targets_to_gids(targets, simulation):
    """Return a numpy array with the gids of the targets given.
    Targets can be any object accepted by key_to_gids or an iterable of any of
    those."""

    if hasattr(targets, "__len__") and type(targets) is not str:
        try:
            return _np.array(targets, dtype="u4")
        except:
            gids = _np.array([], dtype="u4")
            for target in targets:
                gids = _np.append(gids, key_to_gids(target, simulation))
    else:
        gids = key_to_gids(targets, simulation)
    return gids


