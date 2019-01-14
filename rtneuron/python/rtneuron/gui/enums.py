# -*- coding: utf-8 -*-
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

import rtneuron as rtneuron
from PyQt5.QtCore import QObject, Q_ENUMS
from PyQt5.QtQml import qmlRegisterType
import six

__all__ = ['register']

class EnumMetaclass(type(QObject)):

    def __new__(cls, name, bases, attributes):

        name = name[3:]
        # Find the enum type in thr rtneuron module with the same name
        enum = getattr(rtneuron, name)
        # Create the nested class used by PyQt to enumerate the values
        values = type("Values", (), enum.names)
        attributes["Values"] = values
        Q_ENUMS(values)

        # Create the QML capable QObject
        enumType = type(QObject).__new__(cls, name, bases, attributes)

        # Finally we add the register function to the type. This can't be
        # done earlier because we need the type itself.
        @staticmethod
        def register():
            qmlRegisterType(enumType, name, 1, 0, name)
        enumType.register = register

        return enumType

class QmlRepresentationMode(six.with_metaclass(EnumMetaclass, QObject)):
    """A QML ready version of rtneuron.RepresentationMode"""

class QmlColorScheme(six.with_metaclass(EnumMetaclass, QObject)):
    """A QML ready version of rtneuron.ColorScheme"""

def register():
    """Register all rtneuron enums in the QML type system.

    This function must be called after creating the QApplication and before
    loading any QML component."""
    QmlColorScheme.register()
    QmlRepresentationMode.register()


