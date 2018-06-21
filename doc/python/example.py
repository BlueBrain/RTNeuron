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

import os
import os.path

import string

from docutils import nodes
from docutils.parsers.rst import Directive, directives

class ExampleDirective(Directive):

    # this enables content in the directive
    has_content = False
    required_arguments = 1

    def run(self):
        name = self.arguments[0]
        image_path = name + '_files'
        env = self.state.document.settings.env

        files = os.listdir(image_path)
        uris = []
        for f in files:
            path = image_path + '/' + f
            if os.path.isfile(path) and f.endswith('.png'):
                uris.append(path)

        def image_number(x):
            major, minor = x[:-4].split('_')[-2:]
            return int(major), int(minor)
        uris.sort(key=image_number)

        if len(uris) == 0:
            return []

        images = []
        for uri in uris:
            options = {'scale': 20,
                       'uri': directives.uri(uri)}
            images.append(nodes.image(**options))

        section = nodes.section()
        section.attributes['ids'] = [name]
        title = nodes.title()
        link_text = name[0].upper()
        for c in name[1:]:
            if c == '_':
                link_text += ' '
            else:
                link_text += c

        reference = nodes.reference('', link_text, refuri=name + '.html')
        title.append(reference)
        section.append(title)

        for image in images:
            section.append(image)

        return [section]

def setup(app):

    app.add_directive('example', ExampleDirective)

    return {'version': '0.1'}
