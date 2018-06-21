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

from __future__ import print_function
table = """
!!GROUP Display options

#!!OPTION RTNEURON_DETAILED_SYNAPSES
#Use cone-shaped glyphs to depict synapses instead of low tessellation spheres.
#The cones are located at the post and pre synaptic positions reported by
#BBP-SDK and the orientation is computed so that the afferent and efferent
#terminals are facing one to each other.

!!OPTION RTNEURON_FASTER_PSEUDOCYLINDER_NORMALS
Alternative geometry dispatch and shader for pseudocylinders that runs slightly
faster but whose results seemingly differ more from meshes at far distances.

!!OPTION RTNEURON_FORCE_LINEAR_PATH_INTERPOLATION
By default, the camera position for camera paths is interpolated with splines.
With this option piece-wise linear interpolation is used instead.

!!OPTION RTNEURON_INDEXED_LIST
If triangle strips for the mesh models are available, convert them into
triangle soups (degenerate triangles are skipped).

!!OPTION RTNEURON_LOD_CONFIG_FILE <em>file_name</em>
A file to override the default LOD models and transition
distances. The LOD configuration file consists of a list of lines
<span class="fixed"><em>LOD_NAME start end</em></span>, Where <span
class="fixed"><em>LOD_NAME</em></span> can be
<span class="fixed">mesh</span>,
<span class="fixed">tubelets</span>,
<span class="fixed">high_detail_cylinders</span>,
<span class="fixed">low_detail_cylinders</span>,
<span class="fixed">detailed_soma</span> or
<span class="fixed">spherical_soma</span> and
<span class="fixed"><em>start</em></span>
and <span class="fixed"><em>end</em></span> are two numbers between 0
and 1. A LOD is used when an arbitrary function of the pixel area
covered by the bounding box of the neuron (restricted to the interval
[0, 1]) is within the given range. Nothing prevents several LODs from being
visible at the same time.

!!OPTION RTNEURON_ORIGINAL_INDEXED_LIST
Forces to use triangle soups for mesh models even if triangle strips are
available.

!!OPTION RTNEURON_PRELOAD_SIM_DATA
Only useful in standalone mode without a pipelined simulation reader.
This option forces the reader to load on memory all the simulation data for
the specified targets. Be aware of the time and memory requirements of
this option.

!!OPTION RTNEURON_SMOOTH_TUBELETS
Enables a different shader for tubelets models. With the technique used
the normals at tubelet joints are continuous, giving a better appearance.
This technique is also a bit slower than the default rendering mode.

!!OPTION RTNEURON_WIREFRAME_MORPHOLOGIES
Render neurolucida style representations with wireframe capsules instead of
filled objects.


!!GROUP Culling options

!!OPTION RTNEURON_MAX_CAPSULE_LENGTH
Initial maximum capsule length for the capsule bounding volumes in which
the culling algorithms subdivides the section. For any given section the
maximum value is increased in arithmetic progression when the total number
of capsules given the current parameters exceeds 32.

!!OPTION RTNEURON_MAX_OFFAXIS_SEPARATION
Initial maximum separation from a morphological point to the axis of the
capsule that is currently bounding that point. For any given section the
maximum value is increased in arithmetic progression when the total number
of capsules given the current parameters exceeds 32.

!!OPTION RTNEURON_SHARED_MEM_CULL_KERNEL
In CUDA-based culling, perform the GPU compositing of the visibility masks
of each portion of a section using using on-chip shared memory. This
option is for performance evaluation purpose of the culling algorithm and
there is no reason for using it.

!!OPTION RTNEURON_PER_SKELETON_READBACK
Issue a GPU-CPU readback operation per culled skeleton to retrieve its
visibility information instead of doing a single readback at the end of
the culling phase.


!!GROUP Debugging options

!!OPTION RTNEURON_COMPUTE_DEPTH_COMPLEXITY
Debugging option for the depth partitioning code. Not interesting for final
users.

!!OPTION RTNEURON_DEBUG_MAPPING
Debugging option that colors neurons according to their morphology mapping.
Each section get a color based on its id and the color blends to white as
the relative position of the vertices goes from 0 to 1. This option is
applied to all LODs to allow comparisons between the mesh and the underlying
morphology.

!!OPTION RTNEURON_PROFILE_DEPTH_PARTITION
Profiling option for the depth partitioning code. Not interesting for final
users.

!!OPTION RTNEURON_VIEWPORT


!!GROUP Miscelanea

!!OPTION RTNEURON_INVARIANT_IDLE_AA
If set, the calculation of the sample positions used for idle anti-aliasing
will be deterministic.

!!OPTION CAPTURE
This variable only works on Linux. When set to any value, realtime
frame capturing is performed at 30 fps. The frame stream is written to
a binary uncompressed file called stream.raw. The binary file can be
processed by a script called unpack_frames.py and the output fed into
mencoder (e.g <span class="fixed">unpack_frames.py stream.raw |
/usr/bin/mencoder - -demuxer rawvideo -rawvideo format=bgr24:w=1280:h=720
-vf flip</span>

!!OPTION OSG_GL_ERROR_CHECKING
Values can be ONCE_PER_ATTRIBUTE, ON, on, ONCE_PER_FRAME

!!OPTION RTNEURON_SET_MINIMUM_ALPHA_BLENDING_VIEWPORT
If set, the window pixel viewport will be used instead of the current channel
viewport as the size of textures allocated by the transparency algorithm.
The intent of this option is to avoid texture relocations in dynamic 2D
layouts, use with caution as it can lead to other performance problems.

"""

def parse_table(string) :
    current_option = None
    current_group = dict()
    table = [(None, current_group)]
    for line in string.split('\n') :
        if len(line) == 0 or line[0] == '#' :
            continue
        elif line[0:2] == '!!' :
            line = line[2:]
            sep = line.find(' ')
            key = line[:sep]
            text = line[sep:]
            if key == 'GROUP' :
                current_group = dict()
                table.append((text, current_group))
                current_option = None
            elif key == 'OPTION' :
                current_option = text
                current_group[text] = ''
        else :
            if current_option != None :
                current_group[current_option] += ' ' + line
    return table

def print_table(table) :
    print('<table class="options_table", table border="0", width="100%">')
    for group, options in table :
        if group != None :
            print('<tr><th colspan="2"> ' + group + ' </th></tr>')
        keys = list(options.keys())
        keys.sort()
        for option, i in zip(keys, range(0, len(keys))) :
            print('<tr class=' + ("even" if i % 2 else "odd") + '>' +
                  '<td class="option_name">', end="")
            text = options[option]
            option = str.replace(option, ' ', '&nbsp;')
            print(option)
            print('</td><td>', end="")
            print(text, end="")
            print('</td></tr>', end="")
    print('</table>')

def print_style() :
    print("""
\htmlonly

<style type="text/css">
table.options_table
{
  border-spacing: 0px;
}
table.options_table td.option_name
{
  font-family: monospace;
  vertical-align: text-top;
  width: 27em;
}
table.options_table tr.odd
{
  background-color: #F3F3FF;
  border:5px solid red;
}
table.options_table tr.even
{
  background-color: #EAEAFF;
}
table.options_table th
{
  text-align:left;
  background: lightgray;
  height: 2.5em;
}
table.options_table span.fixed
{
  font-family: monospace;
}
</style>
\endhtmlonly
""")
output = open('environmental_variables_table.dox', 'w')
import sys
sys.stdout = output

print('/*! \page environmental_variables Environmental variables')
print_style()
print('\htmlonly')

print_table(parse_table(table))

print("""
\endhtmlonly
*/
""")
