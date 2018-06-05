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

cli_options = """

###############################################################################
!!GROUP General options

!!OPTION -h --help
Displays the command line options help.
!!OPTION --gui
Launch the GUI. This option is not compatible with --eq-config. It can be
combined with --shell.

!!OPTION --demo [demo_name]
Starts the demo of the given name. If no name is provided it will start the
default demo (named projections). This demo shows a default circuit using
somas. When a soma is clicked, the anterograde pathways of the neuron are
shown, if the soma is clicked again, the retrograde pathways are shown.
The shell will also be started.

!!OPTION --demo help
Print out the list of available demos.

!!OPTION --shell
Start an interactive IPython shell after initialization (or a regular console
if IPython is not available). This option can be combined with --gui.

!!OPTION --version
Displays version information.

################################################################################
!!GROUP Data loading options

!!OPTION -b --blue-config <em>filename</em>
The simulation configuration file to use.

#!!OPTION --models
#A comma separated list of paths to files loadable by OSG to include an
#additional geometrical model in the scene (e.g, pial surface, striatum).
#After each model a series of transformations can be specified. The syntax
#is <span class="fixed">model_file[<em>transformations</em>]</span>, where
#transformations is a colon separated list of one of
#<span class="fixed">r@<em>x</em>,<em>y</em>,<em>z</em>,<em>angle</em></span>
#for rotations, <span class="fixed">t@<em>x</em>,<em>y</em>,<em>z</em></span>
#for translations and
#<span class="fixed">s@<em>x</em>,<em>y</em>,<em>z</em></span> for scaling
#(e.g. <span class="fixed">--models foo.osg[r@1,0,0,90:t@10,0,0:s@1,1,2]</span>)

!!OPTION --neuron -n <em>gid</em> [<em>mode</em> [<em>color</em>]]
The MVD file id (starting at 1) of a neuron to be loaded and the its display
options. Mode is one of <span class="fixed">soma|detailed|no_axon|skeleton|none</span>
and color is one o <span class="fixed">RGB[A]|random|all-random|by-branch [RGBA RGBA]|by-width[@attenuation] [RGB[A]]|by-distance-to-soma [RGB[A]]</span>.

!!OPTION --neurons <em>gid</em> <em>gid</em> [<em>mode</em> [<em>color</em>]]
The MVD file id range for a list of neurons to be loaded and their display
options. Mode is one of <span class="fixed">soma|detailed|no_axon|skeleton|none</span>
and color is an RGB[A] tuple or one of <span class="fixed">RGB[A]|random|all-random|by-branch [RGBA RGBA]|by-width[@attenuation] [RGB[A]]|by-distance-to-soma [RGB[A]]</span>.

!!OPTION --target <em>targetname</em> [<em>mode</em> [<em>color</em>]]
The name of a target to load and its display options. Mode is one of
<span class="fixed">soma|detailed|no_axon|skeleton|none</span> and color is
one of <span class="fixed">RGB[A]|random|all-random|by-branch [RGBA RGBA]|by-width[@attenuation] [RGB[A]]|by-distance-to-soma [RGB[A]]</span>.

################################################################################
!!GROUP Style options

!!OPTION --accurate-headlight
Accurate rendering of headlight lighting (required if head tracking is used).

!!OPTION --afferent-syn-color <em>red</em> <em>green</em> <em>blue</em> <em>alpha</em>
Default color for the glyph representing the afferent position of synapses.
Applied to synapse targets appearing after this option in the command line.

!!OPTION --background <em>red</em> <em>green</em> <em>blue</em> [<em>alpha</em>]
Background RGBA color. Alpha channel defaults to 1 if not given. Frame grabbing
considers the alpha channel of the background. If alpha equals to 1, the output
image will have no alpha channel.

!!OPTION --electron
Render neurons with a shader that mimics electron microscope imaging (SEM)

!!OPTION --efferent-syn-color <em>red</em> <em>green</em> <em>blue</em> <em>alpha</em>
Default color for the glyph representing the efferent position of synapses.
Applied to synapse targets appearing after this option in the command line.

!!OPTION --neuron-color <em>red</em> <em>green</em> <em>blue</em>
Default overall color for neurons when no simulation data is being displayed.
Applied to neuron targets appearing after this option in the command line.

!!OPTION --neuron-color all-random
Use a random opaque color per neuron by default.
Applied to neuron targets appearing after this option in the command line.

!!OPTION --neuron-color alpha&#8209;by&#8209;width[@attenuation] [R G B [A]]
Use variable color per vertex depending on the neuron branch width. The final
color is the interpolation between the optional RGBA base color and the color
((R + 0.5, G + 0.5, B + 0.5, 1) using 1 - exp(-width &times; 1/attenuation)
as the interpolation factor. If the base color is not given it defaults to (0,
0.5, 1, 1). If the optional attenuation parameter is not given it defaults to
2. As can be seen, the base color is the color applied to the thinnest
branches.

!!OPTION --neuron-color by-branch
Color axons with blue and dendrites and soma with red. This option overrides
any other coloring scheme for structural display of neurons (i.e. without
simulation data).
Applied to neuron targets appearing after this option in the command line.

!!OPTION --neuron-color random
Initialize the default color for neurons with a random color.
Applied to neuron targets appearing after this option in the command line.

!!OPTION --idle-AA
Enable idle anti-aliasing (16 samples per pixel by default)

!!OPTION --soma-radius <em>microns</em>
If present, specifies the default soma radius to use when the display mode
is soma and the morphology is not available (e.g. --no-morphologies was
present).

################################################################################
!!GROUP Simulation Playback

!!OPTION -r --report <em>name</em>
The report name to load data from. This name must be present in the Blue
Config file passed with -b.

!!OPTION --show-spikes
Deprecated, use --spikes instead.

!!OPTION -s --spikes [<em>spike_url</em>]
Load and show the spikes from the SpikePath pointed by the blue config.
A different spike URL can be provided as an optional argument.

!!OPTION --sim-step
Sets the simulation playback step. When no interface is shown this step is
measured in milliseconds.

!!OPTION -w --sim-window <em>start_time</em> <em>stop_time</em>
The time window of interest for simulation playback (times in ms). If
compartment or spikes reports are loaded, the final interval is the
intersection between this window and the reports' windows.

!!OPTION --spike-tail <em>length</em>
The length (in time) for spike trails along the axon and the decay time for
spikes in the soma.

################################################################################
!!GROUP Scene management options

!!OPTIONS --mesh-partition
Use meshes to compute the spatial partitions in sort-last configurations using
the spatial DB decomposition. Disabled if --no-meshes is provided.

!!OPTIONS --no-cuda
If compiled with CUDA support, disable the use of it. As of now, this applies
only to the culling algorithm

!!OPTION --no-cull
Do not perform fine grained culling of neuronal. The only test left is
a simple per neuron axis-aligned bounding-box intersection test.

!!OPTION --no-strips
Do not use triangle strips as render primitives.

!!OPTION --round-robin-DB-partition
Use heuristic round-robin partition of the neuron targets for DB decomposition
modes. Each instance will only load the meshes and morphologies it needs.

!!OPTION --spatial-DB-partition
Use a balanced k-d tree based on the morphological points of the neurons to
divide the scene in DB decompositions. Each instance will load permanently
only the meshes and morphologies intersected in its k-d tree leaf.

!!OPTION --unique-morphologies
Assume that neuron morphologies are unique for scene building. Off by default,
but RTNeuron will try to determine automatically if morphologies are unique
using the morphology names. Assuming unique morphologies enables lazy loading
of meshes.

!!OPTION --mesh-partition
Use meshes and not only morphologies to compute the distribution of geometry
complexity used to choose the cut plane positions of spatial DB.

################################################################################
!!GROUP Levels of detail

!!OPTION --clod
Use tubelets and pseudo cylinders levels of detail with a continuous transition
between both depending on branch thickness and distance to the camera. Meshes
are used for the cell cores unless --no-meshes is provided.

!!OPTION --no-lod
Do not generate different levels of detail. Only applicable to neurons
displayed with the <em>detailed</em> and <em>no_axon</em> modes.

!!OPTION --use-tubelets
Use tubelets instead of pseudo cylinder for some levels of detail.

!!OPTION --no-meshes
Do not load mesh data.

!!OPTION --no-morphologies
Do not load morphologies if possible. This applies to scene configurations
in which only neuron targets with display mode soma are used. Using any
other display mode causes this option to have no effect.

################################################################################
!!GROUP Network options

!!OPTION --zeroeq-http-server <em>[host[:port]]<em>
Enable the REST interface. All command line options are passed to the
zeroeq::http::Server object for its initialization.
!!OPTION --sync-camera <em>zeroeq-session<em>
Synchronize the camera with other applications.
!!OPTION --sync-selections <em>zeroeq-session<em>
Synchronize selections with other applications.
!!OPTIONS --track-selection
When --sync-selections is used this option tells RTNeuron to react to toggle
selection request events and track the selection state of neuron internally.
Off by default, this means that RTNeuron simply sends toggle events and waits
for an external selection event to update its selections.

################################################################################
!!GROUP GUI options

!!OPTION --tracker <em>device@host[:port][&sensorid=(any|###)]</em>
A VPRN device to be used as head tracker. Only available if
compiled with VRPN support.

!!OPTION --vrpn-manipulator <em>DeviceConfig</em>
Use VPRN devices as a camera manipulator. Only available if compiled with
VRPN support. The option argument specifies the device configuration as
either a JSON string or a path to a JSON file.

<ul>
<li><em>DeviceConfig</em>: Either a DevConfigStr or DevConfigFile.</li>
<li><em>DevConfigStr</em>: Valid JSON devices configuration string.</li>
<li><em>DevConfigFile</em>: Valid JSON devices configuration file.</li>
</ul>

A valid JSON device configuration has the following format:

<pre>
{<br>
  "type": <em>DeviceTypeStr</em>,<br>
  "tracker":<br>
  {<br>
    "url": <em>DeviceURL</em>,<br>
    "sensorid": <em>Id</em> | "any",<br>
    "position_axis": <em>AxisSpec</em> | <em>Matrix3f</em>,<br>
    "attitude_axis": <em>AxisSpec</em> | <em>Matrix3f</em><br>
  },<br>
  "button": <em>DeviceURL</em> | { "url" : <em>DeviceURL</em> }<br>
  "analog": <em>DeviceURL</em> | { "url" : <em>DeviceURL</em> }<br>
  "url": <em>DeviceURL</em><br>
} <br>
</pre>

Where:
<ul>
<li><em>DeviceTypeStr</em>: one of "GYRATION_MOUSE", "INTERSENSE_WAND",
 "SPACE_MOUSE" or "WIIMOTE" (upper or lowercase).</li>
<li><em>DeviceURL</em>: <em>Device@Host[:Port]</em></li>
<li><em>AxisSpec</em>: Maps XYZ tracker coordinates to world coordinates.
  Three character string comprised from 'XYZxyz' where upper-case indicates
  positive orientation, lower-case negative orientation.  Ex., <em>Xzy</em>
  maps (X,Y,X) to (X,-Z,-Y)</li>
<li><em>Device</em>: VRPN device name</li>
<li><em>Host</em>: Host name or address</li>
<li><em>Port</em>: Port number</li>
<li><em>Id</em>: Id number</li>
<li><em>Matrix3f</em>: [<em>number x 9</em>], a coordinate change matrix that
  transforms from tracker local coordinates to world coordinates.</li>
<li><em>*_Vector3f</em>: <em>[number, number, number]</em></li>
</ul>

The <em>url</em> field specifies the default URL to use for <em>tracker</em>,
<em>analog</em> and <em>button</em> fields if their URL declaration is missing.
Which fields are parsed depends on the device type. Gyration and Intersense
devices require the <em>tracker</em>, <em>button</em> and <em>analog</em>
fields; unless the <em>url</em> field is given, in which case the other ones
can be omitted and default values will be used for any other parameter.
The <em>sensorid</em> parameter defaults to "any" if not given. The parameters
<em>position_axis</em> and <em>attitude_axis</em> default to the identity
matrix.<br>

For SpaceMouse and Wiimote you may specify <em>analog</em> or
<em>url</em>, if both are omitted the device URL will default to
"device@localhost" for SpaceMouse and "WiiMote@localhost" for Wiimote (actually
these devices also require a button, but it's assumed to be the same URL than
the analog device).

Coherent devices could originate from different VRPN servers.

Examples:
<pre>
--vrpn-manipulator='{"type": "INTERSENSE_WAND", "url": "dtrackbody@z5-3-1:7702"}'
<br>
--vrpn-manipulator='{"type": "WIIMOTE", "analog" : { "url": "WiiMote@z5-3-1:7702"}}'
<br>
--vrpn-manipulator='{"type": "GYRATION_MOUSE",<br>
                     "tracker": {"url": "dtrackbody@z5-fe",<br>
                                 "sensorid": 1,<br>
                                 "position_axis": [1, 0, 0,<br>
                                                   0, 0,-1,<br>
                                                   0,-1, 0],<br>
                                 "attitude_axis": "XZY"},<br>
                     "button": { "url": "Device0@z5-3-1:7701" },<br>
                     "analog": "Device0@z5-3-1:7701"'}<br>
</pre>

!!OPTION --use-spacemouse <em>device@host</em>
A VRPN spacemouse device for camera manipulation. Only available if
compiled with VRPN support. Deprecated, use --vrpn-manipulation instead

!!OPTION --use-wiimote <em>device@host</em>
A VRPN Wiimote device for camera manipulation. Only available if
compiled with VRPN support and Wiimote support. Deprecated,
use --vrpn-manipulation instead.

!!OPTION --use-wiimote-pointer <em>device@host</em>
A VRPN Wiimote device to use as 3D pointer. Only available if
compiled with VRPN support and Wiimote support.

################################################################################
!!GROUP Transparency options

!!OPTION --alpha-blending
Enable alpha blending for transparency. By default, multi-layer
depth-peeling will be used.

!!OPTION --slices <em>value</em>
Number of slices of the depth partition used in multi-layer depth-peeling.

################################################################################
!!GROUP Frame capture

!!OPTION --file-format
File format to use for frame grabbing

!!OPTION --file-prefix
Name prefix to use for the files produced by frame grabbing. The prefix
can include and absolute path.

!!OPTION --frame-count <em>frames</em>
Render the given number of frames after the command line scene has been
created and then exit. This option has no effect if an interactive console
is started. In combination with --idle-AA the number of output frames may
be less than expected and no idle AA is performed.

!!OPTION --grab-frames
Use this option to write an image files per camera at 25fps, or per frame
if a camera path is given.

################################################################################
!!GROUP Camera paths

!!OPTION --path
Set an RTNeuron specific camera path. This camera path must be in the format
used by any other rendering engine from BlueHubCore. This option creates a
manipulator that takes precedence over any other (e.g. VRPN manipulators).

!!OPTION --path-fps <em>float</em>
Sets the frames per second to use in camera path playback.
If less or equal than 0 real time will be used.

################################################################################
!!GROUP Profiling

!!OPTION --profile <em>[option[,option]...]</em>
Enables profiling together with some additional profiling options. Valid
option are:
<ul>
<li>channels=<em>bool</em>: True to enable profiling and logging of per
channel draw/readback/assemble operations (False by default)</li>
<li>compositing=<em>bool</em>: False to disable frame compositing in
multichannel configurations (True by default)</li>
<li>gldraw=<em>bool</em>: False to disable GL draw calls for neuron models
(synapses and other objects are still drawn, True by default)</li>
<li>logfile=<em>filename</em>: File to write frame times in milliseconds
('./statistics.txt' by default')</li>
<li>readback=<em>bool</em>: False to disable frame readback and
compositing in multichannel configurations (True by default)</li>

</ul>
When profiling is enabled the rendering loop will run without waiting for
any event.

!!OPTION --proof-frame <em>filename</em>
If set, the first frame rendered after the scene has been loaded will be
written to the given filename.

################################################################################
!!GROUP Equalizer options

!!OPTION --eq-config <em>config_file|session</em>
The .eqc config file or hw_sd session name for choose the configuration.

!!OPTION --eq-layout <em>layout</em>
Configuration layout to use

!!OPTION --roi
Enables region of interest clipping for framebuffer readback of Equalizer
compounds.

!!OPTION --window-size <em>width</em> <em>height</em>
Window size hint to use if no Equalizer config is provided.
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
            print('<tr><th colspan="2">\n' + group + '</th></tr>')
        keys = list(options.keys())
        keys.sort()
        for option, i in zip(keys, range(0, len(keys))) :
            print('<tr class=' + ("even" if i % 2 else "odd") + '>' +
                  '<td class="option_name">', end="")
            text = options[option]
            option = str.replace(option, ' ', '&nbsp;')
            print(option, end="")
            print('</td><td>', end="")
            print(text, end="")
            print('</td></tr>')
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
table.options_table span
{
  font-family: monospace;
}
</style>
\endhtmlonly
""")

output = open('command_line_options.dox', 'w')
import sys
sys.stdout = output

print('/*! \page command_line_options Command line options')
print_style()

print("""
\htmlonly
""")
print_table(parse_table(cli_options))

##\ todo This has to be a footnote referenced from the table.
print("""
\endhtmlonly

All RGB and RGBA tuples are space separated numbers between 0 and 1.
""")

print('*/')
