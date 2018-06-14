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
import sys
from random import uniform
import argparse
import json

from rtneuron import AttributeMap, RTNeuron, ColorScheme, RepresentationMode, \
                     NeuronLOD, DataBasePartitioning, CameraPathManipulator

try:
    from rtneuron import VRPNManipulator
    _has_vrpn_support = True
except:
    _has_vrpn_support = False
try:
    import rtneuron.net
    _has_net_support = True
except:
    _has_net_support = False
try:
    from rtneuron.net import RestInterface
    _has_rest_support = True
except:
    _has_rest_support = False

DEFAULT_BLUECONFIG="/nfs4/bbp.epfl.ch/visualization/Circuits/KaustCircuit/BlueConfig"
DEFAULT_APP='projections'

class LODtype:
    NO_LOD, USE_CLOD, USE_TUBELETS, USE_CYLINDERS = range(4)

class KeepMetavar:
    pass

class HelpFormatter(argparse.HelpFormatter):
    """Custom help formatter class to use the metavar string verbatim in
    Actions that inherit from KeepMetavar"""
    def __init__(self, *args, **kwargs) :
        argparse.HelpFormatter.__init__(self, *args, **kwargs)

    def _format_args(self, action, default_metavar):
        if isinstance(action, KeepMetavar):
            # This is a custom action, keeping the metavar as is.
            return action.metavar
        return argparse.HelpFormatter._format_args(
            self, action, default_metavar)

class LODSwitchPoints:

    def _init_by_config_file(filename):
        lods = open(filename)
        tags = [name.lower() for name in NeuronLOD.names]
        # membrane_mesh is renamed to mesh for the LOD config file
        tags.remove("membrane_mesh")
        tags.append("mesh")

        ranges = dict()
        line_number = 0
        for line in lods:
            line_number += 1
            if len(line) == 0 or line[0] == '#':
                continue
            tokens = line.split()
            if len(tokens) != 3:
                raise ValueError("Parse error in line %d" % line_number)
            name, start, stop = tokens
            if name not in tags:
                raise ValueError("Invalid LOD type '" + name +
                                 "' in LOD config file")
            start = float(start)
            stop = float(stop)
            if start >= stop:
                print("Warning: Ignoring invalid LOD range (%g, %g) " \
                      "in LOD config file" % (start, stop))
                continue
            if start < 0.0 or stop > 1.0:
                print("Warning: Invalid LOD range (%g, %g) in LOD " \
                      "config file" % (start, stop))
                start = max(start, 0.0)
                stop = min(stop, 1.0)
                print(", clamping to (%g, %g)" % (start, stop))
            ranges[name] = (start, stop)

        return ranges

    lod_file_ranges = None
    try:
        if "RTNEURON_LOD_CONFIG_FILE" in os.environ:
            lod_file_name = os.environ["RTNEURON_LOD_CONFIG_FILE"]
            lod_file_ranges = _init_by_config_file(lod_file_name)
    except Exception as e:
        print("Parsing LOD config file: " + str(e))

    def __init__(self, lodtype, use_meshes):
        if LODSwitchPoints.lod_file_ranges:
            # The LOD config file is independent of command line options
            # and parsed only once, reusing it
            self.ranges = LODSwitchPoints.lod_file_ranges
        else:
            self._default_init(lodtype, use_meshes)

    def _default_init(self, lodtype, use_meshes):
        self.ranges = dict()
        full_range = (0, 1)
        low_range = (0, 0.3)
        mid_range = (0.3, 0.5)
        mid_top_range = (0.3, 1.0)
        top_range = (0.5, 1.0)

        if lodtype == LODtype.USE_CLOD:
            self.ranges["high_detail_cylinders"] = full_range
            self.ranges["tubelets"] = full_range
            if use_meshes:
                self.ranges["detailed_soma"] = full_range
            else:
                self.ranges["spherical_soma"] = full_range
        elif use_meshes:
            self.ranges["mesh"] = top_range
            self.ranges["low_detail_cylinders"] = low_range
            self.ranges["spherical_soma"] = low_range
            if lodtype == LODtype.USE_CYLINDERS:
                self.ranges["high_detail_cylinders"] = mid_range
            if lodtype == LODtype.USE_TUBELETS:
                self.ranges["tubelets"] = mid_range
            self.ranges["detailed_soma"] = mid_range
        else:
            self.ranges["low_detail_cylinders"] = low_range
            self.ranges["spherical_soma"] = full_range
            if lodtype == LODtype.USE_CYLINDERS:
                self.ranges["high_detail_cylinders"] = mid_top_range
            if lodtype == LODtype.USE_TUBELETS:
                self.ranges["tubelets"] = mid_top_range

def str_to_representation_mode(label):
    cli_name_to_representation_mode = {
        "soma": RepresentationMode.SOMA,
        "skeleton": RepresentationMode.SEGMENT_SKELETON,
        "detailed": RepresentationMode.WHOLE_NEURON,
        "no_axon": RepresentationMode.NO_AXON,
        "none": RepresentationMode.NO_DISPLAY}

    if label not in cli_name_to_representation_mode:
        raise ValueError("Unknown neuron representation mode '" +
                         label + "'. Valid values are: soma, skeleton," \
                         " detailed, no_axon or none.")
    return cli_name_to_representation_mode[label]

def parse_color_option(values, action, min_length = 3):
    color = list(map(float, values))
    length = len(color)
    if length < min_length or length > 4:
        raise argparse.ArgumentError(
            action, "Error parsing " + ("RGB[A]" if min_length == 3 else "RGBA")
                    + " color")
    if length == 3:
        color.append(1.0)
    return color

def parse_neuron_color_scheme(values, namespace, action):

    def valid_by_width(x) :
        x = x.split('@')
        if x[0] != "by-width" and x[0] != "alpha-by-width":
            return False
        if len(x) == 2 :
            try:
                float(x[1])
            except ValueError:
              return False
        if x[0] == "alpha-by-width":
            print("Warning: alpha-by-width is deprecated. Use by-width instead")
        return True

    try:
        color = (ColorScheme.SOLID, parse_color_option(values, action), None)
    except ValueError:
        if values[0] == "all-random":
            color = (ColorScheme.RANDOM, None, None)
            del values[0]

        elif values[0] == "by-branch":
            del values[0]
            color = (ColorScheme.BY_BRANCH_TYPE,
                     [1.0, 0.0, 0.0, 1.0], [0.0, 0.5, 1.0, 1.0])
            if len(values) != 0:
                color = (ColorScheme.BY_BRANCH_TYPE,
                         parse_color_option(values[0:4], action, 4),
                         parse_color_option(values[4:], action, 4))

        elif valid_by_width(values[0]): # by-width
            try :
                attenuation = float(values[0].split('@')[1])
            except :
                attenuation = 2.0
            del values[0]

            color = (ColorScheme.BY_WIDTH,
                     [0, 0.5, 1.0, 0.0], [0.5, 1.0, 1.0, 1.0], attenuation)
            if len(values) != 0:
                primary = parse_color_option(values, action, 4)
                secondary = [min(primary[0] + 0.5, 1.0),
                             min(primary[1] + 0.5, 1.0),
                             min(primary[2] + 0.5, 1.0), 1.0]
                color = (ColorScheme.BY_WIDTH,
                         primary, secondary, attenuation)

        elif (values[0] == "by-distance-to-soma" or
              values[0] == "alpha-by-distance-to-soma"):
            if (values[0] == "alpha-by-distance-to-soma"):
                print("Warning: alpha-by-distance-to-soma is deprecated. "
                      "Use by-distance-to-soma instead")
            del values[0]
            color = (ColorScheme.BY_DISTANCE_TO_SOMA,
                     [0.8, 0.8, 1.0, 1.0], [0.8, 0.8, 1.0, 0])
            if len(values):
                rgba = parse_color_option(values, action, 3)
                color = (ColorScheme.BY_DISTANCE_TO_SOMA,
                         rgba, rgba[:3] + [0])
        elif values[0] == "random":
            color = (ColorScheme.SOLID,
                     [uniform(0, 1), uniform(0, 1), uniform(0, 1), 1],
                     None)
        else:
            raise argparse.ArgumentError(
                action, "Error parsing color scheme. Valid values are: "
                "all-random, by-branch [R G B A R G B A], "
                "by-width [R G B [A]], R G B [A], "
                "by-distance-to-soma [R G B [A]]")
    return color

def parse_target_attributes(values, namespace, action):
    # Parsing optional representation modes
    mode = RepresentationMode.WHOLE_NEURON
    if len(values) > 0:
        try:
            mode = str_to_representation_mode(values[0])
        except ValueError as exc:
            raise argparse.ArgumentError(action, str(exc))
        del values[0]

    # Parsing optional color
    color = namespace.neuron_color
    if len(values) > 0:
        color = parse_neuron_color_scheme(values, namespace, action)

    # Creating the target attributes and storing the target specification
    attributes = AttributeMap()
    attributes.mode = mode
    attributes.color_scheme = color[0]
    if color[1] != None:
        attributes.color = color[1]
    if color[2] != None:
        attributes.secondary_color = color[2]
    if color[0] == ColorScheme.BY_WIDTH :
        attributes.extra = AttributeMap()
        attributes.extra.attenuation = color[3]

    return attributes

class ParseLaunchArgument(argparse.Action, KeepMetavar):
    def __call__(self, parser, namespace, values, option):
        # The position of the working directory may vary in the --co-launch
        # string. We look for the first string that contains the home
        # directory as prefix.
        home = os.environ['HOME']
        for value in values.split("#"):
            if value[:len(home)] == home:
                try:
                    os.chdir(value)
                except:
                    raise RuntimeError(
                        "Error: Could not set working directory to:" + value)
                break

class ParseRGBAColor(argparse.Action, KeepMetavar):
    def __init__(self, *args, **kwargs):
        argparse.Action.__init__(self, *args, **kwargs)
        self.metavar = "R G B [A]"
        self.type = float
        self.nargs = '*'

    def __call__(self, parser, namespace, values, option):
        setattr(namespace, self.dest, parse_color_option(values, self))

class ParseColorScheme(argparse.Action, KeepMetavar):
    def __init__(self, *args, **kwargs):
        argparse.Action.__init__(self, *args, **kwargs)
        self.metavar = "R G B [A] | random | all-random | by-branch [R G B A "\
                       "R G B A] |\n                 by-width[@attenuation] [R G B [A]]\n                 by-distance-to-soma [R G B [A]]"
        self.nargs = '*'

    def __call__(self, parser, namespace, values, option=None):
        scheme = parse_neuron_color_scheme(values, namespace, self)
        setattr(namespace, self.dest, scheme)

class ParseTargetArguments(argparse.Action, KeepMetavar):
    def __call__(self, parser, namespace, values, option):
        try:
            if option == "--neurons":
                target = range(int(values[0]), int(values[1]) + 1)
                del values[0:2]
            elif option == "--target":
                target = values[0]
                del values[0]
            else: # option == "-n" or option == "--neuron"
                target = int(values[0])
                del values[0]
        except TypeError:
            raise argparse.ArgumentError(self, "Invalid target specification")

        attributes = parse_target_attributes(values, namespace, self);
        # Adding the target info to the target list
        if getattr(namespace, "targets", None) is None:
            setattr(namespace, "targets", list())
        namespace.targets.append([target, attributes])

class ParseOldVRPNManipulator(argparse.Action, KeepMetavar):
    def __init__(self, *args, **kwargs):
        argparse.Action.__init__(self, *args, **kwargs)
        self.metavar = "device@host"
        self.type = str
        self.nargs = '?'
        self.dest = "vrpn_manipulator"

    def __call__(self, parser, namespace, value, option):
        attributes = AttributeMap()
        if option == "--use-wiimote":
            attributes.analog = "WiiMote0@localhost" if not value else value
            attributes.type = "wiimote"
        else:  # opt == "--use-spacemouse":
            attributes.analog = "device@localhost" if not value else value
            attributes.type = "space_mouse"

        setattr(namespace, self.dest, attributes)

class ParseVRPNManipulator(argparse.Action, KeepMetavar):
    def __init__(self, *args, **kwargs):
        argparse.Action.__init__(self, *args, **kwargs)
        self.metavar = "filename or config string"
        self.type = str

    def __call__(self, parser, namespace, value, option):
        def dict_to_attribute_map(val):
            am = AttributeMap()

            for k, v in val.iteritems():
                # handle python 2.X; json always encodes unicode
                k = k if (type(k) == str) else k.encode('ascii')

                if type(v) == unicode:
                    v = v.encode('ascii')
                elif type(v) == dict:
                    v = dict_to_attribute_map(v)

                setattr(am, k, v)

            return am

        try:
            if value[0] == '{': # assume JSON string
                data = json.loads(value)
            else:
                with open(value, 'r') as f:
                    data = json.load(f)
            attributes = dict_to_attribute_map(data)
            setattr(namespace, self.dest, attributes)
        except Exception as e:
            raise argparse.ArgumentError(
                self, "Invalid JSON string or file given: " + str(e))

class ParseProfilingOptions(argparse.Action):
    def __call__(self, parser, namespace, values, option):
        def to_bool(x):
            x = x.lower()
            if x in ['true', 'on', '1', 'yes']:
                return True
            elif x in ['false', 'off', '0', 'no']:
                return False
            else:
                raise ValueError()

        options = AttributeMap()
        defaults = {'logfile': (str, 'statistics.txt'),
                    'compositing': (bool, True),
                    'gldraw': (bool, True),
                    'readback': (bool, True),
                    'channels': (bool, False),
                    'dataloading': (bool, False)
                    }
        # Setting defaults if this is the first occurence of --profile
        if not getattr(namespace, self.dest):
            for key, (type, value) in defaults.items():
                setattr(options, key, value)
        # Parsing the option string
        if values:
            keyvalues = [x.split('=', 1) for x in values.split(',')]
            for keyvalue in keyvalues:
                if len(keyvalue) == 1:
                    print("Missing value for profiling option:" +
                          str(keyvalue[0]))
                    continue
                key, value = keyvalue
                if key not in defaults:
                    print("Unknown profiling option:" + str(key))
                    continue
                try:
                    type = defaults[key][0]
                    if type == bool:
                        value = to_bool(value)
                    else:
                        value = type(value)
                    setattr(options, key, value)
                except ValueError:
                    print("Invalid value for profiling option:" + str(key))

        # Merging current options with new ones, this is handy for command
        # lines that are generated automatically (e.g. benchmarking)
        try:
            current = getattr(namespace, self.dest)
            for key in dir(options):
                setattr(current, key, getattr(options, key))
        except Exception:
            setattr(namespace, self.dest, options)

# Declares and parses the command line options
def command_line_options(argv):
    args_parser = argparse.ArgumentParser(
        usage="""rtneuron [-h] [-v] [--shell] [--app app_name]
                       [-c filename] [--target name [neuron_mode [color]]]
                       [-n gid [neuron_mode [color]]]
                       [--neurons start_gid end_gid [neuron_mode [color]]]
                       [options...]""",
        description="RTNeuron command line application",
        epilog="For further help please check the user guide available at: https://bbp.epfl.ch/documentation/code/RTNeuron-${VERSION_MAJOR}.${VERSION_MINOR}/index.html",
        formatter_class=HelpFormatter)

    # Necessary default values
    args_parser.set_defaults(targets=[],
                             # Default assuming no REST support
                             rest=None,
                             # Defaults assuming no net support
                             sync_camera=None,
                             sync_selections=None,
                             # Defaults assuming no VRPN support
                             vrpn_manipulator=None,
                             use_wiimote_pointer=None)

    # General options
    args_parser.add_argument(
        "-v", "--version", action="version", version=RTNeuron.versionString,
        help="Show version and exit")
    args_parser.add_argument(
        "--demo", type=str, nargs='?', default=None, const=DEFAULT_APP,
        help=argparse.SUPPRESS)
    args_parser.add_argument(
        "--app", type=str, nargs='?', default=None, const=DEFAULT_APP,
        metavar="[app_name]",
         help="Starts the app of the given name. If no name is provided it" \
        " will start the default app (named projections). The shell will" \
        " also be started. Use --app help to list all avaiable apps.")
    args_parser.add_argument(
        "--gui", action = "store_true", default = False,
         help = "Launch the GUI. This option is not compatible "
                "with --eq-config")
    args_parser.add_argument(
        "--shell", action="store_true", default=False,
         help="Start an interactive IPython shell after initialization (or" \
        " a regular console if IPython is not available).")

    # Data loading options
    group = args_parser.add_argument_group("Data loading options")
    group.add_argument(
        "-b", "--blue-config", type=str, default=None, dest="config",
        metavar="filename", help="Deprecated, use --config instead."
        "config to load.")
    group.add_argument(
        "-c", "--config", type=str, default=None,
        metavar="filename", help="Blue config, SONATA simulation or circuit "
        "config to load.")
    group.add_argument(
        "--target", action=ParseTargetArguments,
        nargs='+', metavar="name [neuron_mode [color]]",
        help="Specifies the name of a target to load. See --neuron-color " \
        "for possible color specifications")
    group.add_argument(
        "-n", "--neuron", action=ParseTargetArguments,
        nargs='+', metavar="gid [neuron_mode [color]]",
        help="GID of a neuron to load into the scene. See --neuron-color " \
        "for possible color specifications")
    group.add_argument(
        "--neurons", action=ParseTargetArguments,
        nargs='+', metavar="start_gid end_gid [neuron_mode [color]]",
        help="Range of GIDs of neurons to load. See --neuron-color for" \
        "possible color specifications")

    # Rendering style options
    group = args_parser.add_argument_group("Style options")
    group.add_argument(
        "--background", action=ParseRGBAColor, default=[0.1, 0.15, 0.2, 1.0],
        help="Background color. Alpha channel defaults to 1 if not given. If"
        " the alpha channel is equal to 1, frame grabbing also captures"
        " the background.")
    group.add_argument(
        "--color-map", type=str, default=None, metavar="filename",
        help="File name of the color map to be used for simulation mapping")
    group.add_argument(
        "--neuron-color", action=ParseColorScheme,
        default=(ColorScheme.SOLID, [0.8, 0.85, 0.9, 1.0], None),
        help="The default color to use for neurons.")
    group.add_argument(
        "--efferent-syn-color", action=ParseRGBAColor,
        default=[0.5, 0.2, 1.0, 1.0],
        help="Default overall color for efferent locations of synapses.")
    group.add_argument(
        "--afferent-syn-color", action=ParseRGBAColor,
        default=[1, 0.6, 0.2, 1.0],
        help="Default overall color for afferent locations of synapses.")
    group.add_argument(
        "--soma-radius", type=float, default=9, metavar="float",
        help="If present, specifies the default soma radius to be used if" \
        " morphologies are not loaded.")
    group.add_argument(
        "--electron", action="store_true", default=False,
        help="Render with electron microscope imaging appearance.")
    group.add_argument(
        "--accurate-headlight", action="store_true", default=False,
        help="Accurate rendering of headlight lighting (required for CAVEs).")
    group.add_argument(
        "--idle-AA", default=True, help=argparse.SUPPRESS)

    # Transparency style options
    group=args_parser.add_argument_group("Transparency options")
    group.add_argument(
        "--alpha-blending", type=str, nargs='?',
        const="auto", default=None, metavar="algorithm",
        help="Enable alpha blending (the algorithm can be chosen with the"
        " optional parameter.")
    group.add_argument(
        "--alpha-aware", action="store_const", default=0.0, const=0.99,
        dest="alpha_threshold",
        help="Enable alpha based optimizations on alpha blending algorithms.")
    group.add_argument(
        "--slices", type=int, default=4, metavar="int",
        help="Number of slices to use for multi-layer depth peeling")
    args_parser.add_argument_group(group)

    # Simulation playback options
    group = args_parser.add_argument_group("Simulation playback options")
    group.add_argument(
        "-r", "--report", metavar="name", type=str, default=None,
        help="The report name to load data from")
    group.add_argument(
        "--show-spikes", action="store_true", default=None, dest="spikes",
        help="Load and show the spikes from the report pointed by the"
        " blue config. Deprecated, use --spikes instead")
    group.add_argument(
        "-s","--spikes", type=str, nargs='?',
        const=True, default=None, metavar="spike_file",
        help="Load and show spikes from a spikes report. If no spike file is"
        " given, the spike file is taken from the SpikesPath in the blue"
        " config.")
    group.add_argument(
        "--sim-step", metavar="milliseconds", type=float, default=None,
        help="Step between simulation frames during playback")
    group.add_argument(
        "-w", "--sim-window", metavar="begin end", type=float, nargs=2,
        default=None, help="Simulation begin and end times in milliseconds"
        " for simulation playback.")
    group.add_argument(
        "--spike-tail", metavar="ms", type=float, default=None,
        help="Length of spike tails in milliseconds.")

    # Scene management options
    if "${OSGGL3_FOUND}" == "TRUE":
        args_parser.set_defaults(partitioning=DataBasePartitioning.ROUND_ROBIN)
    else:
        args_parser.set_defaults(partitioning=DataBasePartitioning.SPATIAL)
    group = args_parser.add_argument_group("Scene management options")
    group.add_argument(
        "--mesh-partition", action="store_true", default=False,
        help="Use meshes to compute the spatial partitions in sort-last" \
        " configurations using the spatial DB decomposition. Disabled " \
        " if --no-meshes is provided.")
    group.add_argument(
        "--no-cuda", action="store_false", default=True, dest="use_cuda",
        help="If compiled with CUDA support, disable CUDA usage.")
    group.add_argument(
        "--no-cull", action="store_true", default=False,
        help="Do not perform fine grained culling of neuronal branches.")
    group.add_argument(
        "--no-strips", action="store_false", default=True,
        dest="use_strips", help="Do not use triangle strips as rendering" \
        "primitives.")
    group.add_argument(
        "--octree-depth", type=int, default=5, metavar="int",
        help="Maximum node depth for the morphological " \
        "skeleton octree. If must be a positive value.")
    group.add_argument(
        "--use-octree", action="store_true", default=False,
        help="Use a capsule octree for scene " \
        "management. This can be either combined with CUDA culling or not.")
    group.add_argument(
        "--round-robin-DB-partition", action="store_const",
        dest="partitioning", const=DataBasePartitioning.ROUND_ROBIN,
        help="Use heuristic round-robin partition of the neuron targets " \
        "for DB decomposition modes. Each instance will only load the meshes " \
        "and morphologies it needs.")
    group.add_argument(
        "--spatial-DB-partition", action="store_const",
        dest="partitioning", const=DataBasePartitioning.SPATIAL,
        help="Use a balanced k-d tree based on the morphological points of " \
        "the neurons to divide the scene in DB decompositions. Each instance " \
        "will load permanently only the meshes and morphologies intersected " \
        "in its k-d tree leaf.")
    group.add_argument(
        "--unique-morphologies", action="store_true", default=False,
        help="Assume that neuron morphologies are unique for scene " \
        " building. Off by default, but RTNeuron will try to determine " \
        "automatically if morphologies are unique using the morphology " \
        "names. Assuming unique morphologies enables lazy loading of meshes.")

    # Networking options
    if _has_net_support:
        group = args_parser.add_argument_group("Network")
        if _has_rest_support:
            # In this option the parameter is not really used because the whole
            # command line is passed to the RestInterface object.
            group.add_argument(
                "--zeroeq-http-server", type=str, metavar="host[:port]",
                nargs='?', dest="rest", default=None, const=True,
                help="Enable the REST interface.")
        group.add_argument(
            "--sync-camera", type=str, default=None, const=True, nargs="?",
            metavar="zeroeq-session",
            help="Synchronize the camera with other applications.")
        group.add_argument(
            "--sync-selections", type=str, default=None, const=True, nargs="?",
            metavar="zeroeq-session",
            help="Synchronize selections with other applications.")
        group.add_argument(
            "--track-selections", action="store_true", default=False,
            help="Keep track of neuron selection state when synching "
            "selections with other applications.")

    # LOD options
    group = args_parser.add_argument_group("Levels of detail")
    group.add_argument(
        "--no-lod", action="store_false", default=True, dest="use_lod",
        help="Don't generate different levels of detail.")
    group.add_argument(
        "--clod", action="store_true", default=False, dest="use_clod",
        help="Enable continuous levels of detail")
    group.add_argument(
        "--use-tubelets", action="store_true", default=False,
        dest="use_tubelets", help="Use tubelets instead of pseudo " \
        "cylinder for some levels of detail")
    group.add_argument(
        "--no-meshes", action="store_false", default=True,
        dest="use_meshes", help="Do not load meshes for neurons")

    # GUI options
    group = args_parser.add_argument_group("GUI options")
    if _has_vrpn_support:
        group.add_argument(
            "--vrpn-manipulator", action=ParseVRPNManipulator, default=None,
            help="Use generic VRPN devices for camera manipulation. "
            "See online reference for details.")
        group.add_argument(
            "--use-spacemouse", action=ParseOldVRPNManipulator, default=None,
            help="Deprecated, use --vrpn-manipulator instead.")
        group.add_argument(
            "--use-wiimote", action=ParseOldVRPNManipulator, default=None,
            help="Deprecated, use --vrpn-manipulator instead.")
        group.add_argument(
            "--use-wiimote-pointer", type=str, nargs='?',
            default=None, const="WiiMote0@localhost",
            metavar="deviceName@VRPNhost", help="Use Wiimote for 3D pointer.")
        group.add_argument(
            "--tracker", type=str, default=None,
            metavar="deviceName@VRPNHost", help="Head tracker device")

    # Recording options
    group = args_parser.add_argument_group("Recording options")
    group.add_argument(
        "--grab-frames", action="store_true", default=False,
        help="Start movie recording after data loading")
    group.add_argument(
        "--file-prefix", type=str, default="frame", metavar="prefix",
        help="Name prefix to use for the files produced by frame grabbing")
    group.add_argument(
        "--file-format", type=str, default="png", metavar="extension",
        help="File format to use for frame grabbing")
    group.add_argument(
        "--frame-count", type=int, default=0, metavar="frames",
        help="Render the given number of frames after the command line" \
            " scene has been created and then exit. This option has no " \
            "effect if an interactive console is started.")

    # Camera path options
    group = args_parser.add_argument_group("Camera path options")
    group.add_argument(
        "--path", type=str, default=None, dest="camera_path",
        metavar="path_file", help="Camera path to load. This option "
        "creates a manipulator that takes precedence over any other.")
    group.add_argument(
        "--path-fps", type=float, default=0.0, dest="camera_path_fps",
        metavar="milliseconds", help="Frames per second to use in camera" \
        " path playback. If less or equal to 0 real time will be used.")

    # Profiling options
    group = args_parser.add_argument_group("Profiling options")
    group.add_argument(
        "--profile", action=ParseProfilingOptions, nargs='?', default=None,
        metavar="option=value[,option=value]",
        help="Enable capturing of profiling data. See the online reference" \
            " for additional information.")
    group.add_argument(
        "--proof-frame", type=str, default=None, metavar="filename",
        help="Dump the first frame after loading to disk using the file" \
            "name given")

    # Equalizer options
    group=args_parser.add_argument_group("Equalizer options")
    group.add_argument(
        "--roi", action="store_true", default=False, dest="use_roi",
        help="Enables region of interest clipping for framebuffer " \
        "readback of Equalizer compounds.")
    group.add_argument(
        "--window-size", type=int, nargs=2, default=None,
        metavar="width height",
        help="Window size hint to use if no Equalizer config is provided")

    # defined so python doesn't complain about unrecognized options
    args_parser.add_argument("--eq-config", help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-server", help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-config-flags", help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-config-prefixes", help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-layout", help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-client", nargs='?',
                             help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-listen", help=argparse.SUPPRESS)
    args_parser.add_argument("--eq-logfile", help=argparse.SUPPRESS)
    args_parser.add_argument("--lb-logfile", help=argparse.SUPPRESS)
    args_parser.add_argument("--co-globals", help=argparse.SUPPRESS)
    # We need to parse the contents of --co-launch to set the working directory
    # because we can't wait for Equalizer to do it.
    args_parser.add_argument("--co-launch", action=ParseLaunchArgument,
                             help=argparse.SUPPRESS)

    # Parsing command line options
    try:
        if '--eq-client' in argv:
            argv.remove('--') # Removing the first -- added by Equalizer

        options = args_parser.parse_args(argv)

        if '--eq-client' in argv:
            options.gui = False
        elif "--eq-config" in sys.argv:
            if options.gui:
                options.gui = False
                print("GUI is not supported for custom Equalizer"
                      " configurations")

        if options.no_cull:
            options.use_cuda = False
            options.use_octree = False

        # In preparation for the deprecation of --no-lod --clod and
        # --use-tubelets to be replaced with a single command line option
        if not options.use_lod:
            options.lodtype = LODtype.NO_LOD
        elif options.use_clod:
            options.lodtype = LODtype.USE_CLOD
        elif options.use_tubelets:
            options.lodtype = LODtype.USE_TUBELETS
        else:
            options.lodtype = LODtype.USE_CYLINDERS
        del(options.use_lod)
        del(options.use_tubelets)
        del(options.use_clod)

        return options
    except Exception as exc:
        print(exc)
        exit(0)

def choose_camera_manipulator(options):
    """Return the camera manipulator to use according to the options parsed
    from the command line.

    May throw if an error occurs creating the camera manipulator."""
    # The camera path manipulator takes precedence.
    if options.camera_path:
        manipulator = CameraPathManipulator()
        try:
            manipulator.load(options.camera_path)
            if options.camera_path_fps > 0:
                manipulator.frameDelta = 1.0 / options.camera_path_fps * 1000.0
            return manipulator
        except Exception as e:
            print("Warning: Creating camera manipulator:" + str(e))
            return None

    if options.vrpn_manipulator:
        device_name = options.vrpn_manipulator.type.upper()
        try:
            device_type = VRPNManipulator.DeviceType.names[device_name]
        except KeyError:
            print("Warning: unknown VRPN manipulator type " + device_name)
            return None

        try:
            return VRPNManipulator(device_type, options.vrpn_manipulator)
        except Exception as e:
            print("Warning: Creating VRPN manipulator: " + str(e))
            return None

    return None

# Creates the default scene attribute map based on the command line options
# parsed.
def create_scene_attributes(options):
    attr = AttributeMap()

    attr.use_cuda = options.use_cuda
    if options.use_octree:
        octree = AttributeMap()
        octree.max_depth = options.octree_depth
        attr.octree = octree

    primitive_options = AttributeMap()
    primitive_options.use_strips = options.use_strips

    if options.lodtype != LODtype.NO_LOD:
        neurons = AttributeMap()
        ranges = LODSwitchPoints(options.lodtype, options.use_meshes).ranges
        for key in ranges:
            neurons.__setattr__(key, ranges[key])
        if options.lodtype == LODtype.USE_CLOD:
            neurons.clod = True
        attr.lod = AttributeMap()
        attr.lod.neurons = neurons

    attr.use_meshes = options.use_meshes
    attr.unique_morphologies = options.unique_morphologies

    attr.primitive_options = primitive_options

    attr.partitioning = options.partitioning
    attr.mesh_based_partition = options.mesh_partition

    # Scene style attributes
    attr.em_shading = options.electron
    attr.accurate_headlight = options.accurate_headlight

    # While shader composition is not used, the alpha blending technique
    # is part of the scene style and not the view.
    if options.alpha_blending != None:
        alpha_blending = AttributeMap()
        alpha_blending.mode = options.alpha_blending
        if (alpha_blending.mode == "multilayer_depth_peeling" and
            options.slices == 0):
            alpha_blending.mode = "depth_peeling"
        if options.slices < 0:
            raise ValueError("Invalid number of slices for multi-layer "\
                             "depth peeling: %d" % options.slices)
        alpha_blending.slices = options.slices
        if options.alpha_threshold != 0.0:
            # This is only used by the fragment linked list algorithm for
            # moment.
            alpha_blending.alpha_cutoff_threshold = options.alpha_threshold
        attr.alpha_blending = alpha_blending

    return attr

def create_rtneuron_engine_attributes(options):
    attr = AttributeMap()
    if _has_vrpn_support and options.tracker:
        attr.tracker = options.tracker
    if options.neuron_color[1]:
        attr.neuron_color = options.neuron_color[1]
    else:
        attr.neuron_color = "all-random"
    attr.afferent_syn_color = options.afferent_syn_color
    attr.efferent_syn_color = options.efferent_syn_color
    attr.soma_radius = options.soma_radius

    try:
        attr.window_width = int(os.environ['EQ_WINDOW_IATTR_HINT_WIDTH'])
    except:
        pass
    try:
        attr.window_height = int(os.environ['EQ_WINDOW_IATTR_HINT_HEIGHT'])
    except:
        pass
    if options.window_size:
        attr.window_width, attr.window_height = options.window_size
    attr.has_gui = options.gui

    # Old nomenclature
    soma_radii = AttributeMap()

    # Old nomenclature
    soma_radii.NGC = 7.46
    soma_radii.CRC = 7.28
    soma_radii.BP = 6.74
    soma_radii.L6CTPC = 7.00
    soma_radii.ChC = 8.01
    soma_radii.L4PC = 7.84
    soma_radii.L2PC = 8.36
    soma_radii.LBC = 9.28
    soma_radii.NBC = 8.61
    soma_radii.L5STPC = 8.61
    soma_radii.L6CCPC = 7.71
    soma_radii.L3PC = 7.70
    soma_radii.ADC = 7.72
    soma_radii.L5TTPC = 11.17
    soma_radii.AHC = 7.54
    soma_radii.L5UTPC = 9.75
    soma_radii.MC = 9.37
    soma_radii.DBC = 6.58
    soma_radii.SBC = 7.75
    soma_radii.L4SS = 8.87
    soma_radii.BTC = 9.15
    soma_radii.L4SP = 7.76
    soma_radii.L6CLPC = 7.32

    # New nomenclature
    soma_radii.L1_DAC = 7.77242212296
    setattr(soma_radii, "L1_NGC-DA", 7.71745554606)
    setattr(soma_radii, "L1_NGC-SA", 7.53378756841)
    soma_radii.L1_HAC = 7.48774570227
    soma_radii.L1_DLAC = 7.63075244427
    soma_radii.L1_SLAC = 6.54029955183
    soma_radii.L23_PC = 7.7618568695
    soma_radii.L23_MC = 8.25832914484
    soma_radii.L23_BTC = 9.30396906535
    soma_radii.L23_DBC = 6.67491123753
    soma_radii.L23_BP = 6.47212839127
    soma_radii.L23_NGC = 8.40507149696
    soma_radii.L23_LBC = 9.0722948432
    soma_radii.L23_NBC = 8.64244471205
    soma_radii.L23_SBC = 7.77334165573
    soma_radii.L23_ChC = 8.38753128052
    soma_radii.L4_PC = 7.5360350558
    soma_radii.L4_SP = 7.97097187241
    soma_radii.L4_SS = 8.95526583989
    soma_radii.L4_MC = 9.64632482529
    soma_radii.L4_BTC = 8.13213920593
    soma_radii.L4_DBC = 9.06448200771
    soma_radii.L4_BP = 5.80337953568
    soma_radii.L4_NGC = 7.61350679398
    soma_radii.L4_LBC = 8.92902739843
    soma_radii.L4_NBC = 9.02906776877
    soma_radii.L4_SBC = 9.16592723673
    soma_radii.L4_ChC = 8.54107379913
    soma_radii.L5_TTPC1 = 12.4378146535
    soma_radii.L5_TTPC2 = 12.9319117383
    soma_radii.L5_UTPC = 7.55000627041
    soma_radii.L5_STPC = 8.84197418645
    soma_radii.L5_MC = 9.02451186861
    soma_radii.L5_BTC = 9.24175042372
    soma_radii.L5_DBC = 8.92543775895
    soma_radii.L5_BP = 5.98329114914
    soma_radii.L5_NGC = 6.62666320801
    soma_radii.L5_LBC = 10.0915957915
    soma_radii.L5_NBC = 9.00336164747
    soma_radii.L5_SBC = 10.0064823627
    soma_radii.L5_ChC = 7.85296694438
    soma_radii.L6_TPC_L1 = 7.11837192363
    soma_radii.L6_TPC_L4 = 7.08293668837
    soma_radii.L6_UTPC = 7.74583534818
    soma_radii.L6_IPC = 7.71743569988
    soma_radii.L6_BPC = 7.14034402714
    soma_radii.L6_MC = 9.55710192858
    soma_radii.L6_BTC = 9.97289562225
    soma_radii.L6_DBC = 9.88600463867
    soma_radii.L6_BP = 5.98329114914
    soma_radii.L6_NGC = 6.62666320801
    soma_radii.L6_LBC = 9.49501685154
    soma_radii.L6_NBC = 9.33885664259
    soma_radii.L6_SBC = 8.06322860718
    soma_radii.L6_ChC = 8.15173133214
    attr.soma_radii = soma_radii

    # Default view style attributes
    view = AttributeMap()
    view.lod_bias = 0.5
    view.background = options.background
    view.use_roi = options.use_roi
    if options.spike_tail:
        view.spike_tail = options.spike_tail
    attr.view = view

    # Default profiling attributes
    if options.profile:
        # Idle AA is disabled when profiling is enabled because otherwise the
        # final frame count becomes undeterministic in some cases.
        view.idle_AA_steps = 0
        attr.profile = options.profile
        attr.profile.enable = True

    return attr

    # Per layer values for interneurons
    # Layer 1  Layer 2   Layer 3   Layer 4  Layer 5   Layer 6
    # NGC 7.15 NGC  7.61 NGC  7.61 NGC 7.61 NGC  7.61 NGC 7.61
    # CRC 7.27
    #          BP   6.47 BP   6.47 BP  7.10 BP   7.10 BP  7.10
    #          ChC  8.04 ChC  8.01 ChC 7.97 ChC  8.04 ChC 7.98
    # LBC 6.62 LBC  8.53 LBC  8.53 LBC 9.75 LBC 10.03 LBC 9.38
    # NBC 7.48 NBC  8.75 NBC  8.87 NBC 8.20 NBC  7.80 NBC 8.75
    #          MC   9.85 MC   9.41 MC  9.32 MC   8.93 MC  9.48
    # ADC 7.72
    # AHC 7.53
    #          DBC  6.46 DBC  6.63 DBC 6.52 DBC  6.63 DBC 6.85
    # SBC 6.55 SBC  7.02 SBC  7.19 SBC 9.00 SBC  7.76 SBC 7.79
    #          BTC  9.13 BTC  9.57 BTC 8.35 BTC  9.36 BTC 9.62
