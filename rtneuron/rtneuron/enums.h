/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Juan Hernando <juan.hernando@epfl.ch>
 *
 * This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef RTNEURON_API_ENUMS_H
#define RTNEURON_API_ENUMS_H

namespace bbp
{
namespace rtneuron
{
/**
   Partitioning scheme to be applied to neurons in DB (sort-last) rendering
   configurations.
*/
#if DOXYGEN_TO_BREATHE
enum DataBasePartitioning
#else
enum class DataBasePartitioning
#endif
{
    NONE,
    ROUND_ROBIN,
    SPATIAL
};

/**
   Representation mode for neurons
*/
#if DOXYGEN_TO_BREATHE
enum RepresentationMode
#else
enum class RepresentationMode
#endif
{
    SOMA,
    SEGMENT_SKELETON,
    WHOLE_NEURON,
    NO_AXON,
    NO_DISPLAY,
    NUM_REPRESENTATION_MODES
};

/**
   Coloring mode for structural rendering of neurons.
*/
#if DOXYGEN_TO_BREATHE
enum ColorScheme
#else
enum class ColorScheme
#endif
{
    SOLID,          //!< Render the whole neuron with its primary color.
    RANDOM,         //!< Use a random color for the whole neuron.
    BY_BRANCH_TYPE, /*!< Render dendrites with the primary color and axons
                      with the secondary color. */
    BY_WIDTH, /*!< Apply a different color to each vertex based on it branch
        width. The color is interpolated from a color map computed using both
        the primary and secondary colors.

        If simulation display is enabled, the alpha channel of the colormap is
        used to modulate the final rendering color. */
    BY_DISTANCE_TO_SOMA, /*!< Apply per-vertex colors based on the distance to
        the soma. The colormap used is derived from the primary and secondary
        colors by default, unless a \c by_distance_to_soma colormap is set in
        the \c colormaps attribute or the neuron object.

        If simulation display is enabled, the alpha channel of the colormap is
        used to modulate the final rendering color. */
    NUM_COLOR_SCHEMES
};

/**
    Models used for level of detail representation of neurons.
*/
#if DOXYGEN_TO_BREATHE
enum NeuronLOD
#else
enum class NeuronLOD
#endif
{
    MEMBRANE_MESH = 0,
    TUBELETS,
    HIGH_DETAIL_CYLINDERS,
    LOW_DETAIL_CYLINDERS,
    DETAILED_SOMA,
    SPHERICAL_SOMA,
    NUM_NEURON_LODS,
};
}
}
#endif
