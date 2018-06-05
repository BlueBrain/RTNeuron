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

#include <boost/python/enum.hpp>

#include "docstrings.h"

#include "rtneuron/types.h"

using namespace bbp::rtneuron;

void export_Types()
{
    using namespace boost::python;

    enum_<DataBasePartitioning> partitioning(
        "DataBasePartitioning", DOXY_ENUM(bbp::rtneuron::DataBasePartitioning));
    partitioning.value("NONE", DataBasePartitioning::NONE);
    partitioning.value("ROUND_ROBIN", DataBasePartitioning::ROUND_ROBIN);
    partitioning.value("SPATIAL", DataBasePartitioning::SPATIAL);

    enum_<RepresentationMode> representationMode(
        "RepresentationMode", DOXY_ENUM(bbp::rtneuron::RepresentationMode));
    representationMode.value("SOMA", RepresentationMode::SOMA);
    representationMode.value("SEGMENT_SKELETON",
                             RepresentationMode::SEGMENT_SKELETON);
    representationMode.value("WHOLE_NEURON", RepresentationMode::WHOLE_NEURON);
    representationMode.value("NO_AXON", RepresentationMode::NO_AXON);
    representationMode.value("NO_DISPLAY", RepresentationMode::NO_DISPLAY);

    enum_<NeuronLOD> neuronLod("NeuronLOD",
                               DOXY_ENUM(bbp::rtneuron::NeuronLOD));
    neuronLod.value("MEMBRANE_MESH", NeuronLOD::MEMBRANE_MESH);
    neuronLod.value("TUBELETS", NeuronLOD::TUBELETS);
    neuronLod.value("HIGH_DETAIL_CYLINDERS", NeuronLOD::HIGH_DETAIL_CYLINDERS);
    neuronLod.value("LOW_DETAIL_CYLINDERS", NeuronLOD::LOW_DETAIL_CYLINDERS);
    neuronLod.value("DETAILED_SOMA", NeuronLOD::DETAILED_SOMA);
    neuronLod.value("SPHERICAL_SOMA", NeuronLOD::SPHERICAL_SOMA);

    enum_<ColorScheme> colorScheme("ColorScheme",
                                   DOXY_ENUM(bbp::rtneuron::ColorScheme));
    colorScheme.value("SOLID", ColorScheme::SOLID);
    colorScheme.value("RANDOM", ColorScheme::RANDOM);
    colorScheme.value("BY_BRANCH_TYPE", ColorScheme::BY_BRANCH_TYPE);
    colorScheme.value("BY_WIDTH", ColorScheme::BY_WIDTH);
    colorScheme.value("BY_DISTANCE_TO_SOMA", ColorScheme::BY_DISTANCE_TO_SOMA);
}
