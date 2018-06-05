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
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <rtneuron/types.h>

#define BOOST_TEST_MODULE
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(test_representation_mode)
{
    using namespace bbp::rtneuron;

    BOOST_CHECK_EQUAL(lexical_cast<std::string>(RepresentationMode::SOMA),
                      "soma");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(
                          RepresentationMode::SEGMENT_SKELETON),
                      "skeleton");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(
                          RepresentationMode::WHOLE_NEURON),
                      "detailed");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(RepresentationMode::NO_DISPLAY),
                      "none");
    BOOST_CHECK_THROW(lexical_cast<std::string>(
                          RepresentationMode::NUM_REPRESENTATION_MODES),
                      std::runtime_error);

    BOOST_CHECK(lexical_cast<RepresentationMode>("soma") ==
                RepresentationMode::SOMA);
    BOOST_CHECK(lexical_cast<RepresentationMode>("skeleton") ==
                RepresentationMode::SEGMENT_SKELETON);
    BOOST_CHECK(lexical_cast<RepresentationMode>("detailed") ==
                RepresentationMode::WHOLE_NEURON);
    BOOST_CHECK(lexical_cast<RepresentationMode>("none") ==
                RepresentationMode::NO_DISPLAY);
    BOOST_CHECK_THROW(lexical_cast<RepresentationMode>("invalid"),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_neuron_models)
{
    using namespace bbp::rtneuron;

    BOOST_CHECK_EQUAL(lexical_cast<std::string>(NeuronLOD::MEMBRANE_MESH),
                      "mesh");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(NeuronLOD::SPHERICAL_SOMA),
                      "spherical_soma");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(NeuronLOD::DETAILED_SOMA),
                      "detailed_soma");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(
                          NeuronLOD::HIGH_DETAIL_CYLINDERS),
                      "high_detail_cylinders");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(
                          NeuronLOD::LOW_DETAIL_CYLINDERS),
                      "low_detail_cylinders");
    BOOST_CHECK_EQUAL(lexical_cast<std::string>(NeuronLOD::TUBELETS),
                      "tubelets");
    BOOST_CHECK_THROW(lexical_cast<std::string>(NeuronLOD::NUM_NEURON_LODS),
                      std::runtime_error);

    BOOST_CHECK(lexical_cast<NeuronLOD>("mesh") == NeuronLOD::MEMBRANE_MESH);
    BOOST_CHECK(lexical_cast<NeuronLOD>("spherical_soma") ==
                NeuronLOD::SPHERICAL_SOMA);
    BOOST_CHECK(lexical_cast<NeuronLOD>("detailed_soma") ==
                NeuronLOD::DETAILED_SOMA);
    BOOST_CHECK(lexical_cast<NeuronLOD>("high_detail_cylinders") ==
                NeuronLOD::HIGH_DETAIL_CYLINDERS);
    BOOST_CHECK(lexical_cast<NeuronLOD>("low_detail_cylinders") ==
                NeuronLOD::LOW_DETAIL_CYLINDERS);
    BOOST_CHECK(lexical_cast<NeuronLOD>("tubelets") == NeuronLOD::TUBELETS);
    BOOST_CHECK_THROW(lexical_cast<NeuronLOD>("invalid"), std::runtime_error);
}
