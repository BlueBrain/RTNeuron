/* Copyright (c) 2006-2018, Ecole Polytechnique Federale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politecnica de Madrid (UPM)
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

#include "../helpers.h"
#include "docstrings.h"

#include <boost/python.hpp>

#include <rtneuron/sceneops/NeuronClipping.h>

#include <brain/neuron/morphology.h>
#include <brain/synapses.h>

using namespace boost::python;
using namespace bbp::rtneuron;
using namespace bbp::rtneuron::sceneops;

// export_NeuronClipping ------------------------------------------------------

template <typename T>
void _getVector(object in, std::vector<T>& out)
{
    out.reserve(len(in));
    stl_input_iterator<T> i(in), end;
    while (i != end)
        out.push_back(*i++);
}

typedef boost::shared_ptr<NeuronClipping> NeuronClippingPtr;

NeuronClippingPtr NeuronClipping_clip(NeuronClipping& clipping,
                                      object sections_, object starts_,
                                      object ends_)
{
    uint16_ts sections;
    _getVector(sections_, sections);
    floats starts;
    _getVector(starts_, starts);
    floats ends;
    _getVector(ends_, ends);
    return clipping.clip(sections, starts, ends).shared_from_this();
}

NeuronClippingPtr NeuronClipping_unclip(NeuronClipping& clipping,
                                        object sections_, object starts_,
                                        object ends_)
{
    uint16_ts sections;
    _getVector(sections_, sections);
    floats starts;
    _getVector(starts_, starts);
    floats ends;
    _getVector(ends_, ends);
    return clipping.unclip(sections, starts, ends).shared_from_this();
}

NeuronClippingPtr NeuronClipping_clipAll(NeuronClipping& clipping,
                                         const bool alsoSoma)
{
    return clipping.clipAll(alsoSoma).shared_from_this();
}

NeuronClippingPtr NeuronClipping_unclipAll(NeuronClipping& clipping)
{
    return clipping.unclipAll().shared_from_this();
}

NeuronClippingPtr NeuronClipping_unclipAfferentBranches(
    NeuronClipping& clipping, uint32_t gid,
    const brain::neuron::Morphology& morphology,
    const brain::Synapses& synapses)
{
    return clipping.unclipAfferentBranches(gid, morphology, synapses)
        .shared_from_this();
}

NeuronClippingPtr NeuronClipping_unclipEfferentBranches(
    NeuronClipping& clipping, uint32_t gid,
    const brain::neuron::Morphology& morphology,
    const brain::Synapses& synapses)
{
    return clipping.unclipEfferentBranches(gid, morphology, synapses)
        .shared_from_this();
}

void export_NeuronClipping()
// clang-format off
{

class_<NeuronClipping, NeuronClippingPtr,
       bases<Scene::ObjectOperation>, boost::noncopyable>
("NeuronClipping", DOXY_CLASS(bbp::rtneuron::sceneops::NeuronClipping))
    .def("clip", NeuronClipping_clip, args("sections", "starts", "ends"),
         DOXY_FN(bbp::rtneuron::sceneops::NeuronClipping::clip))
    .def("unclip", NeuronClipping_unclip, args("sections", "starts", "ends"),
         DOXY_FN(bbp::rtneuron::sceneops::NeuronClipping::clip))
    .def("clipAll", &NeuronClipping_clipAll, args("alsoSoma") = false,
         DOXY_FN(bbp::rtneuron::sceneops::NeuronClipping::clipAll))
    .def("unclipAll", &NeuronClipping_unclipAll,
         DOXY_FN(bbp::rtneuron::sceneops::NeuronClipping::unclipAll))
    .def("unclipAfferentBranches", &NeuronClipping_unclipAfferentBranches,
         DOXY_FN(bbp::rtneuron::sceneops::NeuronClipping::unclipAfferentBranches))
    .def("unclipEfferentBranches", &NeuronClipping_unclipEfferentBranches,
         DOXY_FN(bbp::rtneuron::sceneops::NeuronClipping::unclipEfferentBranches));
}
