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

#ifndef RTNEURON_API_SCENEOPS_NEURONCLIPPING_H
#define RTNEURON_API_SCENEOPS_NEURONCLIPPING_H

#include "../Scene.h"

#include <brain/neuron/types.h>
#include <brain/types.h>

#include <boost/enable_shared_from_this.hpp>

namespace bbp
{
namespace rtneuron
{
namespace sceneops
{
/**
   This class provides a branch level clipping operation for neurons.

   Culling must be enabled in the scene that contains the target object.
   Otherwise this operation will have no effect.

   The clipping state to apply is specified by a set of functions to make
   visible/invisible ranges of the morphological sections.

   The culling mechanism discretizes sections in a predefined number of
   portions per section. Despite the API provides finer culling description,
   all the operations will work at the resolution defined by the
   implementation.

   The current resolution is at most 32 portions per section (regardless of
   the section length).

   Neuron clipping is affected by the representation mode in the following
   ways:
   - When all representation modes are available and the mode is changed,
     the clipping masks are cleared before applying the masks required by
     the new mode.
   - If the neuron was created with NO_AXON or SOMA modes, changing the
     representation mode does not affect the current clipping.
   - Clipping does not have any effect on the SOMA representation mode under
     any circumstances.
   - When the representation mode is NO_AXON, axon sections cannot be
     unclipped.

   Neuron clipping respects spatial partitions of DB configurations.

   @version 2.7
*/
class NeuronClipping : public Scene::ObjectOperation,
                       public boost::enable_shared_from_this<NeuronClipping>
{
public:
    /*-- Public declarations ---*/
    class _Impl;

    /*-- Public constructors/destructor ---*/

    /**
       Creates a clipping operation for neurons.

       At construction this is a "no operation"
    */
    NeuronClipping();

    ~NeuronClipping();

/*-- Public member functions ---*/

/**
   Mark section ranges for making them invisible.

   Discrete section portions are clipped only if the given range fully
   contains them. The ranges are considered as closed interval. Ranges
   applied to section 0 (assumed to be the soma), are always converted
   into [0, 1]).

   Subsequent calls to NeuronClipping::unclip will cut/split/remove the
   ranges to be applied.

   @param sections Section id list. Ids may be repeated.
   @param starts Relative start positions of the ranges. Each value must
          be smaller than the value at the same position of the 'ends'
          vector, otherwise the range is ignored.
   @param ends Relative end positions of the ranges. Each value must
          be greater than the value at the same position of the 'starts'
          vector, otherwise the range is ignored.
   @return self for operation concatenation.

   @throw std::invalid_argument if arrays have not the same size or if
          a range is ill-defined.
*/
#ifndef DOXYGEN_TO_BREATHE
    NeuronClipping& clip(const uint16_ts& sections, const floats& starts,
                         const floats& ends);
#else
    NeuronClipping& clip(object sections, object starts, objects ends);
#endif

/**
   Mark section ranges for making them visible.

   Discrete section portions are made visible if the given range
   contains at least part of them. The ranges are considered as an open
   interval (just touching the edge of a portion doesn't make it visible).
   Subsequent calls to NeuronClipping::clip will cut/split/remove the
   ranges to be applied. Ranges applied to section 0 (assumed to be the
   soma), are always converted into [0, 1]).

   @param sections Section id list. Ids may be repeated.
   @param starts Relative start positions of the ranges. Each value must
          be smaller than the value at the same position of the 'ends'
          vector.
   @param ends Relative end positions of the ranges. Each value must
          be greather than the value at the same position of the 'starts'
          vector.
   @return self for operation concatenation.

   @throw std::invalid_argument if arrays have not the same size or if
          a range is ill-defined.
*/
#ifndef DOXYGEN_TO_BREATHE
    NeuronClipping& unclip(const uint16_ts& sections, const floats& starts,
                           const floats& ends);
#else
    NeuronClipping& unclip(object sections, object starts, objects ends);
#endif

    /**
       Make all neurites and optionally the soma invisible.

       @param alsoSoma If true, the soma will be clipped.
       @return self for operation concatenation.
     */
    NeuronClipping& clipAll(const bool alsoSoma = false);

    /**
       Make all neurites and the soma visible.

       @return self for operation concatenation.
    */
    NeuronClipping& unclipAll();

    /**
       Apply the unclip masks that make visible the portions of efferent
       branches (dendrites) that connect soma of a neuron to the given
       synapses.
    */
    NeuronClipping& unclipAfferentBranches(
        uint32_t gid, const brain::neuron::Morphology& morphology,
        const brain::Synapses& synapses);

    /**
       Apply the unclip masks that make visible the portions of afferent
       branches (axon) that connect soma of a neuron to the given synapses.
    */
    NeuronClipping& unclipEfferentBranches(
        uint32_t gid, const brain::neuron::Morphology& morphology,
        const brain::Synapses& synapses);

    /**
       Returns true iff the input object is a handler to a neuron set
       consisting of exactly one neuron.
    */
    bool accept(const Scene::Object& object) const;

protected:
    /*--- Protected member functions ---*/
    std::shared_ptr<ObjectOperation::Impl> getImplementation();

private:
    /*--- Private member attributes ---*/

    std::shared_ptr<_Impl> _impl;
};
}
}
}
#endif
