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

#ifndef RTNEURON_SCENE_MODEL_CACHE_H
#define RTNEURON_SCENE_MODEL_CACHE_H

#include "coreTypes.h"
#include "types.h"

#include "NeuronParts.h"

namespace bbp
{
class Neuron;

namespace rtneuron
{
namespace core
{
namespace model
{
class Cache
{
public:
    /*--- Public constructors/destructor ---*/

    Cache(const CircuitScene& scene);
    ~Cache();

    /*--- Public member functions ---*/

    /**
       Get or create a neuron model of the specificied type for a neuron.

       The model will always be complete, i.e., all neuron parts will be
       present.
       Thread safe
    */
    ModelPtr getModel(NeuronLOD lod, const ConstructionData& data) const;

    /**
       Get or create a neuron model of the specificied type for a neuron.

       The model will always be complete, i.e., all neuron parts will be
       present.
       Thread safe
    */
    ModelPtr createModel(NeuronLOD lod, NeuronParts parts,
                         const ConstructionData& data) const;

    /**
       Return the skeleton to be shared by all LODs of the same Neuron object.

       The returned skeleton must only be used inside a specific CircuitScene
       because different subscene may require different clipping.

       @param neuron The neuron for which the skeleton is requested.
       @param parts A hint of the usage that will be given to the returned
              skeleton. Only meaningful for NO_AXON display mode when
              unique morphologies can be assumed. For a given neuron the
              requested parts must always be the same.

       Thread-safe as long as not called with the same neuron as argument.
    */
    SkeletonPtr getSkeleton(const Neuron& neuron, NeuronParts parts) const;

    /**
       Clears all internal global caching including, but not limited to,
       skeletons, primitive sets, and geometry objects.
     */
    static void clear();

private:
    /*--- Private member attributes ---*/

    class Impl;
    mutable Impl* _impl;
};
}
}
}
}
#endif
