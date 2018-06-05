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

#ifndef RTNEURON_CIRCUITSCENEATTRIBUTES_H
#define RTNEURON_CIRCUITSCENEATTRIBUTES_H

#include "../AttributeMap.h"
#include "../types.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class CircuitSceneAttributes
{
    /* Constructor */
public:
    /**
       Attribute map:
       @see Scene for further details about attributes.

       - *connect_first_order_branches* (bool)
       - *load_morphologies* (bool)
       - *lod* (AttributeMap) :
         LOD specifications. A nested attribute map for each object type.
         - *neurons*:
           A model type name and two floats in the range [0, 1]. LOD names are
           *mesh*, *high_detail_cylinders*, *low_detail_cylinders*, *tubelets*
           *detailed_soma*, *spherical_soma*
       - *mesh_based_partition* (bool)
       - *mesh_path* (string):
         Path where neuron meshes are to be located. If set overrides any
         value set to circuitMeshPath later on.
       - *partition_type* (DataBasePartitioning)
       - *preload_skeletons* (bool)
       - *primitive_options* (AttributeMap):
         - *use_strips* (bool)
         - *unroll_stripgs* (bool)
       - *unique_morphologies* (bool)
       - *use_cuda* (bool)
       - *use_meshes* (bool)
     */
    CircuitSceneAttributes(const AttributeMap& attributes = AttributeMap());

public:
    /*--- Public member variables ---*/

    bool useCUDACulling;

    bool useMeshes;
    bool generateMeshes;
    std::string circuitMeshPath;

    bool meshBasedSpatialPartition;
    bool connectFirstOrderBranches;
    bool loadMorphologies;
    bool assumeUniqueMorphologies;

    bool preloadSkeletons;

    /* These attributes are not interpreted because the logic is
       internal to NeuronModel and its derived classes. */
    AttributeMapPtr neuronLODs;

    /* Partition type for sort-last rendering */
    DataBasePartitioning partitionType;

    /* Misc stuff */
    struct
    {
        // cppcheck-suppress unusedStructMember
        bool strips;
        // cppcheck-suppress unusedStructMember
        bool unrollStrips;
    } primitiveOptions;

    /*--- Public member functions ---*/

    bool areMeshesRequired() const;

    const std::string& getMeshPath() const;

private:
    std::string _userMeshPath;
    /* internal, just for mocking the behaviour of pre-BBPSDK removal code. */
    bool _forceMeshLoading;
};
}
}
}
#endif
