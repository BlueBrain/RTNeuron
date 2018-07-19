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

#include "CircuitSceneAttributes.h"

#include "util/attributeMapHelpers.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
bool s_preloadSkeletonsHint = ::getenv("RTNEURON_PRELOAD_SKELETONS") != 0;
}

/*
  CircuitSceneAttributes
*/
CircuitSceneAttributes::CircuitSceneAttributes(const AttributeMap& attributes)
{
    using namespace AttributeMapHelpers;
#ifdef USE_CUDA
    useCUDACulling = attributes("use_cuda", true);
#else
    useCUDACulling = false;
#endif
    useMeshes = attributes("use_meshes", true);
    _forceMeshLoading = attributes("force_mesh_loading", false);
    generateMeshes = attributes("generate_meshes", true);
    attributes.get("mesh_path", _userMeshPath);

    meshBasedSpatialPartition =
        useMeshes && attributes("mesh_based_partition", false);
    connectFirstOrderBranches =
        attributes("connect_first_order_branches", true);
    assumeUniqueMorphologies = attributes("unique_morphologies", false);

    preloadSkeletons = attributes("preload_skeletons", s_preloadSkeletonsHint);

    AttributeMapPtr dummy = attributes("lod.neurons", AttributeMapPtr());
    neuronLODs = dummy;

    partitionType = getEnum<DataBasePartitioning>(attributes, "partitioning",
                                                  DataBasePartitioning::NONE);

    primitiveOptions.strips = attributes("primitive_options.use_strips", true);
    primitiveOptions.unrollStrips =
        attributes("primitive_options.unroll_strips", false);
}

bool CircuitSceneAttributes::areMeshesRequired() const
{
    if (!useMeshes)
        return false;

    if (_forceMeshLoading)
        return true;

    if (neuronLODs)
    {
        double start, end;
        return ((neuronLODs->get("mesh", start, end) == 2 ||
                 neuronLODs->get("detailed_soma", start, end) == 2) &&
                start < end);
    }

    return true;
}

const std::string& CircuitSceneAttributes::getMeshPath() const
{
    if (!_userMeshPath.empty())
        return _userMeshPath;
    return circuitMeshPath;
}
}
}
}
