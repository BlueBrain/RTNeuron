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
#ifdef RTNEURON_USE_NEUMESH
    generateMeshes = attributes("generate_meshes", true);
#else
    generateMeshes = attributes("generate_meshes", false);
#endif
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

void CircuitSceneAttributes::removeMeshLODs()
{
    LBINFO << "Removing mesh-based models from level-of-detail list"
           << std::endl;
    useMeshes = false;
    if (!neuronLODs)
        return;
    auto& lods = *neuronLODs;
    const auto expandLOD = [&lods](const char* name, const double start,
                                   const double end, const bool forceSet){
        double start2 = start, end2 = end;
        if (lods.get(name, start2, end2) == 2)
        {
            start2 = std::min(start, start2);
            end2 = std::max(end, end2);
        }
        else if (!forceSet)
            return false;
        lods.set(name, start2, end2);
        return true;
    };

    double start, end;
    if (lods.get("detailed_soma", start, end) == 2)
    {
        lods.unset("detailed_soma");
        expandLOD("spherical_soma", start, end, true);
    }
    if (lods.get("mesh", start, end) == 2)
    {
        lods.unset("mesh");
        expandLOD("spherical_soma", start, end, true);
        if (!expandLOD("tubelets", start, end, false))
            if (!expandLOD("high_detail_cylinders", start, end, false))
                expandLOD("low_detail_cylinders", start, end, false);
    }
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
