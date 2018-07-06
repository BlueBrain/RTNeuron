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

#ifndef RTNEURON_CORE_TYPES_H
#define RTNEURON_CORE_TYPES_H

#include <brion/types.h>

#include <boost/shared_ptr.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace osg
{
template <typename T>
class ref_ptr;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
class CircuitCache;
using CircuitCachePtr = std::shared_ptr<CircuitCache>;
class CircuitDataStorage;
class CircuitSceneAttributes;
using CircuitSceneAttributesPtr = std::shared_ptr<CircuitSceneAttributes>;
class CircuitScene;
class CollectSkeletonsVisitor;
class ColorMap;
class ColorMapAtlas;
using ColorMapAtlasPtr = std::shared_ptr<ColorMapAtlas>;
class CUDAContext;
class DetailedNeuronModel;
using DetailedNeuronModelPtr = osg::ref_ptr<DetailedNeuronModel>;
class GeometryObject;
using GeometryObjectPtr = std::shared_ptr<GeometryObject>;
using GIDToIndexMap = std::unordered_map<uint32_t, size_t>;
using brion::GIDSet;
class LODNeuronModel;
class NeuronModelClipping;
class ModelObject;
using ModelObjectPtr = std::shared_ptr<ModelObject>;
class ModelViewInverseUpdate;
class Neuron;
using NeuronPtr = std::shared_ptr<Neuron>;
using NeuronList = std::vector<NeuronPtr>;
class Neurons;
class NeuronMesh;
using NeuronMeshPtr = std::shared_ptr<NeuronMesh>;
using NeuronMeshes = std::vector<NeuronMeshPtr>;
using NeuronMap = std::unordered_map<uint32_t, NeuronPtr>;
class NeuronColoring;
class NeuronDisplayAndSelectionControl;
class NeuronModel;
using NeuronModelPtr = osg::ref_ptr<NeuronModel>;
using NeuronModels = std::vector<NeuronModelPtr>;
class NeuronModelDrawable;
class NeuronObject;
using NeuronObjectPtr = std::shared_ptr<NeuronObject>;
template <typename T>
class PerFrameAttribute;
class RenderBinManager;
class SceneObject;
class SceneObjectOperation;
class SceneOperation;
using SceneOperationPtr = std::shared_ptr<SceneOperation>;
class SceneOperations;
class SceneStyle;
using SceneStylePtr = std::shared_ptr<SceneStyle>;
class Skeleton;
using SkeletonPtr = std::shared_ptr<Skeleton>;
class SimulationDataMapper;
using SimulationDataMapperPtr = std::shared_ptr<SimulationDataMapper>;
class SimulationRenderBuffer;
using SimulationRenderBufferPtr = std::shared_ptr<SimulationRenderBuffer>;
class SphereSet;
class SphericalSomaModel;
class SpikeReport;
using SpikeReportPtr = std::shared_ptr<SpikeReport>;
using Strings = std::vector<std::string>;
class SubScene;
using SubScenePtr = std::shared_ptr<SubScene>;
class SynapseObject;
using SynapseObjectPtr = std::shared_ptr<SynapseObject>;
class ViewStyle;
using ViewStylePtr = std::shared_ptr<ViewStyle>;

using brion::floats;
using brion::floatsPtr;
using brion::uint16_ts;
using brion::uint16_tsPtr;
using brion::uint32_ts;
using brion::uint32_tsPtr;
using brion::Vector3fsPtr;

namespace model
{
class ConstructionData;
class NeuronSkeleton;
using SkeletonPtr = std::shared_ptr<NeuronSkeleton>;
class SkeletonModel;
using ModelPtr = std::shared_ptr<SkeletonModel>;
}
}
}
}
#endif
