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

#include "SubScene.h"

#define EQ_IGNORE_GLEW
#include "data/CircuitCache.h"
#include "data/NeuronMesh.h"
#include "data/SimulationDataMapper.h"
#include "render/SceneStyle.h"
#include "scene/SimulationRenderBuffer.h"
#include "scene/models/utils.h"
#include "util/attributeMapHelpers.h"
#include "util/log.h"
#include "util/vec_to_vec.h"
#include "viewer/osgEq/Config.h"
#include "viewer/osgEq/FrameData.h"

#include <brain/circuit.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>

#include <random>

namespace bbp
{
namespace rtneuron
{
namespace core
{
static const size_t SPATIAL_PARTITION_COUNT_BINS = 5000;

/*
  Helper functions
*/
namespace
{
double& (osg::BoundingBoxd::*boxSides[6])() = {&osg::BoundingBoxd::xMin,
                                               &osg::BoundingBoxd::xMax,
                                               &osg::BoundingBoxd::yMin,
                                               &osg::BoundingBoxd::yMax,
                                               &osg::BoundingBoxd::zMin,
                                               &osg::BoundingBoxd::zMax};
double (osg::BoundingBoxd::*constBoxSides[6])() const = {
    &osg::BoundingBoxd::xMin, &osg::BoundingBoxd::xMax,
    &osg::BoundingBoxd::yMin, &osg::BoundingBoxd::yMax,
    &osg::BoundingBoxd::zMin, &osg::BoundingBoxd::zMax};

using SortedMapping = std::unordered_map<uint16_t, floats>;

Neurons operator-(const Neurons& neurons,
                  const std::unordered_set<uint32_t>& gids)
{
    Neurons out;
    for (const auto& neuron : neurons)
    {
        if (gids.find(neuron->getGID()) != gids.end())
            out.insert(neuron);
    }
    return out;
}

/**
   Process a mesh to return a hash map that stores a list of sorted
   relative vertex positions for each section.
   Returns the count of soma vertices or 0 if the mesh was not found.
*/
SortedMapping _sortMeshMappingAndCountSomaVertices(const NeuronMesh& mesh,
                                                   const bool ignoreAxon,
                                                   unsigned int somaVertices)

{
    SortedMapping sortedMapping;
    const auto& sections = *mesh.getVertexSections();
    const auto& positions = *mesh.getVertexRelativeDistances();
    somaVertices = 0;
    for (size_t i = 0; i != sections.size(); ++i)
    {
        if (ignoreAxon && sections[i] == brion::SECTION_AXON)
            continue;
        if (sections[i] == brion::SECTION_SOMA)
            ++somaVertices;
        sortedMapping[sections[i]].push_back(positions[i]);
    }

    /* Sorting the positions within the sections (this should
       be almost sorted already). */
    for (auto& i : sortedMapping)
        std::sort(i.second.begin(), i.second.end());

    return sortedMapping;
}

/* The morphology is expected to be in global coordinates already */
void _appendMorphologicalPoints(
    const Neuron& neuron, const brain::neuron::Morphology& morphology,
    const NeuronMesh* mesh, const bool ignoreAxon,
    std::vector<std::pair<uint32_t, osg::Vec3>>& points,
    std::vector<unsigned int>& weights, size_t& totalWeight,
    osg::Vec3& upperLimits, osg::Vec3& lowerLimits)
{
    totalWeight = 0;
    /* This code is assuming that morphologies are unique. */
    const auto gid = neuron.getGID();

    unsigned int somaVertices = 0;
    SortedMapping sortedMapping;
    if (mesh)
        sortedMapping = _sortMeshMappingAndCountSomaVertices(*mesh, ignoreAxon,
                                                             somaVertices);
    const bool meshBased = !sortedMapping.empty();

    if (meshBased)
    {
        /* Pushing the soma center and its weight */
        points.push_back(std::make_pair(gid, neuron.getPosition()));
        weights.push_back(somaVertices);
        totalWeight += somaVertices;
    }
    else
    {
        /* Pushing the soma center 10 times to increase the weight around
           the soma */
        for (unsigned int i = 0; i < 10; ++i)
            points.push_back(std::make_pair(gid, neuron.getPosition()));
        totalWeight += 10;
    }

    using S = brain::neuron::SectionType;
    auto types = std::vector<S>{S::apicalDendrite, S::dendrite, S::axon};
    if (ignoreAxon)
        types.pop_back();
    for (const auto& section : morphology.getSections(types))
    {
        const float *position = 0, *last = 0;
        if (meshBased)
        {
            const floats& positions = sortedMapping[section.getID()];
            /* Some very small sections can end up with no vertices mapped
               to them, so we need to check if the list is emtpy. */
            if (!positions.empty())
            {
                position = &positions[0];
                last = position + positions.size();
            }
        }
        else
        {
            totalWeight += section.getNumSamples() - 1;
        }

        const auto samples = section.getSamples();
        const auto relativePositions =
            model::util::computeRelativeDistances(samples, section.getLength());
        /* Skipping the first sample */
        auto i = ++samples.begin();
        auto j = ++relativePositions.begin();
        for (; i != samples.end(); ++i, ++j)
        {
            if (meshBased)
            {
                unsigned int weight = 0;
                /* The mesh was loaded, counting the vertices between this
                   sample and the previous one */
                while (position != last && *position < *j)
                {
                    /* Increasing the weight of this segment */
                    ++position;
                    ++weight;
                }
                weights.push_back(weight);
                totalWeight += weight;
            }

            const auto& point = *i;
            points.push_back(
                std::make_pair(gid, osg::Vec3(point[0], point[1], point[2])));
            /* Updating the upper and lower limits with this point. */
            for (unsigned int k = 0; k < 3; ++k)
            {
                upperLimits[k] = std::max(upperLimits[k], point[k]);
                lowerLimits[k] = std::min(lowerLimits[k], point[k]);
            }
        }
    }
}

/**
   Weights only written if the partition is mesh based
*/
using PointList = std::vector<std::pair<uint32_t, osg::Vec3>>;
using Weights = std::vector<unsigned int>;

void _pointListFromNeurons(const Neurons& neurons,
                           const RepresentationMode mode, PointList& points,
                           Weights& weights, size_t& totalWeight,
                           osg::Vec3& upperLimits, osg::Vec3& lowerLimits,
                           const std::string meshPath)
{
    /* This code is supposed to be used by sort-last decompositions, so we
       don't want to keep all the data loaded after the point list is extracted.
       The only way to achive this is by copying the neuron container to a
       new one and clearing the morphology and mesh caches afterwards.
       The original code using BBPSDK was doing something similar, so
       performance is not expected to be worse. */
    const auto& circuit = *neurons.getCircuit();
    const double max = std::numeric_limits<double>::max();
    upperLimits = osg::Vec3(-max, -max, -max);
    lowerLimits = osg::Vec3(max, max, max);

    totalWeight = 0;
    if (mode != RepresentationMode::SOMA)
    {
        for (auto n = neurons.begin(); n != neurons.end();)
        {
            /* Processing neurons in batches of 500 neurons. */

            /* Extracting labels and gids of the batch */
            std::vector<std::string> names;
            names.reserve(std::max(neurons.size(), size_t(500)));
            GIDSet gids;
            size_t index = 0;
            auto first = n;
            for (; n != neurons.end() && index < 500; ++n, ++index)
            {
                names.push_back((*n)->getMorphologyLabel());
                gids.insert(gids.end(), (*n)->getGID());
            }

            /* Loading data */
            const auto morphologies =
                circuit.circuit->loadMorphologies(
                    gids, brain::Circuit::Coordinates::global);
            NeuronMeshes meshes;
            if (!meshPath.empty())
            {
                try
                {
                    MeshCache cache;
                    meshes = NeuronMesh::load(names, meshPath, cache);
                }
                catch (const std::runtime_error& e)
                {
                    std::cerr << "Computing mesh based spatial partition: "
                              << e.what() << std::endl;
                    std::cerr << "Falling back to morphology based partition"
                              << std::endl;
                    points.clear();
                    weights.clear();
                    _pointListFromNeurons(neurons, mode, points, weights,
                                          totalWeight, upperLimits, lowerLimits,
                                          "");
                }
            }

            /* Processing the morphologies to take only the points and the
               GIDs. */
            index = 0;
            for (auto i = first; i != n; ++i, ++index)
            {
                size_t weight;
                _appendMorphologicalPoints(**i, *morphologies[index],
                                           meshPath.empty()
                                               ? nullptr
                                               : meshes[index].get(),
                                           mode == RepresentationMode::NO_AXON,
                                           points, weights, weight, upperLimits,
                                           lowerLimits);
                totalWeight += weight;
            }
        }
    }
    else
    {
        for (const auto& neuron : neurons)
        {
            const auto point = neuron->getPosition();
            points.push_back(std::make_pair(neuron->getGID(), point));
            for (unsigned int i = 0; i < 3; ++i)
                upperLimits[i] = std::max(upperLimits[i], point[i]);
            for (unsigned int i = 0; i < 3; ++i)
                lowerLimits[i] = std::min(lowerLimits[i], point[i]);
        }
    }
}

/*
  Find the approximate the quantile point for a quantile value of a point
  distribution along one axis.
  Points are weighted by an optional weight vector. If empty, each point
  weights 1.
  The points outside the given limits are discarded.
*/

double _findQuantilePoint(const double quantileValue, const PointList& points,
                          const Weights& weights, const size_t totalWeight,
                          const osg::Vec3& upperLimits,
                          const osg::Vec3& lowerLimits,
                          const unsigned int splittingAxis)
{
    assert(weights.empty() || points.size() == weights.size());

    /* To save time, the quantile point is estimated using an approximate
       algorithm.
       Since we have the data range for the points and to avoid having
       to shuffle the points (otherwise quantile estimators based on
       delta increments don't work), we will use a n-bin histogram to
       aproximate where the quantile is.
       (In a second pass, the approximation could be refined) */
    unsigned int numIterations = 1;
    double up = upperLimits[splittingAxis];
    double down = lowerLimits[splittingAxis];

    for (unsigned int i = 0; i < numIterations; ++i)
    {
        std::vector<int> bins(SPATIAL_PARTITION_COUNT_BINS);

        const double binWidth =
            (up - down) / double(SPATIAL_PARTITION_COUNT_BINS);
        for (size_t p = 0; p != points.size(); ++p)
        {
            const double x = points[p].second[splittingAxis];
            if (x < down || x > up)
                continue;
            const size_t bin =
                std::min(SPATIAL_PARTITION_COUNT_BINS,
                         size_t(std::floor((x - down) / binWidth)));
            if (weights.empty())
                ++bins[bin];
            else
                bins[bin] += weights[p];
        }
        /* Finding the bin that contains the quantile */
        const int target =
            quantileValue * (weights.empty() ? points.size() : totalWeight);
        int counted = 0;
        int index = 0;
        while (counted < target)
        {
            counted += bins[index];
            if (counted < target)
                ++index;
        }
        /* Adjusting up and down */
        down = binWidth * index + down;
        up = down + binWidth;
    }
    return up * (quantileValue) + down * (1 - quantileValue);
}

float _findNeuronSize(const Neuron& neuron)
{
    static std::unordered_map<std::string, float> s_sizes;

    float& size = s_sizes[neuron.getMorphologyLabel()];
    if (size == 0)
        size = neuron.getMorphology()->getPoints().size();
    return size;
}
}

/*
  Constructors/destructor
*/
SubScene::SubScene(Scene::_Impl* parent)
    : CircuitScene(parent->_sceneAttributes)
    , _parent(parent)
    , _node(new osg::ClipNode())
    , _simulationBuffer(new SimulationRenderBuffer())
    , _simulationMapper(new SimulationDataMapper())
    , _timestampReady(std::numeric_limits<float>::quiet_NaN())
    , _doSwapSimBuffers(false)
    , _needsMappingUpdate(false)
{
    LBLOG(LOG_SCENE_UPDATES) << "SubScene " << this << std::endl;

    for (auto node : CircuitScene::getNodes())
        _node->addChild(node);

    _simulationMapper->setSimulationRenderBuffer(_simulationBuffer);
    _simulationMapper->simulationUpdated.connect(
        boost::bind(&SubScene::onSimulationUpdated, this, _1));
}

SubScene::~SubScene()
{
    LBLOG(LOG_SCENE_UPDATES) << "~SubScene " << this << std::endl;
}

/*
  Member functions
*/
void SubScene::computeSubtarget(Neurons& neurons, const RepresentationMode mode,
                                const float start, const float end)
{
    /* Mutual exclusion between this function and compositingPosition */
    std::unique_lock<std::mutex> lock(_mutex);

    if (start != 0.0 || end != 1.0)
    {
        switch (getAttributes().partitionType)
        {
        case DataBasePartitioning::ROUND_ROBIN:
            computeRoundRobinSubtarget(neurons, mode, start, end);
            break;
        case DataBasePartitioning::SPATIAL:
            computeSpatialPartitionSubtarget(neurons, mode, start, end);
            break;
        default:;
            computeRoundRobinSubtarget(neurons, mode, start, end);
            break;
        }
    }
}

void SubScene::computeRoundRobinSubtarget(Neurons& neurons,
                                          const RepresentationMode mode,
                                          const float start, const float end)
{
    assert(start <= end);
    if (start == end)
    {
        neurons.clear();
        return;
    }

    if (neurons.size() == 0)
        /* For the moment synapses are not considered.
           When synapses are considered, what do we do regarding the synapse
           radius:
           - limit it so a relatively small maximum
           - just consider it for synapse filtering
           - consider it also in the load balancing algorithm (sound difficult)
        */
        return;

    /* Computing a heuristic size of each element in the target. */
    typedef std::vector<std::pair<float, uint32_t>> SizeNeuronList;
    SizeNeuronList list;
    list.reserve(neurons.size());
    float totalSize = 0;
    if (mode == RepresentationMode::SOMA)
    {
        for (const auto& neuron : neurons)
            list.push_back(std::make_pair(1, neuron->getGID()));
        totalSize = neurons.size();
    }
    else
    {
        for (auto& neuron : neurons)
        {
            /* This code is reading every new morphology that appears to
               compute its size in terms of number of segments. We should
               either add that to a metadata section of the morphology or
               use a sampling method to not load all the morphology files
               in circuits with unique morphologies. */
            const float size = _findNeuronSize(*neuron);
            list.push_back(std::make_pair(size, neuron->getGID()));
            totalSize += size;
        }
    }

    /* For sufficiently large circuits this should provide a similarly
       distributed random sampling of the circuit to each node. */
    std::mt19937 generator(std::mt19937::default_seed);
    std::shuffle(list.begin(), list.end(), generator);

    /* Computing the relative accumulated sizes of the resorted list */
    {
        size_t index = 1;
        for (; index != list.size(); ++index)
        {
            /* Accumulating current */
            list[index].first += list[index - 1].first;
            /* Normalizing previous */
            list[index - 1].first /= totalSize;
        }
        list[index - 1].first = 1.0; /* Last element accumulates the total */
    }

    /* Annotating the GIDs outside the fraction corresponding to the interval
       given (these will be removed from the neurons container */
    typedef std::unordered_set<uint32_t> GIDs;
    GIDs gids;
    bool inside = start == 0.0;
    for (const auto& sizeGID : list)
    {
        if (!inside)
            gids.insert(sizeGID.second);
        /* Checking if the element of the next iteration is inside or outside
           the start, end range */
        if (sizeGID.first >= start)
            inside = true;
        if (sizeGID.first >= end)
            inside = false;
    }

    neurons = neurons - gids;
}

void SubScene::computeSpatialPartitionSubtarget(Neurons& neurons,
                                                RepresentationMode mode,
                                                float start, float end)
{
    assert(neurons.size() != 0 && start <= end);
    if (start == end)
    {
        neurons.clear();
        return;
    }
    if (neurons.size() == 1 && mode == RepresentationMode::SOMA)
    {
        if (start != 0)
            neurons.clear();
        return;
    }

    /* We will use two lists to ping-pong between them during kd-tree
       decomposition. This will use more memory, but it should increase.
       considerably the performance of the recursive steps. */
    PointList pointLists[2];
    Weights weightLists[2];
    int currentList = 0;
    osg::Vec3 upperLimits, lowerLimits;
    size_t totalWeight;

    const auto& attributes = getAttributes();
    const std::string meshPath = attributes.meshBasedSpatialPartition
                                     ? attributes.getMeshPath()
                                     : std::string();
    _pointListFromNeurons(neurons, mode, pointLists[0], weightLists[0],
                          totalWeight, upperLimits, lowerLimits, meshPath);

    /* Estimating the number of participating nodes and the position of
       this node in the range. */
    const double countf = 1 / (end - start);
    const int nodeCount = round(countf);
    const double positionf = start / (end - start);
    const int position = round(positionf);
    /* The threshold 0.0005 is a consequence of the discretization used
       in eq/server/config/resource.cpp (_addDBCompound and _addDSCompound) */
    if (std::abs(nodeCount - countf) > 0.005 ||
        std::abs(position - positionf) > 0.005)
    {
        LBERROR << "The DB range (" << start << ", " << end
                << ") may not belong to an iso-partition. ";
    }

    /* Splitting the scene using a kd-tree based on the morphological
       points. */
    int startNode = 0;
    int endNode = nodeCount - 1;
    int splittingAxis = 1; /* 0 for x, 1 for y, 2 for z */

    osg::BoundingBoxd box;
    /* This is the sequence of subdivision steps taken to reach this
       rendering node leaf.
       Note that the planes are oriented with the normal pointing in the
       opposite direction in which this node's scene is located, i.e.
       the distance between the scene and the plane is negative. */
    _splittingPlanes.clear();
    while (startNode != endNode)
    {
        /* To minimize the bias toward a particular location, we floor or
           round depending on the iteration. */
        typedef double (*Rounding)(double);
        Rounding rounding =
            _splittingPlanes.size() % 2 == 0 ? (Rounding)ceil : (Rounding)floor;
        const unsigned int nodes = endNode - startNode + 1;
        const double quantileValue = rounding(nodes * 0.5) / double(nodes);
        const PointList& points = pointLists[currentList];
        const Weights& weights = weightLists[currentList];
        const double quantilePoint =
            _findQuantilePoint(quantileValue, points, weights, totalWeight,
                               upperLimits, lowerLimits, splittingAxis);

        /* Deciding in which of the half spaces this node will fall. */
        const int midNode =
            int(round((endNode - startNode + 1) * quantileValue)) + startNode;
        const bool negativeSemispace = position < midNode;
        /* Activating the clipping plane for this node.
           The criteria to define the clipping plane is the following:
           The scene will be located in the semispace which is opposite to
           the direcction in which the normal points. Regarding the bounding
           box, the clip plane to update will be the upper one if the scene
           is in the negative semipace (the index is 1 + splittingAxis * 2)
           and the lower one otherwise (the index is splittingAxis * 2). */
        int planeIndex = splittingAxis * 2 + int(negativeSemispace);
        (box.*boxSides[planeIndex])() = quantilePoint;
        osg::Vec3 normal;
        /* The splitting sequence places the scene for this node in the
           negative sub-space. What is the negative space in the kd-tree is
           NOT the same than the negative subspace for this plane. */
        normal[splittingAxis] = negativeSemispace ? 1 : -1;
        _splittingPlanes.push_back(
            osg::Plane(normal, -quantilePoint * normal[splittingAxis]));

        /* Staying with half the points and half the nodes (approximately).
           Note: midNode is counted starting at 1,
                 endNode and startNode start at 0. */
        if (negativeSemispace)
        {
            upperLimits[splittingAxis] = quantilePoint;
            endNode = midNode - 1;
        }
        else
        {
            lowerLimits[splittingAxis] = quantilePoint;
            startNode = midNode;
        }
        pointLists[1 - currentList] = PointList();
        weightLists[1 - currentList] = Weights();
        const size_t reserveSize =
            points.size() *
                (negativeSemispace ? quantileValue : 1 - quantileValue) +
            50;
        pointLists[1 - currentList].reserve(reserveSize);
        if (!weights.empty())
            weightLists[1 - currentList].reserve(reserveSize);
        totalWeight = 0;
        for (size_t i = 0; i != points.size(); ++i)
        {
            const std::pair<uint32_t, osg::Vec3> p = points[i];
            const double value = p.second[splittingAxis];
            /** \todo Find the maximum soma radius instead of hardcoding it. */
            const double offset = mode == RepresentationMode::SOMA ? 50 : 0;
            if ((negativeSemispace && value <= quantilePoint + offset) ||
                (!negativeSemispace && value >= quantilePoint - offset))
            {
                pointLists[1 - currentList].push_back(p);
                if (!weights.empty())
                {
                    unsigned int weight = weights[i];
                    weightLists[1 - currentList].push_back(weight);
                    totalWeight += weight;
                }
            }
        }
        currentList = 1 - currentList;
        /* Next iteration will use a different axis */
        splittingAxis = (splittingAxis + 1) % 3;
    }
    assert(startNode == position);

    /* Filtering the neuron container based on the GIDs of the points. */
    std::unordered_set<uint32_t> gids;
    for (auto& i : pointLists[currentList])
        gids.insert(i.first);
    neurons = neurons - gids;

    _staticClipping = true;
    _applyClipBox(box);
}

unsigned int SubScene::compositingPosition(const osg::Matrix& modelView,
                                           unsigned int nodeCount) const
{
    /* Mutual exclusion between this function and computeSubtarget */
    std::unique_lock<std::mutex> lock(_mutex);

    if (_splittingPlanes.empty())
        return 0;

    const unsigned int totalNodeCount = nodeCount;
    unsigned int nodesBefore = 0;
    bool even = true;
    int splittingAxis = 1;
    for (std::vector<osg::Plane>::const_iterator i = _splittingPlanes.begin();
         i != _splittingPlanes.end();
         ++i, even = !even, splittingAxis = (splittingAxis + 1) % 3)
    {
        osg::Plane plane = *i;
        /* Computing the nodes counts on the negative and positive sides
           of the planes. */
        double quantileValue;
        /* Calculation analogous to computeSpatialPartitionSubtarget */
        if (even)
            quantileValue = ceil(nodeCount * 0.5) / double(nodeCount);
        else
            quantileValue = floor(nodeCount * 0.5) / double(nodeCount);
        int midNode = int(round(nodeCount * quantileValue));
        /* The sign of the original plane normal tells us whether this node
           is at the left or the right of the split point in the sequence.
           (midNode is at the left and the next one is at the right). */
        bool atLeft = plane[splittingAxis] == 1;

        /* The content of this node is in the negative semispace defined by
           the plane. The node is at front if the viewpoint is also in the
           negative semispace (plane.w < 0). */
        plane.transform(modelView);
        bool atFront = plane[3] < 0;
        if (!atFront)
        {
            /* The nodes in the positive semispace of the plane are in front
               of this node. */
            nodesBefore += atLeft ? nodeCount - midNode : midNode;
        }
        nodeCount = atLeft ? midNode : nodeCount - midNode;
    }
    return totalNodeCount - 1 - nodesBefore;
}

void SubScene::addNeurons(const Neurons& neurons,
                          const AttributeMap& attributes)
{
    LBLOG(LOG_SCENE_UPDATES) << "addNeurons, subscene " << this << std::endl;

    using namespace AttributeMapHelpers;
    const auto mode =
        getEnum(attributes, "mode", RepresentationMode::WHOLE_NEURON);
    const NeuronColoring coloring(attributes);
    const int maxVisibleOrder = attributes("max_visible_branch_order", -1);

    std::stringstream progressMessage;
    progressMessage << "Creating " << neurons.size() << " neurons";
    size_t progress = 0;

    reportProgress(progressMessage.str(), progress, neurons.size());

    std::vector<NeuronPtr> toAdd(neurons.size());

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        auto original = neurons[i];
        try
        {
            auto neuron =
                std::make_shared<Neuron>(*original, getAttributes(), mode);
            neuron->applyColoring(coloring);
            if (maxVisibleOrder != -1)
                neuron->setMaximumVisibleBranchOrder(maxVisibleOrder);
            toAdd[i] = neuron;
        }
        catch (std::exception& e)
        {
            LBERROR << "Unable to load data for neuron a" << original->getGID()
                    << ": " << e.what() << std::endl;
        }

#pragma omp atomic update
        ++progress;
#ifdef RTNEURON_USE_OPENMP
        if (omp_get_thread_num() == 0 && progress < toAdd.size())
#endif
            reportProgress(progressMessage.str(), progress, neurons.size());
    }
    reportProgress(progressMessage.str(), progress, toAdd.size());

    Neurons renderable;
    for (const auto& neuron : toAdd)
    {
        if (neuron)
            renderable.insert(neuron);
    }

    CircuitScene::addNeurons(renderable);

    _simulationBuffer->setTargetStateSet(
        getNeuronModelNode()->getOrCreateStateSet());
    _needsMappingUpdate = true;
}

void SubScene::addSynapses(const SynapseObject& synapses)
{
    LBLOG(LOG_SCENE_UPDATES) << "addSynapses, subscene " << this << std::endl;

    const AttributeMap& attributes = synapses.getAttributes();
    using namespace AttributeMapHelpers;
    osg::Vec4 color;
    getColor(attributes, "color", color);
    const double radius = attributes("radius");

    std::vector<osg::Vec3> positions;
    for (const auto& p : synapses.getLocations())
        positions.push_back(vec_to_vec(p));

    _synapseSphereSets[synapses.getID()] =
        getSynapses().addSpheres(positions, radius, color);
}

SphereSet::SubSetID SubScene::findSynapseSubset(
    const unsigned int objectId) const
{
    SynapseSphereSets::const_iterator i = _synapseSphereSets.find(objectId);
    return i == _synapseSphereSets.end() ? SphereSet::SubSetID() : i->second;
}

void SubScene::highlightCells(const GIDSet& cells, const bool highlight)
{
    const auto neurons = getNeurons() & cells;
    for (const auto& neuron : neurons)
        neuron->highlight(highlight);
}

void SubScene::setSimulation(const CompartmentReportPtr& report)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _simulationMapper->setCompartmentReport(report);
    _needsMappingUpdate = true;
}

void SubScene::setSimulation(const SpikeReportPtr& report)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _simulationMapper->setSpikeReport(report);
    _needsMappingUpdate = true;
}

void SubScene::updateSimulationMapping(Neuron& neuron)
{
    if (_simulationMapper->getCompartmentReport() ||
        _simulationMapper->getSpikeReport())
        neuron.setupSimulationOffsetsAndDelays(*_simulationMapper, true);
}

bool SubScene::prepareSimulation(const uint32_t /* frameNumber */,
                                 const float milliseconds)
{
    if (_needsMappingUpdate)
    {
        _simulationMapper->updateMapping(getNeurons());
        _needsMappingUpdate = false;
    }

    _simulationMapper->setSimulationAtTimestamp(milliseconds);
    return true;
}

void SubScene::mapSimulation(const uint32_t /* frameNumber */,
                             const float milliseconds)
{
    std::unique_lock<std::mutex> lock(_mutex);

    if (_needsMappingUpdate)
    {
        _simulationMapper->updateMapping(getNeurons());
        _needsMappingUpdate = false;
    }

    if (_simulationBuffer->getFrontBufferTimestamp() == milliseconds)
    {
        LBLOG(LOG_SIMULATION_PLAYBACK)
            << "simulation timestamp already current: " << milliseconds
            << std::endl;
        return;
    }
    else if (_timestampReady == milliseconds)
    {
        LBLOG(LOG_SIMULATION_PLAYBACK)
            << "simulation timestamp in back buffer, schedule swap: "
            << milliseconds << std::endl;
        _doSwapSimBuffers = true;
        return;
    }

    LBLOG(LOG_SIMULATION_PLAYBACK) << "mapping simulation at: " << milliseconds
                                   << std::endl;

    /* This timestamp has not been prepared. */
    _simulationMapper->setSimulationAtTimestamp(milliseconds);
    /* It's not very desirable to include this unlock here, but
       as the simulation data mapper works right now, some call
       sequences in the SimulationPlayer interface can lead to
       deadlocks without the unlocking. */
    _simulationMapper->unblockUpdate();
    /* Swapping buffers in channelSync because otherwise draw async mode
       can have race conditions. */
    _doSwapSimBuffers = true;

    lock.unlock();
    _simulationMapper->waitForUpdate();
}

void SubScene::applyOperations(const std::vector<co::uint128_t>& commits)
{
    /* May be called from multiple pipes sharing the same subscene */
    std::unique_lock<std::mutex> lock(_mutex);

    for (const auto commit : commits)
    {
        _sceneOperations.sync(commit);
        _sceneOperations.apply(*this);
        _sceneOperations.clear();
    }
}

void SubScene::channelSync()
{
    /* May be called from multiple pipes sharing the same subscene */
    std::unique_lock<std::mutex> lock(_mutex);
    if (_doSwapSimBuffers)
    {
        LBLOG(LOG_SIMULATION_PLAYBACK) << "swap simulation buffers"
                                       << std::endl;

        _simulationBuffer->swapBuffers();
        _simulationMapper->unblockUpdate();
        _doSwapSimBuffers = false;
    }
}

void SubScene::mapDistributedObjects(osgEq::Config* config)
{
    const InitData& initData =
        static_cast<const InitData&>(config->getInitData());
    const auto uuid = initData.sceneUUIDs.find(_parent->getID());
    if (uuid == initData.sceneUUIDs.end())
        throw std::runtime_error("Scene ID not found");
    config->mapObject(&_sceneOperations, uuid->second);
}

void SubScene::unmapDistributedObjects(osgEq::Config* config)
{
    config->unmapObject(&_sceneOperations);
}

void SubScene::onSimulationUpdated(const float timestamp)
{
    std::unique_lock<std::mutex> lock(_mutex);

    if (_simulationBuffer->getFrontBufferTimestamp() == timestamp)
    {
        /* This can happen when simulation playback is paused and
           resumed. mapSimulation wont' have a way to know that
           channelSync still has to unblock the next update, but since
           swapping the buffers is harmless (it's even necessary if the
           simulation has been changed), the buffer swap flag is
           raised to ensure that next frame will unleash the mapper.
           Once more, with dash this will look cleaner. */
        _doSwapSimBuffers = true;
    }
    _timestampReady = timestamp;
    /* _doSwapBuffers is set to true once all subscenes participating
       in the frame are ready (see RTNeuron::_Impl::preNodeUpdate and
       SubScene::mapSimulation) */
}

void SubScene::setClipPlane(const unsigned int index, const osg::Vec4& plane)
{
    if (_staticClipping)
        /* Cannot apply clipping planes to a spatial DB partition. */
        return;

    osg::Vec4d planed(plane);

    /* OpenSceneGraph handles clipping planes in a strange way, after looking
       at the implementation the following code is the only I trust for
       updating planes. */
    for (auto clipPlane : _node->getClipPlaneList())
    {
        if (clipPlane->getClipPlaneNum() == index)
        {
            clipPlane->setClipPlane(planed);
            return;
        }
    }
    auto clipPlane = new osg::ClipPlane(index, planed);
    _node->addClipPlane(clipPlane);
    /* We have to update the clip plane in both nodes due to a shortcoming in
       the GL3 implementation of clipping planes in OpenSceneGraph (child nodes
       do not inherit clip planes from parents). */
    _circuit->addClipPlane(clipPlane);
}

void SubScene::clearClipPlanes()
{
    if (_staticClipping)
        return;

    const unsigned int count = _node->getNumClipPlanes();
    for (unsigned int i = 0; i < count; ++i)
    {
        /* Removal doesn't preserve the plane positions. Always remove the
           first plane. */
        _node->removeClipPlane(0u);
        _circuit->removeClipPlane(0u);
    }
}

void SubScene::_applyClipBox(const osg::BoundingBoxd& box)
{
    for (size_t i = 0; i < 6; ++i)
        _circuit->removeClipPlane(i);

    /* Composing the final clip planes and applying them to the scene. */
    for (size_t i = 0; i < 6; ++i)
    {
        const double value = (box.*constBoxSides[i])();
        /* OpenSceneGraph uses FLT_MAX for both float and double in the
           default constructor of BoundingBoxes */
        if (value != FLT_MAX && value != -FLT_MAX)
        {
            osg::Vec4d plane(0, 0, 0, -value);
            plane[i / 2] = 1.0;
            if (i % 2 != 0)
                plane *= -1;

            osg::ref_ptr<osg::ClipPlane> clipPlane(
                new osg::ClipPlane(i, plane));
            _circuit->addClipPlane(clipPlane);
            _node->addClipPlane(clipPlane);
        }
    }
    _node->setCullingActive(true);
    _circuit->setCullingActive(true);
}
}
}
}
