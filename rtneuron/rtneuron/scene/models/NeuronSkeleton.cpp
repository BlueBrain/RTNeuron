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

#include <boost/unordered_map.hpp>
#ifndef NDEBUG
#include <boost/unordered_set.hpp>
#endif

#include "NeuronSkeleton.h"
#include "data/NeuronMesh.h"
#include "util/triangleClassifiers.h"
#include "util/vec_to_vec.h"
#include "utils.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

#include <osg/io_utils>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
namespace
{
const float MAX_SEPARATION_STEP2 = 0.4f;
const float MAX_LENGTH_STEP2 = 16;
const float DEFAULT_MAX_LENGTH2 = 64;

brain::Vector3f _center(const brain::Vector4f& sample)
{
    return brain::Vector3f(sample[0], sample[1], sample[2]);
}

brain::neuron::Section mainBranchChild(const brain::neuron::Section& section)
{
    assert(section.getNumSamples() > 1);
    const auto& end = section[-1];
    auto endAxis = _center(section[-1]) - _center(section[-2]);
    endAxis.normalize();

    float bestScore = -1;
    brain::neuron::Section bestSection = section; /* No default constructor
                                                     avaiable */
    auto children = section.getChildren();
    for (auto child : children)
    {
        const auto& start = child[0];
        auto v = _center(child[1]) - _center(start);
        v.normalize();

        const float score =
            dot(v, endAxis) * 0.5 +
            (1 - (std::abs(start[3] - end[3]) / std::max(start[3], end[3]))) *
                0.5;

        if (score > bestScore)
        {
            bestScore = score;
            bestSection = child;
        }
    }
    return bestSection;
}

const struct SkeletonParameters
{
public:
    SkeletonParameters()
    {
        const char* string = ::getenv("RTNEURON_MAX_OFFAXIS_SEPARATION");
        char* endptr;
        double value;
        if (string && (value = strtod(string, &endptr)) > 0 && *endptr == '\0')
            maxSeparation2 = value * value;
        else
            maxSeparation2 = 6;

        string = ::getenv("RTNEURON_MAX_CAPSULE_LENGTH");
        if (string && (value = strtod(string, &endptr)) > 0 && *endptr == '\0')
            maxLength2 = value * value;
        else
            maxLength2 = DEFAULT_MAX_LENGTH2;
    }
    double maxSeparation2;
    double maxLength2;
} s_skeletonParameters;

/**
    Computes the square euclidean distance of a line to a point
    @param t The direction vector (must be normalized).
    @param p Any point in the line.
    @param v The point to measure.
*/
inline float _distance2(const brain::Vector3f& t, const brain::Vector3f& o,
                        const brain::Vector3f& v)
{
    /* An arbitrary vector joining v with the line */
    const auto p = v - o;
    /* Length of the projection of v on the line */
    const float p_t = p.dot(t);
    /* Square length of p */
    const float p_p = p.dot(p);
    /* Square euclidean distance between the line and v */
    return p_p - p_t * p_t;
}

inline float _distance2(const osg::Vec3f& t, const osg::Vec3f& o,
                        const osg::Vec3f& v)
{
    /* An arbitrary vector joining v with the line */
    osg::Vec3f p = v - o;
    /* Length of the projection of v on the line */
    const float p_t = p * t;
    /* Square length of p */
    const float p_p = p * p;
    /* Square euclidean distance between the line and v */
    return p_p - p_t * p_t;
}

template <class Vector_Type>
inline float _distance2(const Vector_Type& p, const Vector_Type& q)
{
    const Vector_Type t = p - q;
    return t.dot(t);
}

Skeleton::SectionPortions _extractSectionPortions(
    const brain::neuron::Section& section, const double maxSeparation2,
    const double maxLength2)
{
    Skeleton::SectionPortions portions;

    auto samples = section.getSamples();
    /* The current segment in the iteration has previous as the start point
       and current as the end point. */
    auto current = samples.begin();
    decltype(current) previous = current++;
    uint16_t sampleIndex = 1;

    while (current != samples.end())
    {
        /* Starting a new portion */
        auto start = previous;
        const uint16_t startIndex = sampleIndex - 1;

        brain::Vector3f direction;
        float length;
        do
        {
            direction = _center(*current) - _center(*start);
            length = direction.length();
            ++sampleIndex;
            previous = current++;
        } while (length < 10e-7 && current != samples.end());

        if (length == 0 && current == samples.end())
        {
            /* We've found a 0 length segment at the end of a section
               (should we report this in debug mode?) and it was starting a new
               portion. We just add one to the last portion and return */
            assert(!portions.empty());
            ++portions.back().second;
            return portions;
        }
        assert(length != 0);
        direction = direction / length;

        for (; current != samples.end(); previous = current++, ++sampleIndex)
        {
            /* Stopping at the current segment if the length of the
               portion is above the threshold */
            if (_distance2(_center(*start), _center(*current)) > maxLength2)
                break;

            /* Stoping at the current sample if the current sample or the
               previous one are diverging from the initial segment line in
               more than a user given distance.
               This cannot happen with the first sample at a section start. */
            if (_distance2(direction, _center(*start), _center(*previous)) >
                    maxSeparation2 ||
                _distance2(direction, _center(*start), _center(*current)) >
                    maxSeparation2)
            {
                break;
            }
        }
        /* Storing a new section portion defined from
           startSegmentID to segmentID - 1 */
        portions.push_back(std::make_pair(startIndex, sampleIndex - 1));
    }
    return portions;
}

class CapsuleEnlarger
{
public:
    osg::Vec3f& start;
    osg::Vec3f& end;
    float& width;

    CapsuleEnlarger(osg::Vec3f& start_, osg::Vec3f& end_, float& width_,
                    const osg::Vec3f& axis)
        : start(start_)
        , end(end_)
        , width(width_)
        , _size(0)
        , _axis(axis)
    {
    }

    CapsuleEnlarger(osg::Vec3f& start_, osg::Vec3f& end_, float& width_)
        : start(start_)
        , end(end_)
        , width(width_)
        , _size(0)
    {
        _axis = end - start;
        _axis.normalize();
    }

    void addVertex(const osg::Vec3& v)
    {
        assert(_size < 3);
        _vertices[_size++] = v;
    }

    void solve()
    {
        float width2 = width * width;
        /* Computing The maximum squared euclidean distance of the triangle
           vertices to the skeleton portion. */
        /* This is the distance to the axis of each vertex */
        float distanceToAxis2[3];
        float distance2 = 0;
        for (unsigned int j = 0; j < _size; ++j)
        {
            const float d = _distance2(_axis, start, _vertices[j]);
            distanceToAxis2[j] = d;
            distance2 = std::max(distance2, d);
        }

        if (width2 < distance2)
        {
            width2 = distance2;
            width = sqrtf(distance2);
        }

        /* Now we check the distance from the vertices to the segment edges.
           This time we will lengthen the capsule to include the vertices
           if necessary. */
        float startOffset = 0, endOffset = 0;
        for (unsigned int j = 0; j < _size; ++j)
        {
            const osg::Vec3f toStart = (_vertices[j] - start);
            float projectedDistance = toStart * _axis;
            if (projectedDistance < 0 && width2 < toStart * toStart)
            {
                /*          _--W--------------
                     V..../    |
                     |   |:    |  capsule
                     |  | :    |
                     |  | :    |
                    -V'-+-Q----P-------axis---
                    offset = d(V', P) - d(Q, P)
                    d(V', P) = projectedDistance
                    d(Q, P) = sqrt(d(W, P)^2 - d(V, axis)^2)
                */
                const float offset = fabs(projectedDistance) -
                                     sqrtf(width2 - distanceToAxis2[j]);
                startOffset = std::max(startOffset, offset);
            }
            const osg::Vec3f toEnd = (_vertices[j] - end);
            projectedDistance = toEnd * _axis;
            if (width2 < toEnd * toEnd && projectedDistance > 0)
            {
                const float offset =
                    projectedDistance - sqrtf(width2 - distanceToAxis2[j]);
                endOffset = std::max(endOffset, offset);
            }
        }
        if (startOffset != 0)
            start -= _axis * startOffset;
        if (endOffset != 0)
            end += _axis * endOffset;
    }

private:
    unsigned int _size;
    osg::Vec3 _vertices[3];
    osg::Vec3 _axis;
};
}

/*
  Constructors/destructor
*/

NeuronSkeleton::NeuronSkeleton(const NeuronSkeleton& skeleton)
    : Skeleton(skeleton)
{
}

NeuronSkeleton::NeuronSkeleton(const brain::neuron::Morphology& morphology,
                               const NeuronParts parts)
    : Skeleton()
{
#ifdef USE_CUDA
    if (s_useSharedMemory && parts != NEURON_PARTS_FULL)
        /* The data structures for the shared memory mask reduction variant of
           VFC require all the sections IDs to be a single contiguous
           block. For regular morphologies the axon ids are between the soma
           id (0) and the dendrite ones. */
        LBUNIMPLEMENTED;
#endif

    using S = brain::neuron::SectionType;
    const auto& soma = morphology.getSoma();
    const auto& allSectionTypes =
        std::vector<S>({S::axon, S::dendrite, S::apicalDendrite});

    /* The GPU algorithm uses section IDs as indices in the output visibility
       array. In order to support clipping of skeletons (based on section
       types or clipping planes) we have two options:
       - Keep the output arrays at full skeleton size and use the
         original section ids.
       - Remap the the section ids to take into account the sections that
         get removed.
       While the first option is more space efficient, it makes much more
       difficult to post process the primitives to obtain the per capsule
       ranges because a section ID translation table needs to be
       maintained. For that reason, the current implementation uses option
       two. Accordinly, the output section count is always the max section ID
       present in the morphology. */
    unsigned int maxID = 0;
    const auto ids = morphology.getSectionIDs(allSectionTypes);
    for (const auto id : ids)
        maxID = std::max(id, maxID);
    _sectionCount = maxID + 1;

    brain::neuron::Sections sections;
    switch (parts)
    {
    case NEURON_PARTS_SOMA:
        break;
    case NEURON_PARTS_SOMA_DENDRITES:
        sections = morphology.getSections({S::dendrite, S::apicalDendrite});
        break;
    case NEURON_PARTS_FULL:
        sections = morphology.getSections(allSectionTypes);
        break;
    }
    util::sortSections(sections);
    _extract(soma, sections);

    _computeBranchOrders(morphology);
}

/*
  Member functions
*/
void NeuronSkeleton::_extract(const brain::neuron::Soma& soma,
                              const brain::neuron::Sections& sections,
                              const unsigned maxPortionsPerSection)
{
    /* Allocating the portions per section array (+ 1 for the soma) */
    _portionCounts.reset(new unsigned short[_sectionCount]);
    memset(_portionCounts.get(), 0, sizeof(unsigned short) * _sectionCount);
    _portionCounts[0] = 1; /* Section 0 again expected to be the soma */

    /* Dividing sections into portions according to the max length and
       separation criteria. The soma is excluded from here. */
    Portions portions = _extractPortions(sections, maxPortionsPerSection);

    /* Allocating space for the skeleton arrays */
    allocHostArrays(&portions);

    /* Soma is required to be section 0 */
    _addSomaCapsule(soma);

    size_t blockIndex = 0;
    size_t accumulatedSections = 1;
    size_t index = 1;
    size_t portionIndex = 0;

    /* Storing the information of each portion in the arrays. */
    auto sectionItr = sections.begin();
    auto samples = sectionItr->getSamples();
    auto relativeDistances =
        util::computeRelativeDistances(samples, sectionItr->getLength());

    for (const auto& i : portions)
    {
        const auto sectionID = i.first;
        const auto portion = i.second;
        bool newSection = false;
        while (sectionID != sectionItr->getID())
        {
            /* Jumping to next section. There must be one because this
               portion must belong to some section that hasn't been
               processed so far. */
            ++sectionItr;
            assert(sectionItr != sections.end());
            newSection = true;
        }
        const auto& section = *sectionItr;
        if (newSection)
        {
            portionIndex = 0;
            samples = section.getSamples();
            relativeDistances =
                util::computeRelativeDistances(samples, section.getLength());
        }
        _addCapsule(portion, samples, relativeDistances, index, portionIndex,
                    section.getID(), newSection, blockIndex,
                    accumulatedSections);
        ++index;
        ++portionIndex;
    }
    /* Assigning the final count of accumulated sections and section starts */
    assert(portions.size() > 0);
#ifdef USE_CUDA
    if (s_useSharedMemory)
    {
        ++accumulatedSections; /* Last section needs to be counted here. */
        _accumSectionsPerBlock[_blockCount] = accumulatedSections;
        _perBlockSectionsStarts[accumulatedSections] = 0;
        LBASSERTINFO(accumulatedSections + 1 == _sectionStartsLength,
                     (accumulatedSections + 1) << ' ' << _sectionStartsLength);
    }
#endif
}

Skeleton::Portions NeuronSkeleton::_extractPortions(
    const brain::neuron::Sections& sections,
    const unsigned int maxPortionsPerSection)
{
    Portions portions;
    for (const auto& section : sections)
    {
        SectionPortions sectionPortions;
        double maxSeparation2 = s_skeletonParameters.maxSeparation2;
        double maxLength2 = s_skeletonParameters.maxLength2;
        do
        {
            sectionPortions =
                _extractSectionPortions(section, maxSeparation2, maxLength2);
            maxSeparation2 += MAX_SEPARATION_STEP2;
            maxLength2 += MAX_LENGTH_STEP2;
        } while (sectionPortions.size() > maxPortionsPerSection);

        const auto id = section.getID();
        for (const auto& p : sectionPortions)
            portions.insert(std::make_pair(id, p));
        _portionCounts[id] = sectionPortions.size();
    }
    return portions;
}

void NeuronSkeleton::_addSomaCapsule(const brain::neuron::Soma& soma)
{
    _ends[0] = _starts[0] = vec_to_vec(soma.getCentroid());
    /* The soma width will be corrected later, using the mesh and the
        vertex mapping */
    _widths[0] = soma.getMaxRadius();
    _sections[0] = 0;
    _portions[0] = 0;
    _startPositions[0] = 0;
    _endPositions[0] = 1;
#ifdef USE_CUDA
    if (s_useSharedMemory)
    {
        _firstBlockSection[0] = 0;
        _perBlockSectionsStarts[0] = 0;
        _accumSectionsPerBlock[0] = _accumSectionsPerBlock[_blockCount] = 0;
    }
#endif
}

void NeuronSkeleton::_addCapsule(const Portion& portion,
                                 const brion::Vector4fs& samples,
                                 const brion::floats& relativeDistances,
                                 const size_t index, const size_t portionIndex,
                                 const size_t sectionID,
                                 const bool sectionStart LB_UNUSED,
                                 size_t& blockIndex LB_UNUSED,
                                 size_t& accumulatedSections LB_UNUSED)
{
#ifdef USE_CUDA
    if (s_useSharedMemory)
    {
        /* This only works if as capsules are added, sections are
           exhaustively enumerated and in order. Some sections can be
           skipped at the beginning or at the end, but no gap is allowed
           in between. */
        if (sectionStart)
        {
            if (index != 1)
                ++accumulatedSections;
            _perBlockSectionsStarts[accumulatedSections] =
                index % cuda::CULL_BLOCK_SIZE;
        }

        if (index % cuda::CULL_BLOCK_SIZE == 0)
        {
            /* Starting new block. */
            ++blockIndex;
            _firstBlockSection[blockIndex] = sectionID;
            if (!sectionStart)
                ++accumulatedSections;
            /* The current section is between two blocks. */
            _perBlockSectionsStarts[accumulatedSections] = 0;
            _accumSectionsPerBlock[blockIndex] = accumulatedSections;
            if (!sectionStart)
            {
                _accumSectionsPerBlock[blockIndex - 1] |= 0x4000;
                _accumSectionsPerBlock[blockIndex] |= 0x8000;
            }
        }
    }
#endif
    const auto first = portion.first;
    const auto last = portion.second;
    const auto start = _center(samples[first]);
    const auto end = _center(samples[last]);
    auto axis = end - start;
    axis.normalize();

    float maxRadius = 0;
    float maxDistance2 = 0;
    for (uint16_t j = first; j <= last; ++j)
    {
        const auto& sample = samples[j];
        maxRadius = std::max(maxRadius, sample[3]);
        const float distance = _distance2(axis, start, _center(sample));
        maxDistance2 = std::max(maxDistance2, distance);
    }
    _starts[index] = vec_to_vec(start);
    _ends[index] = vec_to_vec(end);
    _widths[index] = maxRadius + sqrtf(maxDistance2);
    _sections[index] = sectionID;
    _portions[index] = portionIndex;
    _startPositions[index] = relativeDistances[first];
    _endPositions[index] = relativeDistances[last];
}

void NeuronSkeleton::_computeBranchOrders(
    const brain::neuron::Morphology& morphology)
{
    /* Pre-computing the branch order of each section */
    /* morphology.getSections() is the low level access and includes the
       soma. */
    auto sectionCount = morphology.getSections().size();

    _branchOrders.reset(new boost::uint8_t[sectionCount]);
    /* The following assumes that soma is section 0 and section ids are
       exhaustive. */
    _branchOrders[0] = 0;

    typedef std::pair<const brain::neuron::Section, int> SectionOrder;
    std::deque<SectionOrder> sections;
    const auto firstSections = morphology.getSoma().getChildren();
    for (const auto& section : firstSections)
        sections.push_back(SectionOrder(section, 1));
    /* Depth first traversal of all sections */
    while (!sections.empty())
    {
        SectionOrder next = sections.front();
        sections.pop_front();
        /* Annotating the branch order of the section at the top of the
           queue. */
        assert(next.first.getID() < sectionCount);
        const auto order = next.second;
        _branchOrders[next.first.getID()] = order;
        /* Pushing the children into the queue */
        const auto& children = next.first.getChildren();
        const auto continuation = mainBranchChild(next.first);
        for (const auto& child : children)
            sections.push_back(
                SectionOrder(child, child == continuation ? order : order + 1));
    }
}

Skeleton::PortionRanges NeuronSkeleton::postProcess(const NeuronMesh& mesh)
{
    const uint16_t* sections = mesh.getVertexSections()->data();
    const float* positions = mesh.getVertexRelativeDistances()->data();
    const osg::Vec3f* vertices =
        reinterpret_cast<const osg::Vec3f*>(mesh.getVertices()->data());
    const uint32_t* strip = mesh.getTriangleStrip()->data();
    assert(strip != 0);
    const size_t length = mesh.getTriangleStripLength();

    return postProcess(vertices, sections, positions, strip,
                       osg::PrimitiveSet::TRIANGLE_STRIP, length);
}

Skeleton::PortionRanges NeuronSkeleton::postProcess(
    const osg::Vec3* vertices, const uint16_t* sections, const float* positions,
    const uint32_t* primitive, const osg::PrimitiveSet::Mode mode,
    const size_t size)
{
    /* The try catch clauses can be removed once no more old meshes are
       needed. */
    switch (mode)
    {
    case osg::PrimitiveSet::TRIANGLE_STRIP:
        try
        {
            return _postProcessStrip(vertices, sections, positions, primitive,
                                     size);
        }
        catch (...)
        {
            /* Retrying with the different triangle classification */
            triangleClassifier = classifyTriangleOld;
            return _postProcessStrip(vertices, sections, positions, primitive,
                                     size);
        }
    case osg::PrimitiveSet::TRIANGLES:
        try
        {
            return _postProcessTriangles(vertices, sections, positions,
                                         primitive, size);
        }
        catch (...)
        {
            /* Retrying with the different triangle classification */
            triangleClassifier = classifyTriangleOld;
            return _postProcessTriangles(vertices, sections, positions,
                                         primitive, size);
        }
    case osg::PrimitiveSet::LINES:
        return _postProcessLines(vertices, sections, positions, primitive,
                                 size);
    default:
        std::cerr << "Unimplemented: " << __FILE__ << ':' << __LINE__
                  << std::endl;
        abort();
    }
}

Skeleton::PortionRanges NeuronSkeleton::_postProcessLines(
    const osg::Vec3* vertices, const uint16_t* sections, const float* positions,
    const uint32_t* lines, const size_t size)
{
    PortionRanges portionRanges;

    portionRanges.length = _length;
    PortionRanges::Ranges& ranges = portionRanges.ranges;
    ranges.reset(new PortionRanges::Range[_length]);

    uint16_t currentSection = 0;
    unsigned int skeletonIndex = 0;

#ifndef NDEBUG
    uint16_t previous = LB_UNDEFINED_UINT16;
    boost::unordered_set<uint16_t> sectionsUsed;
#endif

    for (unsigned int i = 0; i < _length; ++i)
    {
        ranges[i] = std::make_pair(LB_UNDEFINED_UINT32, 0);
#ifndef NDEBUG
        if (previous != _sections[i])
        {
            previous = _sections[i];
            assert(sectionsUsed.find(previous) == sectionsUsed.end());
            sectionsUsed.insert(previous);
        }
#endif
    }

    for (size_t i = 0; i < size; i += 2)
    {
        /* Classifying the triangle into a section and portion and finding
           the absolute index of the portion in the skeleton array. */
        uint16_t section;

        if (sections[lines[i]] == 0 || sections[lines[i + 1]] == 0)
        {
            section = 0;
        }
        else
        {
            section = sections[lines[i]];
            /* Advancing to the section in the skeleton if needed. */
            assert(section >= currentSection);
            if (section > currentSection)
            {
                for (; _sections[skeletonIndex] < section; ++skeletonIndex)
                    assert(skeletonIndex < _length);
            }

            /* Searching the portion */
            float position = positions[lines[i]];
            position = std::min(1.0f, std::max(position, 0.0f));
            while (_startPositions[skeletonIndex] > position)
                --skeletonIndex;
            while (_endPositions[skeletonIndex] < position)
                ++skeletonIndex;
            assert(_startPositions[skeletonIndex] <= position &&
                   _endPositions[skeletonIndex] >= position);
            currentSection = section;
        }

        /* Enlarging the portion bounding box with with the vertices
           from the line belonging to the portion. */
        if (section == 0)
        {
            /* The soma is a special case */
            const osg::Vec3& center = _starts[0];
            const float distance =
                std::max((vertices[lines[i]] - center).length(),
                         (vertices[lines[i + 1]] - center).length());
            _widths[0] = std::max(distance, _widths[0]);
        }
        else
        {
            CapsuleEnlarger enlarger(_starts[skeletonIndex],
                                     _ends[skeletonIndex],
                                     _widths[skeletonIndex]);
            enlarger.addVertex(vertices[lines[i]]);
            enlarger.addVertex(vertices[lines[i + 1]]);
            enlarger.solve();
        }

        /* Updating index array with the start and end index of the current
           line segment. Realize that it doesn't matter if portion ranges
           overlap because they are eventually collapsed. */
        const boost::uint32_t start = i & ~0x1;
        const boost::uint32_t end = start + 1;
        ranges[skeletonIndex].first =
            std::min(ranges[skeletonIndex].first, start);
        ranges[skeletonIndex].second =
            std::max(ranges[skeletonIndex].second, end);
    }

    return portionRanges;
}

Skeleton::PortionRanges NeuronSkeleton::_postProcessTriangles(
    const osg::Vec3* vertices, const uint16_t* sections, const float* positions,
    const uint32_t* indices, const size_t size)
{
    PortionRanges portionRanges;

    portionRanges.length = _length;
    boost::shared_array<std::pair<boost::uint32_t, boost::uint32_t>>& ranges =
        portionRanges.ranges;
    ranges.reset(new std::pair<boost::uint32_t, boost::uint32_t>[_length]);

    for (unsigned int i = 0; i < _length; ++i)
        ranges[i] = std::make_pair(LB_UNDEFINED_UINT32, 0);

    uint16_t currentSection = 0;
    unsigned int skeletonIndex = 0;
    osg::Vec3 axis;

    for (size_t i = 0; i < size - 2; i += 3)
    {
        const uint32_t* triangle = &indices[i];

        _postProcessTriangle(vertices, sections, positions, triangle,
                             currentSection, skeletonIndex, axis);

        /* Updating index array */
        ranges[skeletonIndex].first =
            std::min(ranges[skeletonIndex].first, (boost::uint32_t)i);
        ranges[skeletonIndex].second =
            std::max(ranges[skeletonIndex].second, (boost::uint32_t)i + 2);
    }

    return portionRanges;
}

Skeleton::PortionRanges NeuronSkeleton::_postProcessStrip(
    const osg::Vec3* vertices, const uint16_t* sections, const float* positions,
    const uint32_t* strip, const size_t size)
{
    PortionRanges portionRanges;

    portionRanges.length = _length;
    boost::shared_array<std::pair<boost::uint32_t, boost::uint32_t>>& ranges =
        portionRanges.ranges;
    ranges.reset(new std::pair<boost::uint32_t, boost::uint32_t>[_length]);

    for (unsigned int i = 0; i < _length; ++i)
        ranges[i] = std::make_pair(std::numeric_limits<size_t>::max(), 0);

    uint16_t currentSection = 0;
    unsigned int skeletonIndex = 0;
    osg::Vec3 axis;
    bool inverted = false;
    for (size_t i = 0; i < size - 2; ++i, inverted = !inverted)
    {
        const uint32_t* triangle = &strip[i];
        if (triangle[0] == triangle[1] || triangle[1] == triangle[2] ||
            triangle[0] == triangle[2])
        {
            /* Degenerated triangle. Skipping */
            continue;
        }

        _postProcessTriangle(vertices, sections, positions, triangle,
                             currentSection, skeletonIndex, axis);

        /* Updating index array */
        const boost::uint32_t start = inverted ? i - 1 : i;
        boost::uint32_t end = i;
        for (; (end < size - 3 && (strip[end + 1] == strip[end + 2] ||
                                   strip[end + 1] == strip[end + 3] ||
                                   strip[end + 2] == strip[end + 3]));
             ++end)
            ;
        end += 2;
        ranges[skeletonIndex].first =
            std::min(ranges[skeletonIndex].first, start);
        ranges[skeletonIndex].second =
            std::max(ranges[skeletonIndex].second, end);
    }

    return portionRanges;
}

unsigned int NeuronSkeleton::_postProcessTriangle(
    const osg::Vec3* vertices, const uint16_t* sections, const float* positions,
    const uint32_t* triangle, uint16_t& currentSection,
    unsigned int& skeletonIndex, osg::Vec3& axis)
{
    /* Classifying the triangle into a section and portion and finding
       the absolute index of the portion in the skeleton array. */
    uint16_t section;
    /* If any vertex belongs to the soma then we assign the triangle
       directly to that section */
    if (sections[triangle[0]] == 0 || sections[triangle[1]] == 0 ||
        sections[triangle[2]] == 0)
    {
        section = 0;
        skeletonIndex = 0;
    }
    else
    {
        const unsigned int corner =
            triangleClassifier(sections, positions, triangle);
        section = sections[triangle[corner]];

        /** Temporary workaround to support the old triangle mapping,
            this should be an assert(section >= currentSection) once the
            old mapping is not used anymore. */
        if (section < currentSection)
            throw "retry"; /* The actual exception doesn't matter */

        /* Advancing to the section in the skeleton if needed.
           Notice that the mapping algorithm might leave very small
           sections near large branches with no vertices assigned. This
           means that, despite the strips are in order, the enumeration
           is not exhaustive. */
        if (section > currentSection)
        {
            for (; _sections[skeletonIndex] < section; ++skeletonIndex)
                assert(skeletonIndex < _length);
        }

        /* Searching the portion */
        float position = positions[triangle[corner]];
        /* Workaround for buggy vertex mappings. This code silences
           potential errors but it's difficult to provide a meaningful
           error message from here. */
        position = std::min(1.0f, std::max(position, 0.0f));
        const unsigned int oldPortion = skeletonIndex;
        while (_startPositions[skeletonIndex] > position)
            --skeletonIndex;
        while (_endPositions[skeletonIndex] < position)
            ++skeletonIndex;

        assert(_startPositions[skeletonIndex] <= position &&
               _endPositions[skeletonIndex] >= position);

        /* Storing the axis of the current portion */
        if (oldPortion != skeletonIndex)
        {
            const osg::Vec3f& start = _starts[skeletonIndex];
            const osg::Vec3f& end = _ends[skeletonIndex];
            axis = end - start;
            axis.normalize();
            currentSection = section;
        }
    }

    /* Enlarging the portion bounding box with
       with the vertices from the triangle belonging to the portion. */
    if (section == 0)
    {
        /* Soma is a special case */
        const osg::Vec3& center = _starts[0];
        const float distance =
            std::max((vertices[triangle[0]] - center).length(),
                     std::max((vertices[triangle[1]] - center).length(),
                              (vertices[triangle[2]] - center).length()));
        _widths[0] = std::max(distance, _widths[0]);
    }
    else
    {
        CapsuleEnlarger enlarger(_starts[skeletonIndex], _ends[skeletonIndex],
                                 _widths[skeletonIndex], axis);
        for (int j = 0; j < 3; ++j)
            enlarger.addVertex(vertices[triangle[j]]);
        enlarger.solve();
    }
    return skeletonIndex;
}

std::ostream& operator<<(std::ostream& out, const NeuronSkeleton& skeleton)
{
    int count = skeleton._length;
    int w = 3;
    std::stringstream strs[5];
    for (int i = 0; i < count; ++i)
        strs[0] << ' ' << std::setw(w) << skeleton._sections[i];
    for (int i = 0; i < count; ++i)
        strs[1] << ' ' << std::setw(w) << (int)skeleton._portions[i];

#ifdef USE_CUDA
    const size_t blockSize = cuda::CULL_BLOCK_SIZE;
    if (NeuronSkeleton::s_useSharedMemory)
    {
        for (size_t i = 0; i < (count + blockSize - 1) / blockSize; ++i)
        {
            strs[2] << ' ' << std::setw(w) << skeleton._firstBlockSection[i];
            if (i != (count + blockSize - 1) / blockSize - 1)
                strs[2] << std::setw((blockSize - 1) * (w + 1)) << ' ';
        }
        for (size_t i = 0; i < (count + blockSize - 1) / blockSize + 1; ++i)
        {
            strs[3] << ' ' << std::setw(w)
                    << (skeleton._accumSectionsPerBlock[i] & ~0xC000)
                    << (skeleton._accumSectionsPerBlock[i] & 0x8000 ? 'X' : '_')
                    << (skeleton._accumSectionsPerBlock[i] & 0x4000 ? 'X'
                                                                    : '_');
            if (i != (count + blockSize - 1) / blockSize)
                strs[3] << std::setw((blockSize - 1) * (w + 1) - 2) << ' ';
        }
        for (int blocks = 0, i = 0, fill = 0;
             (blocks - 1) * (int)blockSize +
                 skeleton._perBlockSectionsStarts[i] <
             count;
             ++i)
        {
            int start = skeleton._perBlockSectionsStarts[i];
            int end = skeleton._perBlockSectionsStarts[i + 1];
            if (end == 0)
            {
                ++blocks;
                end = blockSize;
            }
            strs[4] << std::setw(fill) << ' ' << std::setw(w) << start;
            fill = (end - start) * (w + 1) - w;
        }
    }
#endif
    const char* labels[] = {"Sections", "Portions", "Fst sec ", "Acc sec ",
                            "Sec strt"};
    bool finished = false;
    do
    {
        const unsigned int rows = NeuronSkeleton::s_useSharedMemory ? 5 : 2;
        for (unsigned int i = 0; i < rows; ++i)
        {
            out << labels[i] << ' ';
            for (int n = 0; n < 68 && !strs[i].eof(); ++n)
            {
                int c = strs[i].get();
                if (c != -1)
                    out << (char)c;
            }
            out << std::endl;
        }
        out << std::endl;
        finished = true;
        for (unsigned int i = 0; i < rows; ++i)
            finished = finished && strs[i].eof();
    } while (!finished);

    return out;
}
}
}
}
}
