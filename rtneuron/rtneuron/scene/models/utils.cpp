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

#include "utils.h"

#include "data/NeuronMesh.h"
#include "util/vec_to_vec.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>

#include <osg/Vec4>

#include <boost/bind.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
namespace util
{
namespace
{
inline osg::Vec3 _center(const brain::Vector4f& sample)
{
    return osg::Vec3(sample[0], sample[1], sample[2]);
}
inline float _radius(const brain::Vector4f& sample)
{
    return sample[3] * 0.5;
}

struct Edge
{
    Edge(const unsigned int start_, const unsigned int end_)
        : start(std::min(start_, end_))
        , end(std::max(start_, end_))
    {
    }
    bool operator==(const Edge& edge) const
    {
        return edge.start == start && edge.end == end;
    }
    bool operator<(const Edge& edge) const
    {
        return start < edge.start || (start == edge.start && end < edge.end);
    }
    unsigned int start;
    unsigned int end;
};

/**
   Returns false if a triangle cannot be intersected by a region defined
   as the intersection of hemispaces defined by a plane list and returns
   true if it may or may not be intersected.

   This is an approximate intersection test based on the SAT theorem that
   only tests the separating axis defined by the plane list. Assuming
   that triangles are small relative to the plane region, this should be not
   a problem for effective clipping. */
bool _mayIntersect(const Planes& planes, const osg::Vec3& a, const osg::Vec3& b,
                   const osg::Vec3& c)
{
    const osg::Vec4 v1(a, 1.0);
    const osg::Vec4 v2(b, 1.0);
    const osg::Vec4 v3(c, 1.0);
    for (const auto& plane : planes)
        if (v1 * plane < 0 && v2 * plane < 0 && v3 * plane < 0)
            return false;
    return true;
}

/**
   The same as the function above but for lines
*/
bool _mayIntersect(const Planes& planes, const osg::Vec3& a, const osg::Vec3& b)
{
    const osg::Vec4 v1(a, 1.0);
    const osg::Vec4 v2(b, 1.0);
    for (const auto& plane : planes)
        if (v1 * plane < 0 && v2 * plane < 0)
            return false;
    return true;
}

bool _clipByPlanes(const osg::Vec3* vertices, const Indices& indices,
                   const osg::PrimitiveSet::Mode mode, const Planes& planes,
                   size_t index)
{
    if (mode == osg::PrimitiveSet::TRIANGLES ||
        mode == osg::PrimitiveSet::TRIANGLE_STRIP)
    {
        return _mayIntersect(planes, vertices[indices[index]],
                             vertices[indices[index + 1]],
                             vertices[indices[index + 2]]);
    }
    if (mode == osg::PrimitiveSet::LINES)
    {
        return _mayIntersect(planes, vertices[indices[index]],
                             vertices[indices[index + 1]]);
    }
    /* Can't use LB_UNIMPLEMENTED with free functions */
    std::cerr << "Unimplemented " << __FILE__ << ':' << __LINE__ << std::endl;
    abort();
}

bool _clipAxon(const uint16_t* sections,
               const brain::neuron::Morphology& morphology,
               const Indices& indices, const osg::PrimitiveSet::Mode mode,
               size_t index)
{
    if (mode == osg::PrimitiveSet::TRIANGLES ||
        mode == osg::PrimitiveSet::TRIANGLE_STRIP)
    {
        unsigned int secIds[3] = {sections[indices[index]],
                                  sections[indices[index + 1]],
                                  sections[indices[index + 2]]};
        /* The condition evaluated here is complementary to the condition
           used in NeuronSkeleton::_postProcessTriangle to decide if a
           triangle maps to the soma. This way both the result of hard axon
           clipping and soft clipping based on culling look the same. */
        const auto& sectionTypes = morphology.getSectionTypes();
        for (unsigned int i = 0; i != 3; ++i)
        {
            if (sectionTypes[secIds[i]] != brain::neuron::SectionType::axon)
                return true;
        }
    }
    else
    {
        /* For the moment clipping lines by section is not needed because
           the axon be removed at construction time. */

        /* Can't use LB_UNIMPLEMENTED with free functions */
        std::cerr << "Unimplemented " << __FILE__ << ':' << __LINE__
                  << std::endl;
        abort();
    }
    return false;
}

template <typename T>
void _clipPrimitive(Indices& indices, size_t& length,
                    const osg::PrimitiveSet::Mode mode, bools& referenced,
                    const T& clippingFunctor)
{
    std::vector<unsigned int> newIndices;
    newIndices.reserve(length);

    if (mode == osg::PrimitiveSet::TRIANGLES)
    {
        for (size_t i = 0; i < length; i += 3)
        {
            if (clippingFunctor(i))
            {
                for (unsigned int k = 0; k < 3; ++k)
                {
                    referenced[indices[i + k]] = true;
                    newIndices.push_back(indices[i + k]);
                }
            }
        }
    }
    else if (mode == osg::PrimitiveSet::TRIANGLE_STRIP)
    {
        bool substripStart = true;
        for (size_t i = 0; i < length - 2; ++i)
        {
            if (clippingFunctor(i))
            {
                /* Visible triangle */
                if (substripStart)
                {
                    if (newIndices.size() != 0)
                    {
                        /* Connecting with previous triangle. We need to
                           check the winding of the next triangle in the
                           original index array and the expected winding
                           for the next triangle in the new one.
                           If the widings don't match, we have to add an
                           extra degenerate triangle. */
                        newIndices.push_back(*newIndices.rbegin());
                        newIndices.push_back(indices[i]);
                        if (newIndices.size() % 2 != (i % 2))
                            newIndices.push_back(indices[i]);
                    }
                    else
                    {
                        /* If there's no previous triangle, we still have
                           to ensure that the original winding is preserved */
                        if (i % 2)
                            newIndices.push_back(indices[i]);
                    }
                    for (unsigned int k = 0; k < 3; ++k)
                    {
                        referenced[indices[i + k]] = true;
                        newIndices.push_back(indices[i + k]);
                    }
                    substripStart = false;
                }
                else
                {
                    referenced[indices[i + 2]] = true;
                    newIndices.push_back(indices[i + 2]);
                }
            }
            else
            {
                substripStart = true;
            }
        }
    }
    else if (mode == osg::PrimitiveSet::LINES)
    {
        for (size_t i = 0; i < length; i += 2)
        {
            if (clippingFunctor(i))
            {
                referenced[indices[i]] = true;
                newIndices.push_back(indices[i]);
                referenced[indices[i + 1]] = true;
                newIndices.push_back(indices[i + 1]);
            }
        }
    }
    else
    {
        /* Can't use LB_UNIMPLEMENTED with free functions */
        std::cerr << "Unimplemented " << __FILE__ << ':' << __LINE__
                  << std::endl;
        abort();
    }

    /* Computing the final size of per vertex arrays and annotating the
       index of each vertex to remove together with the number of indices
       that have been removed before that one */
    std::vector<unsigned int> offsets(referenced.size());
    unsigned int removed = 0;
    for (size_t i = 0; i < referenced.size(); ++i)
    {
        if (!referenced[i])
            ++removed;
        else
            offsets[i] = removed;
    }
    for (size_t i = 0; i != newIndices.size(); ++i)
    {
        unsigned int& index = newIndices[i];
        index -= offsets[index];
    }

    /* Replacing old index array with the new one with the just size */
    length = newIndices.size();
    indices.reset(new unsigned int[length]);
    memcpy(indices.get(), &newIndices[0], sizeof(unsigned int) * length);
}
}

const float IN_CORE_FIRST_ORDER_SECTION_LENGTH = 2.0f;

size_t SectionStartPoint::findSampleAfterSectionStart(
    const brain::Vector4fs& samples, const brain::floats& relativeDistances,
    const float sectionLength, float& correctedInitialRadius) const
{
    size_t index = 0;
    /* Find the first sample whose relative position is greater than
       the position of the farthest triangles in the soma mesh. */
    while (index < samples.size() - 1 &&
           relativeDistances[index + 1] <= position)
        ++index;

    const auto& start = samples[index];
    const auto& end = samples[index + 1];
    /* Interpolating the startRadius based on the position at which
       the startPoint is located. */
    const float length = (_center(end) - _center(start)).length();
    float t = 0;
    if (length != 0)
        t = (position - relativeDistances[index - 1]) /
            (length / sectionLength);
    correctedInitialRadius = _radius(start) * (1 - t) + _radius(end) * t;
    return index + 1;
}

bool isInCore(const brain::neuron::Morphology& morphology,
              const uint16_t* sections, const float* positions,
              const uint32_t index)
{
    if (sections[index] == 0)
        return true;

    if (positions[index] >= 0.25f)
        return false;

    // The original code using section objects has become quite slow, using
    // the raw array instead for performance.
    if (morphology.getSections()[sections[index]][1] != 0)
        return false;
    const auto& section = morphology.getSection(sections[index]);
    return section.getLength() * positions[index] <
           IN_CORE_FIRST_ORDER_SECTION_LENGTH;
}

void sortSections(brain::neuron::Sections& sections)
{
    std::sort(sections.begin(), sections.end(),
              [](const brain::neuron::Section& a,
                 const brain::neuron::Section& b) {
                  return a.getID() < b.getID();
              });
}

size_t numberOfPoints(const brain::neuron::Sections& sections)
{
    size_t numPoints = 0;
    for (const auto& section : sections)
        numPoints += section.getNumSamples();
    return numPoints;
}

brain::floats computeRelativeDistances(const brain::Vector4fs& samples,
                                       float totalLength)
{
    brain::floats distances;
    auto current = samples.cbegin();
    auto previous = current++;
    distances.push_back(0);
    float accumLength = 0;
    for (; current != samples.end(); previous = current++)
    {
        accumLength += (_center(*current) - _center(*previous)).length();
        distances.push_back(accumLength / totalLength);
    }
    distances.back() = 1;
    return distances;
}

void removeZeroLengthSegments(brain::Vector4fs& samples,
                              brain::floats& relativeDistances)
{
    assert(samples.size() == relativeDistances.size() && samples.size() > 1);
    /* Filteringe 0 length segments */
    auto i = samples.begin();
    auto j = relativeDistances.begin();
    while (j < --relativeDistances.end())
    {
        if (*j == *(j + 1))
        {
            i = samples.erase(i);
            j = relativeDistances.erase(j);
        }
        else
        {
            ++j;
            ++i;
        }
    }
}

#define IS_IN_CORE(i) isInCore(morphology, sections, positions, indices[i])
SectionStartPointsMap extractCellCoreSectionStartPoints(
    const brain::neuron::Morphology& morphology, const NeuronMesh& mesh)
{
    SectionStartPointsMap points;

    /* Counting the number of polygons the core consists of an creating
       the translation table. */
    const uint32_t* indices;
    size_t indexCount;
    unsigned int step;
    auto triangles = mesh.getTriangles();
    auto strip = mesh.getTriangleStrip();
    if (triangles && !triangles->empty())
    {
        step = 3;
        indices = triangles->data();
        indexCount = triangles->size();
    }
    else if (strip && !strip->empty())
    {
        step = 1;
        indices = strip->data();
        indexCount = strip->size();
    }
    else
    {
        /* No geometry to process found */
        return points;
    }
    const float* positions = mesh.getVertexRelativeDistances()->data();
    const uint16_t* sections = mesh.getVertexSections()->data();

    std::set<Edge> edges;
    std::set<Edge> repeated;

    for (unsigned int i = 0; i < indexCount - 2; i += step)
    {
        /* Skiping degenerated triangles when we are in a strip */
        if (step == 1 &&
            (indices[i + 2] == indices[i] || indices[i] == indices[i + 1] ||
             indices[i + 1] == indices[i + 2]))
            continue;

        /* A triangle is visible if any of its vertices is visible */
        if (IS_IN_CORE(i) || IS_IN_CORE(i + 1) || IS_IN_CORE(i + 2))
        {
            /* Storing edges in the edges set and removing those that
               are repeated. */
            for (unsigned int j = 0, k = 2; j < 3; k = j++)
            {
                const Edge edge(indices[i + j], indices[i + k]);
                if (edges.find(edge) == edges.end() &&
                    repeated.find(edge) == repeated.find(edge))
                {
                    edges.insert(edge);
                }
                else
                {
                    edges.erase(edge);
                    repeated.insert(edge);
                }
            }
        }
    }

    /* Extracting the vertices from the border edges and classifying them
       into bins depending on their sections. */
    typedef std::map<uint16_t, std::set<unsigned int>> PerSectionVertices;
    PerSectionVertices perSectionVertices;
    for (const auto& edge : edges)
    {
        if (sections[edge.start] != 0)
            perSectionVertices[sections[edge.start]].insert(edge.start);
        if (sections[edge.end] != 0)
            perSectionVertices[sections[edge.end]].insert(edge.end);
    }
    const auto* vertices = mesh.getVertices()->data();
    for (auto sectionVertices : perSectionVertices)
    {
        osg::Vec3 center;
        float position = 0;
        const auto& vertexIndices = sectionVertices.second;
        for (const auto j : vertexIndices)
        {
            center += vec_to_vec(vertices[j]);
            position += positions[j];
        }
        position /= vertexIndices.size();
        center /= vertexIndices.size();
        points[sectionVertices.first] = SectionStartPoint{center, position};
    }
    return points;
}
#undef IS_IN_CORE

brain::neuron::Section mainBranchChildSection(
    const brain::neuron::Section& section)
{
    const auto& children = section.getChildren();

    float bestDiff = std::numeric_limits<float>::max();
    float bestCosAngle = -1;
    const auto last = section[-1];
    const auto lastDiameter = last[3];
    osg::Vec3 lastAxis = _center(last) - _center(section[-2]);
    lastAxis.normalize();

    const brain::neuron::Section* best = nullptr;
    for (const auto& child : children)
    {
        const auto& first = child[0];
        const auto diff = std::abs(first[3] - lastDiameter);
        if (diff < bestDiff)
        {
            osg::Vec3 v = _center(child[1]) - _center(first);
            v.normalize();
            bestCosAngle = v * lastAxis;
            bestDiff = diff;
            best = &child;
        }
        else if (diff == bestDiff)
        {
            osg::Vec3 v = _center(child[1]) - _center(first);
            v.normalize();
            const float a = v * lastAxis;
            if (a > bestCosAngle)
            {
                bestCosAngle = a;
                best = &child;
            }
        }
    }
    if (best == 0)
        return section;
    return *best;
}

void clipPrimitive(Indices& indices, size_t& length,
                   const osg::PrimitiveSet::Mode mode,
                   const osg::Vec3* vertices, const Planes& planes,
                   bools& referenced)
{
    return _clipPrimitive(indices, length, mode, referenced,
                          boost::bind(_clipByPlanes, vertices, indices, mode,
                                      boost::cref(planes), _1));
}

void clipAxonFromPrimitive(Indices& indices, size_t& length,
                           const osg::PrimitiveSet::Mode mode,
                           const uint16_t* sections,
                           const brain::neuron::Morphology& morphogy,
                           bools& referenced)
{
    return _clipPrimitive(indices, length, mode, referenced,
                          boost::bind(_clipAxon, sections,
                                      boost::cref(morphogy), indices, mode,
                                      _1));
}
}
}
}
}
}
