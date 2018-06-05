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

#include "NeuronClipping.h"
#include "NeuronClippingImpl.h"

#include "data/Neuron.h"
#include "data/Neurons.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/synapses.h>

#include <vmmlib/vector.hpp>

namespace bbp
{
namespace rtneuron
{
namespace sceneops
{
using namespace core;

typedef NeuronClipping::_Impl::FloatInterval FloatInterval;
typedef NeuronClipping::_Impl::Interval Interval;
typedef NeuronClipping::_Impl::Intervals Intervals;
typedef NeuronClipping::_Impl::SectionIntervals SectionIntervals;
typedef NeuronClipping::_Impl::PerSectionIntervals PerSectionIntervals;

/*
  Helper functions
*/
namespace
{
inline osg::Vec3 _center(const brain::Vector4f& sample)
{
    return osg::Vec3(sample[0], sample[1], sample[2]);
}
bool _valid(const float x)
{
    return x >= 0 && x <= 1;
}

PerSectionIntervals _extractIntervals(const uint16_ts& sections,
                                      const floats& starts, const floats& ends)
{
    PerSectionIntervals intervals;
    if (sections.size() != starts.size() || sections.size() != ends.size())
    {
        throw std::invalid_argument(
            "Section range arrays have not the same size");
    }

    for (size_t i = 0; i != sections.size(); ++i)
    {
        if (starts[i] >= ends[i] || !_valid(starts[i]) || !_valid(ends[i]))
        {
            std::stringstream msg;
            msg << "Invalid section range: [" << starts[i] << ", " << ends[i]
                << "]";
            throw std::invalid_argument(msg.str().c_str());
        }

        if (sections[i] == 0)
            /* Soma ranges are internally converted into [0, 1] */
            intervals[sections[i]].add(Interval(0, 1));
        else
            intervals[sections[i]].add(Interval(starts[i], ends[i]));
    }
    return intervals;
}

PerSectionIntervals _unclipToPointsRanges(
    const std::vector<std::pair<uint16_t, float>>& points,
    const brain::neuron::Morphology& morphology)
{
    PerSectionIntervals intervals;

    std::unordered_map<uint16_t, float> ranges;
    for (const auto& point : points)
    {
        if (point.first == 0)
            continue; /* Assume section 0 is soma. */
        auto position = point.second;

        for (auto section = morphology.getSection(point.first);;
             section = section.getParent())
        {
            const auto id = section.getID();
            auto& range = ranges[id];
            if (range >= position)
                break;
            range = std::max(position, range);

            if (!section.hasParent())
                break;
            position = 1;
        }
    }

    for (const auto& range : ranges)
    {
        auto position = range.second;
        if (range.second != 0)
        {
            /* Clamping the position because there are synapses that report
               values slightly over 1 */
            position = std::max(0.f, std::min(position, 1.f));
            intervals[range.first].add(Interval(0, position));
        }
    }
    return intervals;
}

float _computeSectionRelativeDistance(
    const brain::neuron::Morphology& morphology, const uint16_t sectionID,
    const size_t segmentID, const float segmentDistance)
{
    const auto section = morphology.getSection(sectionID);
    if (segmentID >= section.getNumSamples() - 1)
        throw std::runtime_error("Segment index out of bounds");
    float accum = 0;
    for (size_t i = 0; i < segmentID; ++i)
        accum += (_center(section[i + 1]) - _center(section[i])).length();

    return (accum + segmentDistance) / section.getLength();
    return 0.5;
}
}

/*
  Constructors/destructor
*/
NeuronClipping::NeuronClipping()
    : _impl(new _Impl())
{
}

NeuronClipping::~NeuronClipping()
{
}

/*
  Member functions
*/
NeuronClipping& NeuronClipping::clip(const uint16_ts& sections,
                                     const floats& starts, const floats& ends)
{
    const auto intervals = _extractIntervals(sections, starts, ends);
    _impl->clip(intervals);
    return *this;
}

NeuronClipping& NeuronClipping::unclip(const uint16_ts& sections,
                                       const floats& starts, const floats& ends)
{
    const auto intervals = _extractIntervals(sections, starts, ends);
    _impl->unclip(intervals);
    return *this;
}

NeuronClipping& NeuronClipping::clipAll(const bool alsoSoma)
{
    _impl->clipAll(alsoSoma);
    return *this;
}

NeuronClipping& NeuronClipping::unclipAll()
{
    _impl->unclipAll();
    return *this;
}

NeuronClipping& NeuronClipping::unclipAfferentBranches(
    uint32_t gid, const brain::neuron::Morphology& morphology,
    const brain::Synapses& synapses)
{
    std::vector<std::pair<uint16_t, float>> points;
    const auto gids = synapses.postGIDs();
    const auto sections = synapses.postSectionIDs();
    const auto segments = synapses.postSegmentIDs();
    const auto distances = synapses.postDistances();
    for (size_t i = 0; i != synapses.size(); ++i)
    {
        auto section = sections[i];
        if (section == 0 or gids[i] != gid)
            continue;
        const auto distance =
            _computeSectionRelativeDistance(morphology, section, segments[i],
                                            distances[i]);
        points.push_back(std::make_pair(section, distance));
    }
    _impl->unclip(_unclipToPointsRanges(points, morphology));
    return *this;
}

NeuronClipping& NeuronClipping::unclipEfferentBranches(
    uint32_t gid, const brain::neuron::Morphology& morphology,
    const brain::Synapses& synapses)
{
    std::vector<std::pair<uint16_t, float>> points;
    const auto gids = synapses.preGIDs();
    const auto sections = synapses.preSectionIDs();
    const auto segments = synapses.preSegmentIDs();
    const auto distances = synapses.preDistances();
    for (size_t i = 0; i != synapses.size(); ++i)
    {
        auto section = sections[i];
        if (section == 0 or gids[i] != gid)
            continue;
        const auto distance =
            _computeSectionRelativeDistance(morphology, section, segments[i],
                                            distances[i]);
        points.push_back(std::make_pair(section, distance));
    }
    _impl->unclip(_unclipToPointsRanges(points, morphology));
    return *this;
}

bool NeuronClipping::accept(const Scene::Object& object) const
{
    try
    {
        const auto& o = object.getObject();
        const auto& gids = boost::any_cast<const uint32_ts&>(o);
        if (gids.size() != 1)
            return false;
        _impl->setNeuronGID(gids[0]);
        return true;
    }
    catch (const boost::bad_any_cast&)
    {
        return false;
    }
}

std::shared_ptr<Scene::ObjectOperation::Impl>
    NeuronClipping::getImplementation()
{
    return _impl;
};
}
}
}
