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

#include "NeuronClippingImpl.h"

#include "../SceneImpl.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "render/Skeleton.h"
#include "scene/NeuronModel.h"
#include "scene/NeuronModelClipping.h"
#include "scene/SubScene.h"

namespace bbp
{
namespace rtneuron
{
namespace sceneops
{
/*
  Helper functions
*/
namespace
{
using namespace core;

typedef NeuronClipping::_Impl::FloatInterval FloatInterval;
typedef NeuronClipping::_Impl::Interval Interval;
typedef NeuronClipping::_Impl::Intervals Intervals;
typedef NeuronClipping::_Impl::SectionIntervals SectionIntervals;
typedef NeuronClipping::_Impl::PerSectionIntervals PerSectionIntervals;

void _joinIntervals(const PerSectionIntervals& input,
                    PerSectionIntervals& target)
{
    for (const auto& intervals : input)
    {
        const uint16_t section = intervals.first;
        for (const auto& interval : intervals.second)
            target[section].add(interval);
    }
}

void _cutIntervals(const PerSectionIntervals& input,
                   PerSectionIntervals& target)
{
    for (const auto& intervals : input)
    {
        const uint16_t section = intervals.first;
        auto sectionToCut = target.find(section);
        if (sectionToCut == target.end())
            continue;

        for (const auto& interval : intervals.second)
        {
            sectionToCut->second -= interval;
            /* If the intervals become empty, they are removed from the
               map */
            if (sectionToCut->second.empty())
            {
                target.erase(sectionToCut);
                /* No more intervals to cut for this section */
                break;
            }
        }
    }
}

template <class Archive>
void _save(const PerSectionIntervals& sectionIntervals, Archive& archive)
{
    const size_t size = sectionIntervals.size();
    archive& size;
    for (const auto& intervals : sectionIntervals)
    {
        archive& intervals.first;
        const size_t count = intervals.second.iterative_size();
        archive& count;
        for (const auto& interval : intervals.second)
        {
            const float start = boost::icl::lower(interval);
            const float end = boost::icl::upper(interval);
            archive& start& end;
        }
    }
}

template <class Archive>
void _load(PerSectionIntervals& sectionIntervals, Archive& archive)
{
    size_t sections = 0;
    archive& sections;
    for (size_t i = 0; i != sections; ++i)
    {
        uint16_t section = 0;
        size_t ranges = 0;
        archive& section& ranges;
        Intervals& intervals = sectionIntervals[section];
        for (size_t j = 0; j != ranges; ++j)
        {
            float start = 0, end = 0;
            archive& start& end;
            intervals.add(Interval(start, end));
        }
    }
}

void _expand(const PerSectionIntervals& allIntervals, uint16_ts& sections,
             floats& starts, floats& ends)
{
    for (const auto& intervals : allIntervals)
    {
        const uint16_t section = intervals.first;
        for (const auto& interval : intervals.second)
        {
            sections.push_back(section);
            starts.push_back(boost::icl::lower(interval));
            ends.push_back(boost::icl::upper(interval));
        }
    }
}
}

/*
   Helper classes
*/

struct NeuronClipping::_Impl::State
{
    GlobalClippingEnum _globalClip;
    PerSectionIntervals _clipIntervals;
    PerSectionIntervals _unclipIntervals;

    uint32_t _gid;

    SomaClippingEnum getSomaClipping() const
    {
        SomaClippingEnum result = SomaClipping::NO_OPERATION;
        /* Checking global clipping */
        switch (_globalClip)
        {
        case GlobalClipping::NONE:
            result = SomaClipping::UNCLIP;
            break;
        case GlobalClipping::ALL:
            result = SomaClipping::CLIP;
            break;
        default:;
        }

        /* Checking whether section 0 appears in either the clip or unclip maps
           (it can't appear in both by design). */
        if (_clipIntervals.find(0) != _clipIntervals.end())
            result = SomaClipping::CLIP;
        else if (_unclipIntervals.find(0) != _unclipIntervals.end())
            result = SomaClipping::UNCLIP;

        return result;
    }
};

class NeuronClipping::_Impl::ModelOperation : public core::NeuronModelClipping
{
public:
    ModelOperation(const StatePtr& state)
        : _state(state)
    {
    }

    void operator()(core::NeuronModel& model) const { model.softClip(*this); }
    void operator()(core::Skeleton& skeleton) const final
    {
        using namespace core;

        /* Global operations are applied first */
        switch (_state->_globalClip)
        {
        case GlobalClipping::ALL:
            skeleton.softClipAll();
            break;
        case GlobalClipping::ALL_BUT_SOMA:
        {
            skeleton.softClipAll();
            uint16_ts soma;
            soma.push_back(0);
            skeleton.softUnclip(soma);
            break;
        }
        case GlobalClipping::NONE:
            skeleton.softUnclipAll();
            break;
        default:;
        }

        /* And then finer-grained clipping that may have be requested
           afterwards.
           Clip and unclip intervals are disjoint by construction, and the
           set of capsules they affect is also disjoint by the definition of
           the operations. This implies that the order of the following
           operations if interchangable. However, unclipping is applied
           first, because if there are no masks, it becomes no operation. */
        uint16_ts sections;
        floats starts;
        floats ends;
        _expand(_state->_unclipIntervals, sections, starts, ends);
        skeleton.softUnclip(sections, starts, ends);

        sections.clear();
        starts.clear();
        ends.clear();
        _expand(_state->_clipIntervals, sections, starts, ends);
        skeleton.softClip(sections, starts, ends);
    }

    SomaClippingEnum getSomaClipping() const final
    {
        return _state->getSomaClipping();
    }

private:
    StatePtr _state;
};

/*
  Constructors/destructor
*/
NeuronClipping::_Impl::_Impl()
    : _gid(0)
    , _state(new State())

{
}

NeuronClipping::_Impl::~_Impl(){};

/*
  Member functions
*/
void NeuronClipping::_Impl::operator()(core::SubScene& subScene) const
{
    auto neuron = subScene.getNeurons().find(_gid);
    if (!neuron)
        return;
    neuron->applyPerModelOperation(ModelOperation(_state));
}

void NeuronClipping::_Impl::operator()(Scene::_Impl&) const
{
}

void NeuronClipping::_Impl::clip(const PerSectionIntervals& intervals)
{
    /* Removing the input intervals from the unclip intervals */
    _cutIntervals(intervals, _state->_unclipIntervals);

    /* If a global clip all has been enabled we don't need to do anything
       else. */
    if (_state->_globalClip == GlobalClipping::ALL)
        return;

    /* If "clip all but soma" has been enabled, we need to check if the
       soma is now going to be clipped. */
    if (_state->_globalClip == GlobalClipping::ALL_BUT_SOMA)
    {
        /* This code assumes that soma is section 0! */
        PerSectionIntervals::const_iterator soma = intervals.find(0);
        if (soma != intervals.end())
            _state->_clipIntervals[0].add(
                Interval(0, 1)); /* Soma range is always [0, 1] */
    }
    else
    {
        _joinIntervals(intervals, _state->_clipIntervals);
    }
}

void NeuronClipping::_Impl::unclip(const PerSectionIntervals& intervals)
{
    /* Removing the input intervals from the clip intervals */
    _cutIntervals(intervals, _state->_clipIntervals);

    /* Unclip intervals are updated only if a global unclip all has not
       been enabled.  */
    if (_state->_globalClip != GlobalClipping::NONE)
        _joinIntervals(intervals, _state->_unclipIntervals);
}

void NeuronClipping::_Impl::clipAll(const bool alsoSoma)
{
    _state->_globalClip =
        alsoSoma ? GlobalClipping::ALL : GlobalClipping::ALL_BUT_SOMA;
    /* Clearing fine-grained clip/unclip */
    _state->_clipIntervals.clear();
    _state->_unclipIntervals.clear();
}

void NeuronClipping::_Impl::unclipAll()
{
    _state->_globalClip = GlobalClipping::NONE;
    /* Clearing fine-grained clip/unclip */
    _state->_clipIntervals.clear();
    _state->_unclipIntervals.clear();
}

void NeuronClipping::_Impl::setNeuronGID(const uint32_t gid)
{
    _gid = gid;
}

template <class Archive>
void NeuronClipping::_Impl::serialize(Archive& archive,
                                      const unsigned int version)
{
    archive& boost::serialization::base_object<Scene::ObjectOperation::Impl>(
        *this);
    archive& _gid;
    archive & _state->_globalClip;
    boost::serialization::split_member(archive, *this, version);
}

template <class Archive>
void NeuronClipping::_Impl::load(Archive& archive, const unsigned int)
{
    _load(_state->_clipIntervals, archive);
    _load(_state->_unclipIntervals, archive);
}

template <class Archive>
void NeuronClipping::_Impl::save(Archive& archive, const unsigned int) const
{
    _save(_state->_clipIntervals, archive);
    _save(_state->_unclipIntervals, archive);
}

template void NeuronClipping::_Impl::serialize<net::DataOStreamArchive>(
    net::DataOStreamArchive&, const unsigned int);

template void NeuronClipping::_Impl::serialize<net::DataIStreamArchive>(
    net::DataIStreamArchive&, const unsigned int);

template void NeuronClipping::_Impl::load<net::DataIStreamArchive>(
    net::DataIStreamArchive& archive, const unsigned int version);

template void NeuronClipping::_Impl::save<net::DataOStreamArchive>(
    net::DataOStreamArchive& archive, const unsigned int version) const;
}
}
}

#include <boost/serialization/export.hpp>

BOOST_CLASS_EXPORT(bbp::rtneuron::sceneops::NeuronClipping::_Impl)
