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

#ifndef RTNEURON_SCENEOPS_NEURONCLIPPING_IMPL_H
#define RTNEURON_SCENEOPS_NEURONCLIPPING_IMPL_H

#include "NeuronClipping.h"

#include "../scene/SceneObjectOperation.h"

#include <boost/icl/interval_set.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class SubScene;
class Skeleton;
}

namespace sceneops
{
namespace GlobalClipping
{
enum Enum
{
    NO_OPERATION,
    ALL,
    NONE,
    ALL_BUT_SOMA
};
}
typedef GlobalClipping::Enum GlobalClippingEnum;

class NeuronClipping::_Impl : public Scene::ObjectOperation::Impl
{
public:
    /*--- Public declarations ---*/

    struct State;
    typedef boost::icl::interval<float> FloatInterval;
    typedef FloatInterval::type Interval;
    typedef boost::icl::interval_set<float> Intervals;
    typedef std::pair<uint16_t, Intervals> SectionIntervals;
    typedef std::map<uint16_t, Intervals> PerSectionIntervals;

    /*--- Public member constructors/destructor ---*/

    _Impl();

    ~_Impl();

    /*--- Public member functions ---*/

    void operator()(core::SubScene& subScene) const;

    void operator()(Scene::_Impl& scene) const;

    void clip(const PerSectionIntervals& intervals);

    void unclip(const PerSectionIntervals& intervals);

    void clipAll(const bool alsoSoma);

    void unclipAll();

    void setNeuronGID(const uint32_t gid);

    template <class Archive>
    void serialize(Archive& archive, const unsigned int version);

    template <class Archive>
    void load(Archive& archive, const unsigned int);

    template <class Archive>
    void save(Archive& archive, const unsigned int) const;

private:
    /*--- Private declarations ---*/
    class ModelOperation;

    /*--- Private member attributes ---*/

    using StatePtr = std::shared_ptr<State>;

    uint32_t _gid;
    StatePtr _state;
};
}
}
}
#endif
