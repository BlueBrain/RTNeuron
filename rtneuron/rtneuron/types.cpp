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

#include "types.h"

#include <lunchbox/any.h>
#include <lunchbox/debug.h>

#ifdef final
#undef final
#endif
#include <boost/assign/list_of.hpp>
#include <boost/bimap.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>

using namespace bbp::rtneuron;

/*
  Registration of types from types.h for serialization of their any
  placeholders.
*/
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<DataBasePartitioning>, "a:DBP");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<RepresentationMode>, "a:RM");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<ColorScheme>, "a:CS");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<NeuronLOD>, "a:NL");

namespace bbp
{
namespace rtneuron
{
namespace
{
/*
  Helper types and functions
*/
typedef boost::bimap<RepresentationMode, std::string> RepresentationModeStrings;
const static RepresentationModeStrings representationModeStrings =
    boost::assign::list_of<RepresentationModeStrings::relation>(
        RepresentationMode::SOMA,
        "soma")(RepresentationMode::WHOLE_NEURON,
                "detailed")(RepresentationMode::SEGMENT_SKELETON, "skeleton")(
        RepresentationMode::NO_AXON, "no_axon")(RepresentationMode::NO_DISPLAY,
                                                "none");

typedef boost::bimap<NeuronLOD, std::string> NeuronLODStrings;
const static NeuronLODStrings neuronModelStrings =
    boost::assign::list_of<NeuronLODStrings::relation>(NeuronLOD::MEMBRANE_MESH,
                                                       "mesh")(
        NeuronLOD::TUBELETS, "tubelets")(NeuronLOD::HIGH_DETAIL_CYLINDERS,
                                         "high_detail_cylinders")(
        NeuronLOD::LOW_DETAIL_CYLINDERS,
        "low_detail_cylinders")(NeuronLOD::DETAILED_SOMA,
                                "detailed_soma")(NeuronLOD::SPHERICAL_SOMA,
                                                 "spherical_soma");

template <bool right, typename Key, typename Value>
struct ChooseView;

template <typename Key, typename Value>
struct ChooseView<true, Key, Value>
{
    ChooseView(const boost::bimap<Key, Value>& bimap)
        : map(bimap.right)
    {
    }
    typedef typename boost::bimap<Key, Value>::right_map::const_iterator
        const_iterator;
    const typename boost::bimap<Key, Value>::right_map& map;
};

template <typename Key, typename Value>
struct ChooseView<false, Key, Value>
{
    ChooseView(const boost::bimap<Key, Value>& bimap)
        : map(bimap.left)
    {
    }
    typedef typename boost::bimap<Key, Value>::left_map::const_iterator
        const_iterator;
    const typename boost::bimap<Key, Value>::left_map& map;
};

template <bool right, typename Key, typename Value>
typename boost::mpl::if_c<right, Key, Value>::type findTranslation(
    const typename boost::mpl::if_c<right, Value, Key>::type& key,
    const boost::bimap<Key, Value>& bimap)
{
    typedef typename boost::mpl::if_c<right, Key, Value>::type To;
    typedef typename boost::mpl::if_c<right, Value, Key>::type From;

    typedef ChooseView<right, Key, Value> View;
    View view(bimap);

    typename View::const_iterator i = view.map.find(key);
    if (i == view.map.end())
        throw std::runtime_error("Invalid conversion from " +
                                 lunchbox::className(From()) + " to " +
                                 lunchbox::className(To()));
    return i->second;
}
}

/*
  Lexical casts for NeuronLOD
*/
template <>
std::string lexical_cast(const NeuronLOD& value)
{
    return findTranslation<false>(value, neuronModelStrings);
};

template <>
NeuronLOD lexical_cast(const std::string& value)
{
    return findTranslation<true>(value, neuronModelStrings);
}

/*
  Lexical casts for RepresentationMode
*/
template <>
std::string lexical_cast(const RepresentationMode& value)
{
    return findTranslation<false>(value, representationModeStrings);
}

template <>
RepresentationMode lexical_cast(const std::string& value)
{
    return findTranslation<true>(value, representationModeStrings);
}
}
}
