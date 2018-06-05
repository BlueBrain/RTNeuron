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

#include "Neurons.h"

#include "Neuron.h"
#include "util/vec_to_vec.h"

#include <brain/circuit.h>
#include <brain/types.h>

#include <algorithm>

#define CHECK_COMPATIBLE_CONTAINERS(a, b)                             \
    if ((a)._circuit && (b)._circuit && (a)._circuit != (b)._circuit) \
    {                                                                 \
        throw std::runtime_error(                                     \
            "Neuron containers are not from the same circuit");       \
    }

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
struct Comparator
{
    bool operator()(const uint32_t gid, const NeuronPtr& neuron) const
    {
        return gid < neuron->getGID();
    }
    bool operator()(const NeuronPtr& neuron, const uint32_t gid) const
    {
        return neuron->getGID() < gid;
    }
    bool operator()(const NeuronPtr& a, const NeuronPtr& b) const
    {
        return a->getGID() < b->getGID();
    }
};
}
Neurons::Neurons(const GIDSet& gids, const CircuitPtr& circuit)
    : _circuit(circuit)
{
    const auto positions = circuit->getPositions(gids);
    const auto orientations = circuit->getRotations(gids);
    auto gid = gids.begin();
    for (size_t i = 0; i != gids.size(); ++i, ++gid)
    {
        auto neuron = NeuronPtr(new Neuron);
        neuron->_circuit = circuit;
        neuron->_gid = *gid;
        neuron->_position = vec_to_vec(positions[i]);
        const auto& q = orientations[i];
        neuron->_orientation = osg::Quat(q.x(), q.y(), q.z(), q.w());
        _neurons.push_back(neuron);
    }
}

Neurons::Neurons(Neurons&&) = default;
Neurons::Neurons(const Neurons&) = default;
Neurons& Neurons::operator=(Neurons&&) = default;

Neurons::operator GIDSet() const
{
    GIDSet set;
    for (const auto& neuron : _neurons)
        set.insert(set.end(), neuron->getGID());
    return set;
}

NeuronPtr Neurons::find(uint32_t gid) const
{
    Comparator comp;
    auto it = std::lower_bound(_neurons.begin(), _neurons.end(), gid, comp);
    return it != _neurons.end() && (*it)->getGID() == gid ? *it : NeuronPtr();
}

void Neurons::insert(const NeuronPtr& neuron)
{
    if (!_circuit)
        _circuit = neuron->getCircuit();
    if (_circuit != neuron->getCircuit())
        throw std::runtime_error("Cannot add neuron from a different circuit");
    /* Fast code path for in order insertion */
    auto gid = neuron->getGID();
    if (_neurons.empty() || _neurons.back()->getGID() < gid)
    {
        _neurons.push_back(neuron);
        return;
    }
    auto position =
        std::lower_bound(_neurons.begin(), _neurons.end(), gid, Comparator());
    /* Not adding a neuron with a GID already present, even if the object has
       a different identify than the existing one. */
    if (position != _neurons.end() && (*position)->getGID() == gid)
        return;
    _neurons.insert(position, neuron);
}

Neurons Neurons::operator&(const GIDSet& gids) const
{
    Neurons result;
    std::set_intersection(_neurons.begin(), _neurons.end(), gids.begin(),
                          gids.end(), std::back_inserter(result._neurons),
                          Comparator());
    result._circuit = _circuit;
    return result;
}

Neurons Neurons::operator-(const GIDSet& gids) const
{
    Neurons result;
    result._circuit = _circuit;
    std::set_difference(_neurons.begin(), _neurons.end(), gids.begin(),
                        gids.end(), std::back_inserter(result._neurons),
                        Comparator());
    return result;
}

Neurons Neurons::operator&(const Neurons& other) const
{
    CHECK_COMPATIBLE_CONTAINERS(*this, other);
    Neurons result;
    std::set_intersection(_neurons.begin(), _neurons.end(),
                          other._neurons.begin(), other._neurons.end(),
                          std::back_inserter(result._neurons), Comparator());
    result._circuit = _circuit;
    return result;
}

Neurons Neurons::operator+(const Neurons& other) const
{
    CHECK_COMPATIBLE_CONTAINERS(*this, other);
    Neurons result;
    result._neurons.reserve(other._neurons.size() + _neurons.size());
    std::set_union(_neurons.begin(), _neurons.end(), other._neurons.begin(),
                   other._neurons.end(), std::back_inserter(result._neurons),
                   Comparator());
    result._circuit = _circuit;
    return result;
}

Neurons Neurons::operator-(const Neurons& other) const
{
    CHECK_COMPATIBLE_CONTAINERS(*this, other);
    Neurons result;
    result._neurons.reserve(other._neurons.size() + _neurons.size());
    std::set_difference(_neurons.begin(), _neurons.end(),
                        other._neurons.begin(), other._neurons.end(),
                        std::back_inserter(result._neurons), Comparator());
    result._circuit = _circuit;
    return result;
}

Neurons& Neurons::operator+=(const Neurons& other)
{
    CHECK_COMPATIBLE_CONTAINERS(*this, other);
    std::vector<NeuronPtr> neurons;
    neurons.reserve(_neurons.size() + other.size());
    std::set_union(_neurons.begin(), _neurons.end(), other._neurons.begin(),
                   other._neurons.end(), std::back_inserter(neurons),
                   Comparator());
    _neurons = std::move(neurons);
    _circuit = other._circuit;
    return *this;
}

void Neurons::clear()
{
    _neurons.clear();
    _circuit.reset();
}

GIDSet operator-(const GIDSet& gids, const Neurons& neurons)
{
    GIDSet result;
    std::set_difference(gids.begin(), gids.end(), neurons._neurons.begin(),
                        neurons._neurons.end(),
                        std::inserter(result, result.end()), Comparator());
    return result;
}

GIDSet operator&(const GIDSet& gids, const Neurons& neurons)
{
    GIDSet result;
    std::set_intersection(gids.begin(), gids.end(), neurons._neurons.begin(),
                          neurons._neurons.end(),
                          std::inserter(result, result.end()), Comparator());
    return result;
}
}
}
}
