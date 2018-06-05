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

#ifndef RTNEURON_NEURONS_H
#define RTNEURON_NEURONS_H

#include "coreTypes.h"
#include "types.h"

#include <vector>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/** A Container of neurons

    Set operations between containers check both containers are from the same
    circuit and compare neurons by gid.
*/
class Neurons
{
private:
    typedef std::vector<NeuronPtr> _Neurons;

public:
    /*--- Public declarations ---*/

    using iterator = _Neurons::iterator;
    using const_iterator = _Neurons::const_iterator;

    friend GIDSet operator-(const GIDSet&, const Neurons&);
    friend GIDSet operator&(const GIDSet&, const Neurons&);

    /*--- Public constructors/destructor ---*/
    Neurons() {}
    /** Create a set of neurons from a circuit.
        This function is more efficient than creating and inserting the neurons
        one by one.
    */
    Neurons(const GIDSet& gids, const CircuitPtr& circuit);
    Neurons(Neurons&& other);
    Neurons(const Neurons& other);

    /*--- Public member functions ---*/
    Neurons& operator=(Neurons&& other);

    Neurons& operator=(const Neurons& other) = delete;

    operator GIDSet() const;

    NeuronPtr find(uint32_t gid) const;

    void insert(const NeuronPtr& neuron);

    Neurons operator&(const GIDSet& gids) const;

    Neurons operator-(const GIDSet& gids) const;

    /** Return the container intersection.
        Neurons in the result are taken from this container (left side of
        the operator).
    */
    Neurons operator&(const Neurons& gids) const;

    Neurons operator+(const Neurons& neurons) const;

    Neurons operator-(const Neurons& neurons) const;

    Neurons& operator+=(const Neurons& neurons);

    void clear();

    size_t size() const { return _neurons.size(); }
    bool empty() const { return _neurons.empty(); }
    auto begin() { return _neurons.begin(); }
    auto end() { return _neurons.end(); }
    auto begin() const { return _neurons.begin(); }
    auto end() const { return _neurons.end(); }
    auto cbegin() const { return _neurons.cbegin(); }
    auto cend() const { return _neurons.cend(); }
    const NeuronPtr& operator[](size_t index) const { return _neurons[index]; }
    const CircuitPtr& getCircuit() const { return _circuit; }
private:
    /*--- Private member variables ---*/

    /* Neurons are ordered by GID */
    std::vector<NeuronPtr> _neurons;
    CircuitPtr _circuit;
};

GIDSet operator-(const GIDSet&, const Neurons&);
GIDSet operator&(const GIDSet&, const Neurons&);
}
}
}
#endif
