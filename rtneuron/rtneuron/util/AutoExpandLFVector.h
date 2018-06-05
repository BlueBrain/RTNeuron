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

#ifndef RTNEURON_AUTOEXPANDLFVECTOR_H
#define RTNEURON_AUTOEXPANDLFVECTOR_H

#include <lunchbox/lfVector.h>
#include <lunchbox/scopedMutex.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
 */
template <typename T>
class AutoExpandLFVector
{
public:
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    bool empty() const { return _vector.empty(); }
    typedef typename lunchbox::LFVector<T>::iterator iterator;
    typedef typename lunchbox::LFVector<T>::const_iterator const_iterator;
    iterator begin() { return _vector.begin(); }
    iterator end() { return _vector.end(); }
    const_iterator begin() const { return _vector.begin(); }
    const_iterator end() const { return _vector.end(); }
    size_t size() const { return _vector.size(); }
private:
    mutable lunchbox::LFVector<T> _vector;
};
}
}
}

#include "AutoExpandLFVector.ipp"

#endif
