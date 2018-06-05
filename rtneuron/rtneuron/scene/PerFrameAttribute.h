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

#ifndef RTNEURON_PERFRAMEATTRIBUTE_H
#define RTNEURON_PERFRAMEATTRIBUTE_H

#include <vector>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   \brief This is a class to store per frame values of classes similar to
   POD types.

   The template parameter must be copy constructable and assignable type.
   This class is not thread safe
*/
template <typename T>
class PerFrameAttribute
{
public:
    /*--- Public constructors/destructor ---*/
    PerFrameAttribute(const T& value = T())
        : _latest(value)
        , _latestFrame(0)
    {
        _values.reserve(1);
    }

    /*--- Public member functions ---*/

    /**
       Sets the global maximum number of values to be stored in each
       PerFrameAttribute apart from the latest.
       The value cannot be set to less than 1.
    */
    static void setMaximumLatency(const size_t latency)
    {
        s_maxLatency = std::max(size_t(1), latency);
    }

    /**
       Returns the latest value of the attribute.
    */
    T getValue() const { return _latest; }
    /**
       Returns the value of the attribute depending on a frame number.
       This functions will return false if and only if there is no value
       stored for the given frame number.
    */
    inline bool getValue(const size_t frame, T& value) const;

    /**
       Pushes a new value to be used for the last stored frame number
       plus 1.
     */
    inline void setValue(const T& value);

    /**
       Sets a value for the given frame replacing the previous value
       if it existed.
     */
    inline void setValue(const size_t frame, const T& value);

private:
    /*--- Private definitions ---*/
    typedef std::pair<size_t, T> FrameValuePair;

    typedef std::vector<FrameValuePair> FrameValueList;

    /*--- Private member variables ---*/

    static size_t s_maxLatency;
    T _latest;
    size_t _latestFrame;
    FrameValueList _values;
};
}
}
}

#include "PerFrameAttribute.ipp"

#endif
