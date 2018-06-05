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

#ifndef RTNEURON_PERFRAMEATTRIBUTE_IPP
#define RTNEURON_PERFRAMEATTRIBUTE_IPP

namespace bbp
{
namespace rtneuron
{
namespace core
{
template <typename T>
bool PerFrameAttribute<T>::getValue(const size_t frame, T& value) const
{
    if (frame == _latestFrame)
    {
        value = _latest;
        return true;
    }

    for (typename FrameValueList::const_iterator i = _values.begin();
         i != _values.end() && i->first >= frame; ++i)
    {
        if (i->first == frame)
        {
            value = i->second;
            return true;
        }
    }
    return false;
}

template <typename T>
void PerFrameAttribute<T>::setValue(const T& value)
{
    if (_values.size() == s_maxLatency)
        _values.pop_back();

    _values.insert(_values.begin(), std::make_pair(_latestFrame, _latest));
    _latest = value;
    _latestFrame = _latestFrame + 1;
}

template <typename T>
void PerFrameAttribute<T>::setValue(const size_t frame, const T& value)
{
    if (_latestFrame == frame)
    {
        _latest = value;
        return;
    }

    typename FrameValueList::iterator position = _values.begin();
    for (; position != _values.end() && position->second >= frame; ++position)
    {
        if (position->first == frame)
        {
            position->second = value;
            return;
        }
    }

    if (_values.size() == s_maxLatency)
        _values.pop_back();

    _values.insert(position, std::make_pair(frame, value));
}

template <typename T>
size_t PerFrameAttribute<T>::s_maxLatency = 1;
}
}
}
#endif
