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

#ifndef STRINGS_ARRAY_H
#define STRINGS_ARRAY_H

#include <boost/array.hpp>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
inline boost::array<std::string, 1> strings(const std::string& s1)
{
    boost::array<std::string, 1> array;
    array[0] = s1;
    return array;
}

inline boost::array<std::string, 2> strings(const std::string& s1,
                                            const std::string& s2)
{
    boost::array<std::string, 2> array;
    array[0] = s1;
    array[1] = s2;
    return array;
}

inline boost::array<std::string, 3> strings(const std::string& s1,
                                            const std::string& s2,
                                            const std::string& s3)
{
    boost::array<std::string, 3> array;
    array[0] = s1;
    array[1] = s2;
    array[2] = s3;
    return array;
}

inline boost::array<std::string, 4> strings(const std::string& s1,
                                            const std::string& s2,
                                            const std::string& s3,
                                            const std::string& s4)
{
    boost::array<std::string, 4> array;
    array[0] = s1;
    array[1] = s2;
    array[2] = s3;
    array[3] = s4;
    return array;
}

inline boost::array<std::string, 5> strings(const std::string& s1,
                                            const std::string& s2,
                                            const std::string& s3,
                                            const std::string& s4,
                                            const std::string& s5)
{
    boost::array<std::string, 5> array;
    array[0] = s1;
    array[1] = s2;
    array[2] = s3;
    array[3] = s4;
    array[4] = s5;
    return array;
}
}
}
}
#endif
