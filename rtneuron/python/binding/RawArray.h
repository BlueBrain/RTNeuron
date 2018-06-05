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

#ifndef RTNEURON_PYTHON_RAW_ARRAY
#define RTNEURON_PYTHON_RAW_ARRAY

#include <boost/python/object.hpp>

#include <memory>
#include <vector>

namespace bbp
{
namespace rtneuron
{
template <typename T>
struct RawArray
{
    RawArray() {}
    RawArray(const boost::python::object& o);

    RawArray(std::unique_ptr<T[]>&& array_,
             const std::vector<size_t>& shape_ = {0})
        : array(std::move(array_))
        , shape(shape_)
    {
    }
    RawArray(std::unique_ptr<T[]>&& array_, std::vector<size_t>&& shape_ = {0})
        : array(std::move(array_))
        , shape(std::move(shape_))
    {
    }
    RawArray(T* array_, const std::vector<size_t>& shape_)
        : array(array_)
        , shape(shape_)
    {
    }
    RawArray(T* array_, std::vector<size_t>&& shape_)
        : array(array_)
        , shape(std::move(shape_))
    {
    }
    std::unique_ptr<T[]> array;
    /* This is the original shape of the original Python object */
    std::vector<size_t> shape;

    size_t size() const
    {
        if (shape.empty())
            return 0;
        size_t s = 1;
        for (auto i : shape)
            s *= i;
        return s;
    }
};
}
}
#endif
