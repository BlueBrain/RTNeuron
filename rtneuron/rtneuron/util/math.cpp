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

#include "math.h"

#include <osg/Matrixd>
#include <vmmlib/vector.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
template <typename T>
struct TypeSelector;

template <>
struct TypeSelector<float>
{
    typedef osg::Vec3f osgVec3;
    typedef brion::Vector3f Vec3;
    typedef brion::Vector4f Vec4;
};

template <>
struct TypeSelector<double>
{
    typedef osg::Vec3d osgVec3;
    typedef brion::Vector3d Vec3;
    typedef brion::Vector4d Vec4;
};
}

template <typename Matrix>
void _decomposeMatrix(
    Matrix matrix,
    typename TypeSelector<typename Matrix::value_type>::Vec3& position,
    typename TypeSelector<typename Matrix::value_type>::Vec4& orientation)
{
    matrix.invert(matrix);

    typedef typename Matrix::value_type value_type;

    typename TypeSelector<value_type>::osgVec3 t, s;
    osg::Quat r, so;
    matrix.decompose(t, r, s, so);
    position = typename TypeSelector<value_type>::Vec3(t[0], t[1], t[2]);

    double x, y, z, angle;
    r.getRotate(angle, x, y, z);
    orientation =
        typename TypeSelector<value_type>::Vec4(x, y, z, angle * 180 / M_PI);
}

void decomposeMatrix(const osg::Matrixd& matrix, brion::Vector3d& position,
                     brion::Vector4d& orientation)
{
    _decomposeMatrix(matrix, position, orientation);
}

void decomposeMatrix(const osg::Matrixf& matrix, brion::Vector3f& position,
                     brion::Vector4f& orientation)
{
    _decomposeMatrix(matrix, position, orientation);
}
}
}
}
