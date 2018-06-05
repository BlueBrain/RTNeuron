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

#ifndef VEC_TO_VEC_H
#define VEC_TO_VEC_H

#include <osg/Vec3>
#include <osg/Vec4>

#include <brion/types.h>

#include <vmmlib/vector.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! vmml to osg vector conversion
inline osg::Vec3 vec_to_vec(const brion::Vector3f& v)
{
    return osg::Vec3(v.x(), v.y(), v.z());
}

//! vmml to osg vector conversion
inline osg::Vec4 vec_to_vec(const brion::Vector4f& v)
{
    return osg::Vec4(v.x(), v.y(), v.z(), v.w());
}

//! OSG to vmml vector conversion
inline vmml::Vector3f vec_to_vec(const osg::Vec3& v)
{
    return brion::Vector3f(v[0], v[1], v[2]);
}

//! vmml to OSG vector reinterpret cast
inline osg::Vec3f& vector_cast(vmml::Vector3f& v)
{
    return *static_cast<osg::Vec3f*>(static_cast<void*>(&v));
}

//! vmml to OSG vector const reinterpret cast
inline const osg::Vec3f& vector_cast(const vmml::Vector3f& v)
{
    return *static_cast<const osg::Vec3f*>(static_cast<const void*>(&v));
}

//! OSG to vmml vector reinterpret cast
inline const vmml::Vector3f& vector_cast(const osg::Vec3f& v)
{
    return *static_cast<const brion::Vector3f*>(static_cast<const void*>(&v));
}
}
}
}
#endif
