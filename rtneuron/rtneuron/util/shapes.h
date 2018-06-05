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

#ifndef RTNEURON_SHAPES_H
#define RTNEURON_SHAPES_H

#include <osg/ShapeDrawable>

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! Creates a drawable object of a sphere
osg::ShapeDrawable* sphereDrawable(const osg::Vec3& center, const float radius,
                                   const osg::Vec4& color = osg::Vec4(1, 1, 1,
                                                                      1),
                                   const double detailRatio = 0.8);

//! Creates a drawable object of capsule
osg::ShapeDrawable* capsuleDrawable(const osg::Vec3& center, const float radius,
                                    const osg::Vec3& axis,
                                    const osg::Vec4& color = osg::Vec4(1, 1, 1,
                                                                       1),
                                    const double detailRatio = 0.8);
}
}
}
#endif
