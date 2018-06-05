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

#include <osg/Quat>
#include <osg/ShapeDrawable>

#include "util/shapes.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
osg::ShapeDrawable* sphereDrawable(const osg::Vec3& center, const float radius,
                                   const osg::Vec4& color,
                                   const double detailRatio)
{
    osg::Sphere* sphere = new osg::Sphere(center, radius);
    osg::ShapeDrawable* drawable = new osg::ShapeDrawable(sphere);
    osg::TessellationHints* tessHints = new osg::TessellationHints();
    tessHints->setDetailRatio(detailRatio);
    drawable->setTessellationHints(tessHints);
    drawable->setColor(color);
    return drawable;
}

osg::ShapeDrawable* capsuleDrawable(const osg::Vec3& center, const float radius,
                                    const osg::Vec3& axis,
                                    const osg::Vec4& color,
                                    const double detailRatio)
{
    osg::Capsule* capsule = new osg::Capsule(center, radius, axis.length());
    osg::Quat rotation;
    rotation.makeRotate(osg::Vec3(0, 0, 1), axis);
    capsule->setRotation(rotation);
    osg::ShapeDrawable* drawable = new osg::ShapeDrawable(capsule);
    osg::TessellationHints* tessHints = new osg::TessellationHints();
    tessHints->setDetailRatio(detailRatio);
    drawable->setTessellationHints(tessHints);
    drawable->setColor(color);
    return drawable;
}
}
}
}
