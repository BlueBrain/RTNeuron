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

#ifndef RTNEURON_DETAILEDNEURONMODELDRAWABLE_H
#define RTNEURON_DETAILEDNEURONMODELDRAWABLE_H

#include <osg/Geometry>
#include <osg/Timer>

#include "scene/DetailedNeuronModel.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class DetailedNeuronModel::Drawable : public osg::Geometry
{
    /* Constructors */
public:
    Drawable() {}
    Drawable(const osg::Geometry& drawable,
             const osg::CopyOp& op = osg::CopyOp::SHALLOW_COPY)
        : osg::Geometry(drawable, op)
    {
        setUseDisplayList(drawable.getUseDisplayList());
        setUseVertexBufferObjects(drawable.getUseVertexBufferObjects());
    }

    Drawable(const Drawable& drawable,
             const osg::CopyOp& op = osg::CopyOp::SHALLOW_COPY)
        : osg::Geometry(drawable, op)
    {
        setUseDisplayList(drawable.getUseDisplayList());
        setUseVertexBufferObjects(drawable.getUseVertexBufferObjects());
    }

    /* We don't want to render any primitive before the skeleton is culled. */
    virtual void compileGLObjects(osg::RenderInfo&) const {}
protected:
    virtual ~Drawable() {}
};
}
}
}
#endif
