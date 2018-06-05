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

#ifdef USE_CUDA

#include "CollectSkeletonsVisitor.h"

#include "render/LODNeuronModelDrawable.h"
#include "render/Skeleton.h"
#include "scene/LODNeuronModel.h"
#include "util/cameraToID.h"

#include <osg/Geode>
#include <osg/Version>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Constructors/destructor
*/
CollectSkeletonsVisitor::CollectSkeletonsVisitor(osg::RenderInfo& renderInfo)
    : osg::NodeVisitor(osg::NodeVisitor::NODE_VISITOR,
                       osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    , _renderInfo(renderInfo)
{
    setTraversalMask(0xffffffff);
    setNodeMaskOverride(0xffffffff);
}

/*
  Member functions
*/
void CollectSkeletonsVisitor::apply(osg::Geode& geode)
{
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    for (unsigned int i = 0; i < geode.getNumDrawables(); i++)
    {
        LODNeuronModel::Drawable* lod =
            dynamic_cast<LODNeuronModel::Drawable*>(geode.getDrawable(i));
#else
    typedef osg::Geode::DrawableList Drawables;
    const Drawables& drawables = geode.getDrawableList();
    for (Drawables::const_iterator i = drawables.begin(); i != drawables.end();
         ++i)
    {
        LODNeuronModel::Drawable* lod =
            dynamic_cast<LODNeuronModel::Drawable*>(i->get());
#endif
        if (lod)
        {
            lod->getModel().accept(*this);
            continue;
        }

        osg::Drawable* drawable =
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
            geode.getDrawable(i);
#else
            i->get();
#endif
        const Skeleton::CullCallback* callback =
            dynamic_cast<const Skeleton::CullCallback*>(
                drawable->getCullCallback());
        if (callback)
            callback->accept(*this);
    }
}
}
}
}
#endif // USE_CUDA
