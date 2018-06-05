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

#ifndef RTNEURON_COLLECTSKELETONSVISITOR_H
#define RTNEURON_COLLECTSKELETONSVISITOR_H

#include <osg/NodeVisitor>

#include <memory>
#include <unordered_set>

namespace osg
{
class RenderInfo;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
class Skeleton;

/**
   Node visitor that collects all the skeletons of a subgraph which haven't
   been allocated yet for the device of the graphics context provided.
*/
class CollectSkeletonsVisitor : public osg::NodeVisitor
{
public:
    /*--- Public declarations ---*/

    typedef std::unordered_set<std::shared_ptr<Skeleton>> SkeletonSet;

    /*--- Public member attributes ---*/

    /* Set of skeletons with a visibility array to be allocated */
    SkeletonSet visibilityAllocPending;
    /* Set of skeletons with its device capsule data to be allocated */
    SkeletonSet deviceAllocPending;

    /*--- Public Constructor/destructor ---*/

    CollectSkeletonsVisitor(osg::RenderInfo& renderInfo);

    /*--- Public Member functions ---*/

    virtual void reset()
    {
        visibilityAllocPending.clear();
        deviceAllocPending.clear();
    }

    virtual void apply(osg::Geode& geode);

    osg::RenderInfo& getRenderInfo() { return _renderInfo; }
private:
    /*--- Private Member attributes ---*/

    osg::RenderInfo& _renderInfo;
};
}
}
}
#endif // USE_CUDA

#endif
