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

#ifndef RTNEURON_OSGEQ_SCENE_H
#define RTNEURON_OSGEQ_SCENE_H

#include "types.h"

#include <lunchbox/referenced.h>

#include <osg/BoundingBox>
#include <osg/Group>
#include <osg/Matrix>
#include <osg/Vec4>
#include <osg/ref_ptr>

#include <map>

namespace osg
{
class Node;
}

namespace eq
{
namespace fabric
{
class Range;
class PixelViewport;
}
}

namespace bbp
{
namespace osgTransparency
{
class BaseRenderBin;
}

namespace rtneuron
{
namespace osgEq
{
#define UNDEFINED_SCENE_ID LB_UNDEFINED_UINT32

class Scene
{
public:
    /*--- Declarations ---*/

    enum DBCompositingTechnique
    {
        /* Per pixel sorting with multiple fragments per frame, works for both
           opaque and transparent geometry. */
        ORDER_INDEPENDENT_MULTIFRAGMENT,
        /* Per pixel sorting with one fragment per frame, only works for opaque
           geometry. */
        ORDER_INDEPENDENT_SINGLE_FRAGMENT,
        /* Each scene range is a non overlapping spatial region, frames must
           be composited back to front. Works for both opaque and transparent
           geometry. */
        ORDER_DEPENDENT
    };

    /*--- Public constructors/destructor ---*/

    virtual ~Scene() {}
    Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    /*--- Public member functions ---*/

    //! Returns the global unique identifier for this scene.
    virtual unsigned int getID() const = 0;

    /**
       Returns the scenegraph node to be used for a given database range.

       The implementation may choose to create the whole scenegraph at
       the moment or return an empty node and populate it later.
       In any case, the range is assumed to be a quasi static.
       Calling this method multiple times with the same range will return
       the same node (so 2D or hybrid decompositions can resuse the same
       scenegraph)

       Range updates are only possible with updateSubsceneRange.
    */
    virtual osg::ref_ptr<osg::Node> getOrCreateSubSceneNode(
        const eq::fabric::Range& range) = 0;

    /**
       Returns the position of a subscene range in back to front compositing
       (smaller numbers go first).
    */
    virtual unsigned int computeCompositingPosition(
        const osg::Matrix& modelView, const eq::fabric::Range& range) const = 0;

    /**
       Updates scene-dependent data stored in the camera.

       For example this is used to update the CircuitScene pointers in
       stored by CameraData.

       @param range The DB range of the channel invoking this function.
       @param camera The camera in which the data is updated.
    */
    virtual void updateCameraData(const eq::fabric::Range& range,
                                  osg::Camera* camera) = 0;

    /**
       Update the dynamic data or range of a scene to the content expected
       at a given frame so it's available at subsequent channelSync calls.

       The update process is intended for loading pending data and perform DB
       decompositions as needed. This process mustn't interfere with the
       rendering of sub scenes that have already been created and might
       be rendering (e.g. in dplex modes).

       The actual scenegraph update traversal is still performed from Node.

       Called on all scenes from Node::frameStart after frame data is synched
       and updatePendingSubScenes called and before preNodeUpdate is called.
    */
    /*
       This function will also be responsible of updating the data range of
       already created subscenes when updates are possible.
    */
    virtual void nodeSync(const uint32_t frameNumber,
                          const osgEq::FrameData& frameData) = 0;

    /**
       Update the dynamic data of the subscene associated with a range to
       the content expected at a given frame.

       To be called from Channel::frameDraw. For shared subscenes
       (2D decompositions) this function may be called from multiple
       pipes at the same time.
    */
    virtual void channelSync(const eq::fabric::Range& range,
                             const uint32_t frameNumber,
                             const osgEq::FrameData& frameData) = 0;

    /**
       Returns the region of interest for a channel in float pixel viewport
       coordinates.

       If the region is empty returns all zeros.
     */
    osg::Vec4 getRegionOfInterest(const Channel& channel) const;

    /** @return the axis aligned bounding box of a subscene in world
        coordinates.
    */
    virtual osg::BoundingBox getSceneBoundingBox(
        const eq::fabric::Range& range) const = 0;

    /** @return the frame compositing technique to be used for this scene. */
    virtual DBCompositingTechnique getDBCompositingTechnique() const = 0;

    virtual osgTransparency::BaseRenderBin* getAlphaBlendedRenderBin() = 0;
};
}
}
}
#endif
