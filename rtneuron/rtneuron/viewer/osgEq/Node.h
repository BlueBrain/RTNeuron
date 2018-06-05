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

#ifndef RTNEURON_OSGEQ_NODE_H
#define RTNEURON_OSGEQ_NODE_H

#include "FrameData.h"
#include "types.h"

#include "render/ViewStyle.h"

#include <eq/config.h>

#include <lunchbox/refPtr.h>
#include <memory>
#include <mutex>

namespace osgViewer
{
class View;
}

namespace osgGA
{
class KeySwitchMatrixManipulator;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
//! A rendering node in the cluster configuration
/**
   A node contains a NodeViewer, which can hold several osg::View (are the
   reflection of the different eq::Channel mapped to osg::Camera).

   Not to confuse with a co::Node a co::LocalNode, this is a distributed object
*/
class Node : public eq::Node
{
public:
    /*--- Public constructors/destructor ---*/

    Node(eq::Config* parent);

    ~Node();

    /*--- Public Member functions ---*/

    void addView(osgViewer::View* view);
    void removeView(osgViewer::View* view);

    NodeViewer* getViewer();
    const NodeViewer* getViewer() const;

    Config* getConfig();
    const Config* getConfig() const;

    /**
       Stores a weak reference to a scene indexable by its ID.
    */
    void registerScene(const ScenePtr& scene);

    /**
       Returns a shared_ptr to a Scene if the ID is valid and a null pointer
       otherwise.
     */
    ScenePtr findScene(unsigned int sceneID);

    /**
       \brief Node-local read-only instance of the frame data.

       Not to be used by channels or pipes, use eq::Pipe::getFrameData in
       those cases. This is intended to be used only at node level
       (e.g. Application::preNodeUpdate).
    */
    const FrameData& getFrameData() const { return _frameData; }
protected:
    /*--- Protected functions ---*/

    virtual bool configInit(const eq::uint128_t& initDataID);

    virtual bool configExit();

    /**
       @sa eq::Node::frameStart
     */
    virtual void frameStart(const eq::uint128_t& frameID,
                            const uint32_t frameNumber);

    /**
       @sa eq::Node::frameFinish
    */
    virtual void frameFinish(const eq::uint128_t& frameID,
                             const uint32_t frameNumber);

private:
    /*--- Private declarations */

    typedef std::map<unsigned int, std::weak_ptr<Scene>> SceneMap;

    /*--- Private member attributes */

    FrameData _frameData; /* This copy is not used by pipes or channels */

    std::mutex _mutex;
    osg::ref_ptr<NodeViewer> _viewer;

    SceneMap _scenes;

    /*--- Private member functions */

    /**
       \brief Calls nodeSync in all registered scenes.
    */
    void _updateScenes(const uint32_t frameNumber);
};

typedef lunchbox::RefPtr<Node> NodePtr;
}
}
}
#endif
