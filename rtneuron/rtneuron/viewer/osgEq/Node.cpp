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

#include "Node.h"

#include "Application.h"
#include "Client.h"
#include "Config.h"
#include "ConfigEvent.h"
#include "InitData.h"
#include "NodeViewer.h"
#include "Scene.h"

#include "viewer/StereoAnimationPath.h"
#include "viewer/StereoAnimationPathManipulator.h"

#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/TrackballManipulator>

#include <algorithm>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*
  Constructors/destructor
*/

Node::Node(eq::Config* parent)
    : eq::Node(parent)
    , _viewer(new NodeViewer())
{
}

Node::~Node()
{
}

/*
  Member functions
*/
bool Node::configInit(const eq::uint128_t& initDataID)
{
    if (!eq::Node::configInit(initDataID))
        return false;

    Client* client = getConfig()->getClient();
    client->_localNode = this;

    Config* config = getConfig();
    if (!config->mapDistributedObjects(initDataID))
        return false;

    const InitData& initData = config->getInitData();
    if (!config->mapObject(&_frameData, initData.frameDataID))
        return false;

    client->getApplication()->configInit(this);

    return true;
}

bool Node::configExit()
{
    Config* config = getConfig();
    Client* client = config->getClient();
    client->getApplication()->configExit(this);

    _viewer = 0;

    config->unmapObject(&_frameData);
    config->unmapDistributedObjects();

    return eq::Node::configExit();
}

NodeViewer* Node::getViewer()
{
    return _viewer.get();
}

const NodeViewer* Node::getViewer() const
{
    return _viewer.get();
}

void Node::addView(osgViewer::View* view)
{
    /* In multipipe configurations this function can be called from several
       Channel objects and the same time. */
    std::unique_lock<std::mutex> lock(_mutex);
    _viewer->addView(view);
}

void Node::removeView(osgViewer::View* view)
{
    /* In multipipe configurations this function can be called from several
       Channel objects and the same time. */
    std::unique_lock<std::mutex> lock(_mutex);
    _viewer->removeView(view);
}

Config* Node::getConfig()
{
    return static_cast<Config*>(eq::Node::getConfig());
}

const Config* Node::getConfig() const
{
    return static_cast<const Config*>(eq::Node::getConfig());
}

void Node::registerScene(const ScenePtr& scene)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _scenes[scene->getID()] = scene;

    /** \todo Cleanup relinquished scenes? */
}

ScenePtr Node::findScene(unsigned int sceneID)
{
    std::unique_lock<std::mutex> lock(_mutex);
    SceneMap::iterator s = _scenes.find(sceneID);
    ScenePtr scene;
    if (s != _scenes.end())
    {
        scene = s->second.lock();
        if (!scene)
        {
            _scenes.erase(s);
        }
    }
    return scene;
}

void Node::frameStart(const eq::uint128_t& frameID, const uint32_t frameNumber)
{
    _frameData.sync(frameID);

    _updateScenes(frameNumber);

    Config* config = getConfig();
    Client* client = config->getClient();

    assert(client->_frameStamp.get());
    client->_frameStamp->setFrameNumber(frameNumber);

    osg::FrameStamp* frameStamp = _viewer->getFrameStamp();
    frameStamp->setFrameNumber(frameNumber);

    Application* app = client->getApplication();

    app->preNodeUpdate(frameNumber);

    _viewer->eventTraversal();
    _viewer->updateTraversal();

    app->preNodeDraw(frameNumber);

    /* Unblock all pipes for this frame */
    eq::Node::frameStart(frameID, frameNumber);
}

void Node::frameFinish(const eq::uint128_t& /* frameID */,
                       const uint32_t frameNumber)
{
    releaseFrame(frameNumber);
}

void Node::_updateScenes(const uint32_t frameNumber)
{
    typedef std::vector<ScenePtr> SceneList;
    SceneList scenes;
    {
        std::unique_lock<std::mutex> lock(_mutex);
        for (const auto& i : _scenes)
        {
            const auto scene = i.second.lock();
            if (scene)
                scenes.push_back(scene);
            /** \todo Remove unused scenes */
        }
    }
    for (auto& scene : scenes)
        scene->nodeSync(frameNumber, _frameData);

    /* Using codash for scenes this could be the point to sync the per node
       copy of the scene. */
}
}
}
}
