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

#ifndef OSGEQ_APPLICATION_H
#define OSGEQ_APPLICATION_H

#include <eq/types.h>

class QObject;

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Config;

/*!
  \brief Class with callbacks to be derived to implement the application
  logic inside the main loop.

  \todo Review the convenience of this monolithic object. Given that the
  common abstract Viewer class has been dropped we can try forwarding
  some of this requests to the responsible objects directly. Independence
  can be preserved if we separate the interface in different proxies.
*/
class Application
{
public:
    virtual ~Application() {}
    /**
       This method executes actions to be done after the frame
       loop has been entered and before emitting a frame with
       eq::Config::startFrame. This callback is only called in the
       application node.
    */
    virtual void preFrame() = 0;

    /**
       Only called in the call application node

       Called after eq::Config::finishFrame. This callback is only called
       in the application node.
    */
    virtual void postFrame() = 0;

    /**
       Called from Node::frameStart before the scenegraph update traversal
       is run.
    */
    virtual void preNodeUpdate(const uint32_t frameNumber) = 0;

    /**
       Called from Node::frameStart after the scenegraph update traversal
       is run and before eq::Node::frameStart is called.
    */
    virtual void preNodeDraw(const uint32_t frameNumber) = 0;

    /**
       Called from osgEq::Config::init before the init data is registered
       and eq::Config::init is called.
       Use this callback, for instance, to register application specific
       distributed objects.
    */
    virtual void init(osgEq::Config* config) = 0;

    /**
       Called from the application loop before osgEq::Config::exit.
    */
    virtual void exit(osgEq::Config* config) = 0;

    /**
       Called at the end of Node::configInit.
     */
    virtual void configInit(eq::Node* node) = 0;

    /**
       Called from at the beginning of Node::configExit.
    */
    virtual void configExit(eq::Node* node) = 0;

    virtual bool handleEvent(eq::EventICommand command) = 0;

    virtual void onActiveViewTextureUpdated(const unsigned textureID) = 0;

    virtual void onActiveViewEventProcessorUpdated(QObject* glWidget) = 0;

    virtual void onFrameGrabbed(const eq::uint128_t& viewID,
                                const std::vector<uint8_t>& data) = 0;

    virtual void onIdle() = 0;

    virtual void onDone() = 0;

    virtual void onLayoutChanged(const eq::Layout* oldLayout,
                                 const eq::Layout* newLayout) = 0;
};
}
}
}
#endif
