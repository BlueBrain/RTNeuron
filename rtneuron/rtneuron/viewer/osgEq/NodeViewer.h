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

#ifndef RTNEURON_OSGEQ_NODEVIEWER_H
#define RTNEURON_OSGEQ_NODEVIEWER_H

#include <eq/gl.h>

#include <osgGA/EventQueue>
#include <osgViewer/CompositeViewer>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Channel;

//! Override of osgViewer::CompositeViewer that belongs to a Node
/**
   This class works much as osgViewer::CompositeViewer but the event, update
   and render traversal methods have been overriden with a simpler customized
   implementation.
*/
class NodeViewer : public osgViewer::CompositeViewer
{
public:
    /*--- Public declarations ---*/

    typedef std::list<osg::ref_ptr<osgGA::GUIEventHandler>> EventHandlers;

    /*--- Public constructors/destructor ---*/

    NodeViewer()
        : _appEventQueue(new osgGA::EventQueue())
    {
        /** \todo Set the traits of the graphics window */
        setThreadingModel(osgViewer::ViewerBase::SingleThreaded);
    }

    /*--- Public member functions ---*/

    EventHandlers& getEventHandlers() { return _eventHandlers; }
    osgGA::EventQueue* getApplicationEventQueue()
    {
        return _appEventQueue.get();
    }

    void eventTraversal();

    void updateTraversal();

    void renderTraversal(osg::Camera* camera);

private:
    /*--- Private Member attributes ---*/

    EventHandlers _eventHandlers;

    osg::ref_ptr<osgGA::EventQueue> _appEventQueue;
};
}
}
}
#endif
