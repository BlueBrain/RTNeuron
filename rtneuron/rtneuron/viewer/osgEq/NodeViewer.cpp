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

#include "NodeViewer.h"

#include <osg/Version>
#include <osgViewer/Renderer>

#include <cassert>
#include <iostream>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*
  Member functions
*/
void NodeViewer::eventTraversal()
{
    if (_done)
        return;

    if (_views.empty())
        return;

    typedef std::map<osgViewer::View*, osgGA::EventQueue::Events> ViewEventsMap;
    ViewEventsMap viewEventsMap;

    Contexts contexts;
    getContexts(contexts);

    Scenes scenes;
    getScenes(scenes);

    osgViewer::View* masterView =
        getViewWithFocus() ? getViewWithFocus() : _views[0].get();

    osgGA::EventQueue::Events appEvents;
    _appEventQueue->takeEvents(appEvents);

    for (RefViews::iterator i = _views.begin(); i != _views.end(); ++i)
    {
        osgViewer::View* view = i->get();
        view->getEventQueue()->frame(getFrameStamp()->getReferenceTime());
        view->getEventQueue()->takeEvents(viewEventsMap[view]);
    }

    for (ViewEventsMap::iterator i = viewEventsMap.begin();
         i != viewEventsMap.end(); ++i)
    {
        osgViewer::View* view = i->first;
        _eventVisitor->setActionAdapter(view);
        osgGA::EventQueue::Events events(appEvents);
        events.splice(events.end(), i->second);
        for (osgGA::EventQueue::Events::iterator e = events.begin();
             e != events.end(); ++e)
        {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 0)
            osgGA::Event* event = e->get();
#else
            osgGA::GUIEventAdapter* event = e->get();
#endif
            for (osgViewer::View::EventHandlers::iterator j =
                     view->getEventHandlers().begin();
                 j != view->getEventHandlers().end(); ++j)
            {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 0)
                (*j)->handle(event, 0, _eventVisitor);
#else
                (*j)->handleWithCheckAgainstIgnoreHandledEventsMask(*event,
                                                                    *view, 0,
                                                                    0);
#endif
            }

            if (view == masterView)
            {
                for (EventHandlers::iterator j = _eventHandlers.begin();
                     j != _eventHandlers.end(); ++j)
                {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 0)
                    (*j)->handle(event, 0, _eventVisitor);
#else
                    (*j)->handleWithCheckAgainstIgnoreHandledEventsMask(
                        *event, *masterView, 0, 0);
#endif
                }
            }

            if (view->getCameraManipulator())
            {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 0)
                view->getCameraManipulator()->handle(event, 0, _eventVisitor);
#else
                view->getCameraManipulator()
                    ->handleWithCheckAgainstIgnoreHandledEventsMask(*event,
                                                                    *view);
#endif
            }
        }
    }

    /* Clearing the application wide event queue  */
    if (_eventVisitor.valid())
    {
        _eventVisitor->setFrameStamp(getFrameStamp());
        _eventVisitor->setTraversalNumber(getFrameStamp()->getFrameNumber());

        for (ViewEventsMap::iterator v = viewEventsMap.begin();
             v != viewEventsMap.end(); ++v)
        {
            osgViewer::View* view = v->first;
            if (view->getSceneData())
            {
                for (osgGA::EventQueue::Events::iterator e = v->second.begin();
                     e != v->second.end(); ++e)
                {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 0)
                    osgGA::Event* event = e->get()->asGUIEventAdapter();
#else
                    osgGA::GUIEventAdapter* event = e->get();
#endif
                    _eventVisitor->reset();
                    _eventVisitor->addEvent(event);

                    view->getSceneData()->accept(*_eventVisitor);

                    /* Call any camera update callbacks, but only traverse
                       that callback, don't traverse its subgraph. Leave
                       that to the scene update traversal */
                    osg::NodeVisitor::TraversalMode tm =
                        _eventVisitor->getTraversalMode();
                    _eventVisitor->setTraversalMode(
                        osg::NodeVisitor::TRAVERSE_NONE);

                    if (view->getCamera() &&
                        view->getCamera()->getEventCallback())
                        view->getCamera()->accept(*_eventVisitor);

                    for (unsigned int i = 0; i < view->getNumSlaves(); ++i)
                    {
                        osg::Camera* camera = view->getSlave(i)._camera.get();
                        if (camera && camera->getEventCallback())
                            camera->accept(*_eventVisitor);
                    }

                    _eventVisitor->setTraversalMode(tm);
                }
            }
        }
    }
}

void NodeViewer::updateTraversal()
{
    if (_done)
        return;

    _updateVisitor->reset();
    _updateVisitor->setFrameStamp(getFrameStamp());
    _updateVisitor->setTraversalNumber(getFrameStamp()->getFrameNumber());

    Scenes scenes;
    getScenes(scenes);
    for (Scenes::iterator i = scenes.begin(); i != scenes.end(); ++i)
    {
        osg::ref_ptr<osgViewer::Scene> scene = *i;
        if (scene->getSceneData())
        {
            _updateVisitor->setImageRequestHandler(scene->getImagePager());
            scene->getSceneData()->accept(*_updateVisitor);
        }

        if (scene->getDatabasePager())
        {
            /* synchronize changes required by the DatabasePager thread
               to the scene graph. */
            scene->getDatabasePager()->updateSceneGraph(*getFrameStamp());
        }

        if (scene->getImagePager())
        {
            /* Synchronize changes required by the DatabasePager
               thread to the scene graph. */
            scene->getImagePager()->updateSceneGraph(*getFrameStamp());
        }
    }

    if (_updateOperations.valid())
        _updateOperations->runOperations(this);

    for (RefViews::iterator i = _views.begin(); i != _views.end(); ++i)
    {
        osgViewer::View* view = i->get();

        /* Call any camera update callbacks, but only traverse that
           callback, don't traverse its subgraph leave that to the scene
           update traversal. */
        osg::NodeVisitor::TraversalMode tm = _updateVisitor->getTraversalMode();
        _updateVisitor->setTraversalMode(osg::NodeVisitor::TRAVERSE_NONE);

        if (view->getCamera() && view->getCamera()->getUpdateCallback())
            view->getCamera()->accept(*_updateVisitor);

        for (unsigned int j = 0; j < view->getNumSlaves(); ++j)
        {
            osg::Camera* camera = view->getSlave(j)._camera.get();
            if (camera && camera->getUpdateCallback())
                camera->accept(*_updateVisitor);
        }

        _updateVisitor->setTraversalMode(tm);
    }

    for (Scenes::iterator i = scenes.begin(); i != scenes.end(); ++i)
    {
        osg::ref_ptr<osgViewer::Scene> scene = *i;
        if (scene->getSceneData())
        {
            /* Fire off a build of the bounding volumes while we are still
               running single threaded. */
            scene->getSceneData()->getBound();
        }
    }
}

void NodeViewer::renderTraversal(osg::Camera* camera)
{
    if (_done)
        return;

    osgViewer::Renderer* renderer =
        dynamic_cast<osgViewer::Renderer*>(camera->getRenderer());
    assert(renderer);

    renderer->cull_draw();
}
}
}
}
