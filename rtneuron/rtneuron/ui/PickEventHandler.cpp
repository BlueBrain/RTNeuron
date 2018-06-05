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

#include "PickEventHandler.h"

#include "../Scene.h"

#include "viewer/osgEq/EventAdapter.h"
#include "viewer/osgEq/View.h"

namespace bbp
{
namespace rtneuron
{
PickEventHandler::PickEventHandler()
    : EventHandler()
{
}

bool PickEventHandler::handle(const osgEq::EventAdapter& event,
                              const osgEq::View& view)
{
    if (event.getButton() != osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)
        return false;

    switch (event.getEventType())
    {
    case osgGA::GUIEventAdapter::PUSH:
        _pickPoint.set(event.getXnormalized(), event.getYnormalized());
        return false;
    case osgGA::GUIEventAdapter::RELEASE:
    {
        const Vector2f point(event.getXnormalized(), event.getYnormalized());
        if (point != _pickPoint)
            return false;
        break;
    }
    default:
        return false;
    }

    const eq::Matrix4f projection =
        view.useOrtho() ? event.getContext().ortho.computeOrthoMatrix()
                        : event.getContext().frustum.computePerspectiveMatrix();
    const osg::Matrix MVP(view.getModelMatrix() *
                          osg::Matrix(projection.data()));

    osg::Matrix inverseMVP;
    inverseMVP.invert(MVP);

    osg::Vec3 start(_pickPoint.x(), _pickPoint.y(), -1.0f);
    osg::Vec3 end(_pickPoint.x(), _pickPoint.y(), 1.0f);

    start = start * inverseMVP;
    end = end * inverseMVP;
    const osg::Vec3 direction = end - start;

    pick(start, direction);
    return true;
}
}
}
