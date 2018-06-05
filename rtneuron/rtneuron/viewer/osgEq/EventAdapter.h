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

#ifndef RTNEURON_OSGEQ_EVENTADAPTER_H
#define RTNEURON_OSGEQ_EVENTADAPTER_H

#include <co/version.h>
#include <eq/types.h>

#include <osgGA/GUIEventAdapter>

#include <eq/fabric/renderContext.h>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
//! osgGA::GUIEventAdapter implementation for Equalizer events.
/*! The only event types for which implementation is provided are:
  PUSH, RELEASE, DRAG, MOVE, KEYDOWN, KEYUP, FRAME and RESIZE.

  All methods are self explanatory. If in doubt refer to OSG documentation.

  In all osgEQ events mouse positions will be [-1, 1].
  The reported range by X,Y min max functions is always that.
*/
class EventAdapter : public osgGA::GUIEventAdapter
{
public:
    /*--- Public declarations ---*/

    //! Additional of information of a GUI event
    struct Context
    {
        lunchbox::UUID originator;
        eq::fabric::uint128_t frameID;
        eq::Vector2i offset;
        eq::Frustumf frustum;
        eq::Frustumf ortho;
        eq::Matrix4f headTransform;
    };

    /*--- Public constructors/destructor ---*/

    EventAdapter();

    EventAdapter(eq::EventICommand command,
                 const eq::RenderContext& lastFocusedContext);

    EventAdapter(const EventAdapter& event,
                 const osg::CopyOp& op = osg::CopyOp())
        : osgGA::GUIEventAdapter(event, op)
        , _context(event._context)
    {
    }

    /*--- Public member functions ---*/

    META_Object(osgEq, EventAdapter);

    const Context& getContext() const { return _context; }
    /* Member attributes */
protected:
    Context _context;

private:
    void _setDefaults();
};
}
}
}
#endif
