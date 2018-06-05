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

#ifndef RTNEURON_OSGEQ_WINDOW_H
#define RTNEURON_OSGEQ_WINDOW_H

#include <osgGA/EventQueue>

#include <eq/window.h>

#include "EmbeddedWindow.h"

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class Channel;

/*! \brief The class that holds the osg::GraphicsContext for OSG as a
  osgViewer::EmbeddedWindow.

  It also processes the window system events to send them to the master node
  as config events (however this responsibility may be moved to Camera)
*/
class Window : public eq::Window
{
    /* Contructors */
public:
    Window(eq::Pipe* parent)
        : eq::Window(parent)
    {
    }

    /* Destructor */
protected:
    virtual ~Window() {}
    /* Member functions */
public:
    virtual bool configInitGL(const eq::uint128_t& initID);

    virtual bool configExitGL();

    bool configInitSystemWindow(const eq::uint128_t& initID) override;

    /**
       The osgViewer::GraphicsWindowEmbedded object for this window.
     */
    osgViewer::GraphicsWindowEmbedded* embedded() const
    {
        return _osgWindow.get();
    }

protected:
    osg::ref_ptr<EmbeddedWindow> _osgWindow;
    eq::SystemWindow* _guiWindow = nullptr;
};
}
}
}
#endif
