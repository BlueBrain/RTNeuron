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

#include <eq/qt/window.h>

#include <eq/gl.h>
#include <eq/qt/shareContextWindow.h>

#include "Channel.h"
#include "Config.h"
#include "ConfigEvent.h"
#include "InitData.h"
#include "Window.h"

#if RTNEURON_USE_CUDA
#include "cuda/CUDAContext.h"
#endif

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
bool Window::configInitSystemWindow(const eq::uint128_t& initID)
{
    /* A stencil buffer is needed by the new implementation of fragment linked
       list in osgTransparency. */
    setIAttribute(eq::WindowSettings::IATTR_PLANES_STENCIL, 1);

    Config* config = static_cast<Config*>(getConfig());
    const eq::Pipe* pipe = getPipe();
    const InitData& initData = config->getInitData();
    if (!initData.shareContext || pipe->getWindowSystem().getName() != "Qt")
        return eq::Window::configInitSystemWindow(initID);

#ifndef EQUALIZER_USE_QT5WIDGETS
    return eq::Window::configInitSystemWindow(initID);
#else
    LBASSERT(pipe->getWindowSystem().getName() == "Qt");
    eq::WindowSystem windowSystem = pipe->getWindowSystem();
    eq::WindowSettings settings = getSettings();
    const bool isOnscreen =
        settings.getIAttribute(eq::WindowSettings::IATTR_HINT_DRAWABLE) ==
        eq::WINDOW;

    _guiWindow = new eq::qt::ShareContextWindow(initData.shareContext, *this,
                                                eq::WindowSettings());

    settings.setSharedContextWindow(_guiWindow);
    /* All windows will go offscreen if there's a main window from Qt. This
       is needed mainly for the main destination channel window, but since it's
       not easy to differentiate the rest, we do it for all */
    settings.setIAttribute(eq::WindowSettings::IATTR_HINT_DRAWABLE, eq::FBO);

    eq::SystemWindow* systemWindow = windowSystem.createWindow(this, settings);
    LBASSERT(systemWindow);
    if (!systemWindow->configInit())
    {
        LBWARN << "System window initialization failed" << std::endl;
        systemWindow->configExit();
        delete systemWindow;
        return false;
    }

    setPixelViewport(systemWindow->getPixelViewport());
    setSystemWindow(systemWindow);
    if (isOnscreen)
    {
        assert(dynamic_cast<eq::qt::Window*>(systemWindow));
        eq::qt::Window* qtSystemWindow =
            static_cast<eq::qt::Window*>(systemWindow);
        config->sendEvent(ConfigEvent::ACTIVE_VIEW_EVENT_PROCESSOR)
            << qtSystemWindow->getEventProcessor();
    }
    return true;
#endif
}

bool Window::configInitGL(const eq::uint128_t&)
{
    /* The base method is not invoked on purpose because:
       - OSG will do its own initialization.
       - The call to swapBuffers triggers a warning in Qt5. */

    osg::GraphicsContext::Traits* traits = new osg::GraphicsContext::Traits();

    /* Setting up traits
       \todo Check if the display settings variable needs to be used here or
       not */

    traits->windowName = eq::Window::getName();
    if (getPipe()->getPort() == LB_UNDEFINED_UINT32)
        traits->displayNum = 0;
    else
        traits->displayNum = getPipe()->getPort();

    if (getPipe()->getDevice() == LB_UNDEFINED_UINT32)
        traits->screenNum = 0;
    else
        traits->screenNum = getPipe()->getDevice();

    const eq::PixelViewport& vp = getPixelViewport();
    traits->x = vp.x;
    traits->y = vp.y;
    traits->width = vp.w;
    traits->height = vp.h;

    traits->red = traits->green = traits->blue =
#if EQ_VERSION_LT(1, 7, 2)
        getIAttribute(IATTR_PLANES_COLOR);
#else
        getIAttribute(eq::WindowSettings::IATTR_PLANES_COLOR);
#endif

    const eq::fabric::DrawableConfig& config = getDrawableConfig();
    traits->alpha = config.alphaBits;
    traits->stencil = config.stencilBits;
    traits->doubleBuffer = config.doublebuffered;

#if EQ_VERSION_LT(1, 7, 2)
    traits->samples = getIAttribute(IATTR_PLANES_SAMPLES);
#else
    traits->samples = getIAttribute(eq::WindowSettings::IATTR_PLANES_SAMPLES);
#endif

    traits->sharedContext =
        getSharedContextWindow() == this
            ? dynamic_cast<const Window*>(getSharedContextWindow())->embedded()
            : 0;
    traits->windowDecoration =
#if EQ_VERSION_LT(1, 7, 2)
        getIAttribute(IATTR_HINT_DECORATION);
#else
        getIAttribute(eq::WindowSettings::IATTR_HINT_DECORATION);
#endif

    _osgWindow = new EmbeddedWindow(this, traits);
#if RTNEURON_USE_CUDA
    /* Unconditionally create a CUDA context to avoid errors related to
       the cudaGLGetDevices bug if Equalizer is restarted. */
    core::CUDAContext::getOrCreateContext(_osgWindow);
#endif

    return true;
}

bool Window::configExitGL()
{
    if (_osgWindow)
    {
        _osgWindow->close(true);
        _osgWindow = 0;
    }
    delete _guiWindow;
    _guiWindow = 0;
    return true;
}
}
}
}
