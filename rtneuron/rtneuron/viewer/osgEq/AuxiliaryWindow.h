/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef RTNEURON_OSGEQ_AUXILIARYWINDOW_H
#define RTNEURON_OSGEQ_AUXILIARYWINDOW_H

#include <eq/admin/types.h>
#include <eq/config.h>
#include <eq/fabric/pixelViewport.h>

#include <viewer/osgEq/ConfigEvent.h>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class View;
class Client;

/**
   An abstract class for windows that are added to a running Equalizer
   configuration at runtime.

   The window is added using the administrative API from Equalizer.
*/
class AuxiliaryWindow : public boost::noncopyable
{
public:
    /**
       Creates the necessary objects in order to allow the addition of a new
       window in the context of an Equalizer application. The Equalizer
       configuration is automatically updated at the end of the creation
       process.

       @param view Parent view of the additional window
       @param attribute Type of window (WINDOW, FBO, etc)
     */
    AuxiliaryWindow(View* view, const eq::fabric::IAttribute& attribute);

    virtual ~AuxiliaryWindow();

    /**
       Resizes the window according to given width and height. The Equalizer
       configuration is automatically updated after the resizing.

       The results are undefined if the value any of the parameters is 0.

       @param width New width of the window.
       @param height New height of the window
     */
    void resize(const size_t width, const size_t height);

    /**
       Resizes the window according to given scale. The scaling is applied to
       the size of the existing application on-screen window. The Equalizer
       configuration is automatically updated after the resizing.

       The results are undefined if a negative or zero scale factor is applied.

       @param scale Scaling ratio applied to the size of the main on-screen
              window
     */
    void resize(const float scale);

    /**
       Return the window pixel viewport.
    */
    const eq::fabric::PixelViewport& getPixelViewport() const;

    /**
       Sets a wall projection to be used by the internal view.
     */
    void setWall(const eq::Wall& wall);

    /**
       Requests a reconfiguration of the window.

       In particular, the window is resized as needed.

       @throw std::runtime_error if the reconfiguration failed.
     */
    void configure();

    /**
       Render this window.

       Calls configure internally if not done so far.

       @param waitForCompletion Determine if the method should wait until the
              rendering in the additional window is fully processed
     */
    void render(const bool waitForCompletion);

protected:
    View* _mainView;
    eq::Config* _mainConfig;
    bool _configured;

    /* Object living in the Admin part of Equalizer. */
    struct
    {
        eq::admin::Window* window;
        eq::admin::Channel* channel;
        eq::admin::Canvas* canvas;
        eq::admin::Segment* segment;
        eq::admin::Layout* layout;
        eq::admin::View* view;
        eq::admin::Config* config;
    } _admin;

    /**
       Pure virtual method used by the inherited classes to execute the
       specific code needed by the rendering of the additional window
     */
    virtual void configUpdated(eq::Config* config) = 0;

    /**
       Implementation method called from render after the configuration
       has been finished.
    */
    virtual void renderImpl(const bool waitForCompletion) = 0;

private:
    /**
       Sends the RECONFIGURE_REQUEST event to the current config, waits
       for its completion and calls configUpdated()
       @return true if the window is created and the equalizer config
               is updated accordingly, false otherwise.
    */
    bool _configure();
};
}
}
}
#endif
