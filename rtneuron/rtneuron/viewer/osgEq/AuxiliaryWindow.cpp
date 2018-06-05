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

#include <QApplication>

#include "AuxiliaryWindow.h"

#include "Config.h"
#include "ConfigEvent.h"
#include "View.h"

#include <eq/admin/admin.h>
#include <eq/config.h>
#include <eq/fabric/error.h>
#include <eq/fabric/iAttribute.h>

#include <QThread>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class FindViewWindows : public eq::admin::NodeVisitor
{
public:
    FindViewWindows(eq::View* view)
        : _view(view)
    {
    }
    virtual ~FindViewWindows() {}
    virtual eq::admin::VisitorResult visit(eq::admin::Channel* channel)
    {
        if (channel->getViewVersion().identifier != _view->getID())
            return eq::admin::TRAVERSE_CONTINUE;

        eq::admin::Window* window = channel->getWindow();
        if (_windows.empty())
        {
            _windows.push_back(window);
            return eq::admin::TRAVERSE_CONTINUE;
        }

        /* Store all on-screen windows */
        if (window->getIAttribute(
                eq::fabric::WindowSettings::IATTR_HINT_DRAWABLE) !=
            eq::fabric::WINDOW)
        {
            return eq::admin::TRAVERSE_CONTINUE;
        }

        /* If first element was an off-screen window,
           replace it by the on-screen one just found */
        if (_windows.at(0)->getIAttribute(
                eq::fabric::WindowSettings::IATTR_HINT_DRAWABLE) !=
            eq::fabric::WINDOW)
            _windows.at(0) = window;
        else
            _windows.push_back(window);

        return eq::admin::TRAVERSE_CONTINUE;
    }

    const eq::admin::Windows& getResult() const { return _windows; }
private:
    eq::View* const _view;
    eq::admin::Windows _windows;
};

AuxiliaryWindow::AuxiliaryWindow(osgEq::View* view,
                                 const eq::fabric::IAttribute& attribute)
    : _mainView(view)
    , _configured(false)
{
    _mainConfig = view->getConfig();
    LBASSERT(_mainConfig);

    /* Code taken from Equalizer addWindow */

    /* Find the first pipe */
    _admin.config = static_cast<Config*>(_mainConfig)->getAdminConfig();

    const eq::admin::Nodes& nodes = _admin.config->getNodes();
    if (nodes.empty())
        LBTHROW(std::runtime_error("No nodes defined in the configuration"));

    const eq::admin::Node* node = nodes.front();

    FindViewWindows viewWindows(view);
    node->accept(viewWindows);

    const eq::admin::Windows& sourceWindows = viewWindows.getResult();
    LBASSERT(!sourceWindows.empty());
    eq::admin::Window* sourceWindow = sourceWindows.front();
    LBASSERT(sourceWindow);

    /* Adding Window */
    eq::admin::Pipe* pipe = sourceWindow->getPipe();
    _admin.config = pipe->getConfig();

    _admin.window = new eq::admin::Window(pipe);
    _admin.channel = new eq::admin::Channel(_admin.window);
    _admin.channel->setName("aux channel");
    _admin.canvas = new eq::admin::Canvas(_admin.config);
    _admin.segment = new eq::admin::Segment(_admin.canvas);
    _admin.layout = new eq::admin::Layout(_admin.config);
    _admin.view = new eq::admin::View(_admin.layout);

    if (sourceWindows.size() > 1)
        LBWARN << "More than one window was found. The new window "
                  "will be created from: "
               << sourceWindow->getName() << std::endl;

    const size_t width = sourceWindow->getPixelViewport().w;
    const size_t height = sourceWindow->getPixelViewport().h;

    _admin.window->setSettings(sourceWindow->getSettings());
    _admin.window->setIAttribute(
        eq::fabric::WindowSettings::IATTR_HINT_DRAWABLE, attribute);
    _admin.window->setPixelViewport(
        eq::fabric::PixelViewport(0, 0, width, height));

    _admin.segment->setChannel(_admin.channel);
    _admin.canvas->addLayout(_admin.layout);

    _admin.config->commit();
}

AuxiliaryWindow::~AuxiliaryWindow()
{
    delete _admin.view;
    delete _admin.segment;
    delete _admin.canvas;
    delete _admin.layout;
    delete _admin.channel;
    delete _admin.window;

    _admin.config->commit();

    _mainConfig->sendEvent(osgEq::ConfigEvent::RECONFIGURE_REQUEST) << false;
}

void AuxiliaryWindow::resize(const size_t width, const size_t height)
{
    const eq::PixelViewport& viewport = _admin.window->getPixelViewport();
    if (viewport.w == (int)width && viewport.h == (int)height)
        return;

    _configured = false;

    _admin.window->setPixelViewport(
        eq::fabric::PixelViewport(0, 0, width, height));

    eq::Wall viewWall = _mainView->getWall();
    viewWall.resizeHorizontalToAR((float)width / height);
    _admin.view->setWall(viewWall);

    _admin.config->commit();
}

void AuxiliaryWindow::resize(const float scale)
{
    const size_t width = _admin.window->getPixelViewport().w * scale;
    const size_t height = _admin.window->getPixelViewport().h * scale;

    resize(width, height);
}

const eq::fabric::PixelViewport& AuxiliaryWindow::getPixelViewport() const
{
    return _admin.window->getPixelViewport();
}

void AuxiliaryWindow::setWall(const eq::Wall& wall)
{
    if (_configured)
    {
        osgEq::View* view = static_cast<osgEq::View*>(
            _mainConfig->find<eq::View>(_admin.view->getID()));
        view->setWall(wall);
    }
    else
    {
        _admin.view->setWall(wall);
        _admin.config->commit();
    }
}

void AuxiliaryWindow::configure()
{
    if (!_configured && !_configure())
    {
        std::stringstream s;
        const eq::fabric::Errors& errors = _mainConfig->getErrors();
        for (const auto& error : errors)
        {
            if (error == eq::fabric::ERROR_FRAMEBUFFER_INVALID_SIZE)
            {
                LBTHROW(std::runtime_error(
                    "The resolution specified "
                    "is too large. No action was performed"));
            }
            else
                s << error << std::endl;
        }
        if (!errors.empty())
            LBTHROW(std::runtime_error(s.str()));
    }
    _configured = true;
}

void AuxiliaryWindow::render(const bool waitForCompletion)
{
    if (!_configured)
        configure();

    renderImpl(waitForCompletion);
}

bool AuxiliaryWindow::_configure()
{
    /* Create the request on which we will wait until the RECONFIGURE_REQUEST
       event is processed by main application loop. */
    lunchbox::Request<bool> request =
        _mainConfig->getLocalNode()->registerRequest<bool>();
    /* Posting the event */
    _mainConfig->sendEvent(osgEq::ConfigEvent::RECONFIGURE_REQUEST) << true
                                                                    << request;
    /* Wait for RECONFIGURE_REQUEST to be processed giving Qt the opportunity
       to dispatch events, otherwise the Equalizer window can't be created. */
    bool success;
    while (true)
    {
        try
        {
            success = request.wait(500);
            break;
        }
        catch (const lunchbox::FutureTimeout&)
        {
            QCoreApplication* app = QApplication::instance();
            if (app)
            {
                assert(QThread::currentThread() == app->thread());
                QApplication::instance()->processEvents();
            }
        }
    }
    if (success)
        configUpdated(_mainConfig);
    return success;
}
}
}
}
