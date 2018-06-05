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

#include "View.h"
#include "Camera.h"
#include "Scene.h"
#include "ViewImpl.h"
#include "ui/TrackballManipulator.h"
#include "viewer/SnapshotWindow.h"

#include <iostream>

namespace bbp
{
namespace rtneuron
{
size_t _readSizeEnvVar(const char* name, const size_t defaultValue)
{
    char* variable = getenv(name);
    if (!variable)
        return defaultValue;
    const long int value = strtol(variable, &variable, 10);
    if (value <= 0 || variable[0] != '\0')
        return defaultValue;
    return value;
}

const size_t MAX_FRAMEBUFFER_WIDTH =
    _readSizeEnvVar("RTNEURON_MAX_FB_WIDTH", 16384);
const size_t MAX_FRAMEBUFFER_HEIGHT =
    _readSizeEnvVar("RTNEURON_MAX_FB_HEIGHT", 16384);

namespace
{
/**
   Helper class for tiled rendering.

   This class takes the frustum definition from a camera and a desired output
   image size and based on these, computes the wall definition to be applied
   to a SnapshotWindow given the tile parameters.
   The calculation takes into account the different aspect ratios of the
   original frustum and the output window.
*/
class TileFrustum
{
public:
    TileFrustum(Camera& camera, const size_t width, const size_t height)
        : _near(0)
        , _isOrtho(camera.isOrtho())
    {
        float right;
        float bottom;
        if (camera.isOrtho())
            camera.getProjectionOrtho(_left, right, bottom, _top);
        else
            camera.getProjectionFrustum(_left, right, bottom, _top, _near);

        const float desiredAspectRatio = width / float(height);
        const float aspectRatio = (right - _left) / (_top - bottom);
        const float ratio = desiredAspectRatio / aspectRatio;
        _left *= ratio;
        right *= ratio;

        _width = right - _left;
        _height = _top - bottom;
    }

    /**
       Apply a wall definition to a SnapshotWindow to render the given tile.

       Tile coordinate origin is at the top left corner.
    */
    void applyTile(const float x, const float width, const float y,
                   const float height, SnapshotWindow& window)
    {
        const float left = _left + _width * x;
        const float right = _left + _width * (x + width);
        const float top = _top - _height * y;
        const float bottom = _top - _height * (y + height);

        eq::Wall wall;
        wall.bottomLeft = eq::Vector3f(left, bottom, -_near);
        wall.bottomRight = eq::Vector3f(right, bottom, -_near);
        wall.topLeft = eq::Vector3f(left, top, -_near);
        wall.type = eq::Wall::TYPE_FIXED;
        window.setWall(wall);
    }

private:
    float _left;
    float _width;
    float _top;
    float _height;
    float _near;
    bool _isOrtho;
};

void _tiledSnapshot(SnapshotWindow& window, Camera& camera,
                    const std::string& filename, const size_t width,
                    const size_t height)
{
    TileFrustum frustum(camera, width, height);

    size_t position = filename.find_last_of('.');
    std::string prefix = filename.substr(0, position);
    std::string extension =
        position == std::string::npos ? "" : filename.substr(position);

    const double maxTileWidth = MAX_FRAMEBUFFER_WIDTH / double(width);
    const double maxTileHeight = MAX_FRAMEBUFFER_HEIGHT / double(height);
    double x = 0;
    size_t column = 0;
    for (size_t i = 0; i < width;
         ++column, i += MAX_FRAMEBUFFER_WIDTH, x += maxTileWidth)
    {
        const float tileWidth = std::min(maxTileWidth, 1.0 - x);
        const size_t pixelTileWidth =
            std::min(MAX_FRAMEBUFFER_WIDTH, width - i);

        double y = 0;
        size_t row = 0;
        for (size_t j = 0; j < height;
             ++row, j += MAX_FRAMEBUFFER_HEIGHT, y += maxTileHeight)
        {
            const float tileHeight = std::min(maxTileHeight, 1.0 - y);
            const size_t pixelTileHeight =
                std::min(MAX_FRAMEBUFFER_HEIGHT, height - j);

            window.resize(pixelTileWidth, pixelTileHeight);
            /* The auxiliary window needs to be reconfigured before changing
               the frustum, otherwise render will trigger the reconfiguration
               and despite the frustum parameters have been applied to the
               admin view, the output channel is not getting the correct wall
               specification. */
            window.configure();
            /* Adjusting the view wall to the position and dimensions of
               the current tile */
            frustum.applyTile(x, tileWidth, y, tileHeight, window);

            /* Assigning the tile name */
            std::stringstream tileName;
            tileName << prefix << row << '-' << column << extension;
            window.setFilename(tileName.str());

            window.render(true);
        }
    }
}
}

/*
  Constructors
*/
View::View(osgEq::View* view, const RTNeuronPtr& application,
           const AttributeMap& style)
    : _impl(new _Impl(view, style))
    , _application(application)
{
    _impl->setCameraManipulator(
        CameraManipulatorPtr(new TrackballManipulator()));
}

/*
   Destructor
*/
View::~View()
{
    delete _impl; /* deleting 0 is well defined. */
}

/*
  Member functions
*/

AttributeMap& View::getAttributes()
{
    _checkValid();
    return _impl->getAttributes();
}

const AttributeMap& View::getAttributes() const
{
    _checkValid();
    return _impl->getAttributes();
}

void View::setScene(const ScenePtr& scene)
{
    _checkValid();
    if (scene && scene->_application.lock() != _application.lock())
    {
        LBTHROW(
            std::runtime_error("Can't assign a scene to a view "
                               "from a different RTNeuron application"));
    }
    _impl->setScene(scene);
}

ScenePtr View::getScene() const
{
    _checkValid();
    return _impl->_scene;
}

CameraPtr View::getCamera() const
{
    _checkValid();
    return _impl->_camera;
}

void View::setCameraManipulator(const CameraManipulatorPtr& manipulator)
{
    _checkValid();
    _impl->setCameraManipulator(manipulator);
}

CameraManipulatorPtr View::getCameraManipulator() const
{
    _checkValid();
    return _impl->getCameraManipulator();
}

void View::computeHomePosition()
{
    _checkValid();
    _impl->computeHomePosition();
}

void View::setPointer(const PointerPtr& pointer)
{
    _checkValid();
    _impl->setPointer(pointer);
}

PointerPtr View::getPointer()
{
    _checkValid();
    return _impl->getPointer();
}

void View::record(bool enable)
{
    _checkValid();
    _impl->record(enable);
}

void View::grabFrame()
{
    _checkValid();
    _impl->grabFrame();
}

void View::snapshot(const std::string& fileName, const bool waitForCompletion)
{
    _checkValid();

    if (fileName.empty())
        throw std::runtime_error(
            "A file name for the snapshot "
            "must be specified.");

    _impl->snapshot(fileName, waitForCompletion);
}

void View::snapshot(const std::string& fileName, const float scale)
{
    _checkValid();

    if (fileName.empty())
        throw std::invalid_argument(
            "A file name for the snapshot must be specified.");
    if (scale <= 0)
        throw std::invalid_argument(
            "The window scale factor must be positive.");

    SnapshotWindow window(_impl->getEqView(), eq::fabric::FBO);
    const eq::fabric::PixelViewport& pvp = window.getPixelViewport();
    if (pvp.w * scale > MAX_FRAMEBUFFER_WIDTH ||
        pvp.h * scale > MAX_FRAMEBUFFER_HEIGHT)
    {
        _tiledSnapshot(window, *getCamera(), fileName, pvp.w * scale,
                       pvp.h * scale);
    }
    else
    {
        window.resize(scale);
        window.setFilename(fileName);
        window.render(true);
    }
}

void View::snapshot(const std::string& fileName, const size_t resX,
                    const size_t resY)
{
    _checkValid();

    if (fileName.empty())
        throw std::invalid_argument(
            "A file name for the snapshot must be specified.");
    if (resX == 0 || resY == 0)
        throw std::invalid_argument(
            "Both horizontal and vertical resolutions must be greater than 0.");

    SnapshotWindow window(_impl->getEqView(), eq::fabric::FBO);
    if (resX > MAX_FRAMEBUFFER_WIDTH || resY > MAX_FRAMEBUFFER_HEIGHT)
    {
        _tiledSnapshot(window, *getCamera(), fileName, resX, resY);
    }
    else
    {
        window.resize(resX, resY);
        window.setFilename(fileName);
        window.render(true);
    }
}

osgEq::View* View::getEqView()
{
    _checkValid();
    return _impl->getEqView();
}

const osgEq::View* View::getEqView() const
{
    _checkValid();
    return _impl->getEqView();
}

void View::disconnectAllSlots()
{
    if (!_impl)
        return;
    _impl->getAttributes().disconnectAllSlots();
    FrameGrabbedSignal().swap(frameGrabbed);
}

void View::_invalidate()
{
    delete _impl;
    _impl = 0;
    _application.reset();
}

void View::_checkValid() const
{
    if (_impl == 0)
        LBTHROW(std::runtime_error("Invalid operation on View object."));
}
}
}
