/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Jafet Villafrance <jafet.villafrancadiaz@epfl.ch>
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
#ifndef RTNEURON_RENDER_DEPTHOFFIELD_H
#define RTNEURON_RENDER_DEPTHOFFIELD_H

#include "config/constants.h"
#include "data/loaders.h"

#include <osg/PolygonMode>
#include <osg/Texture2D>
#include <osg/ref_ptr>

namespace bbp
{
namespace rtneuron
{
namespace core
{
typedef std::pair<osg::Camera*, osg::Texture2D*> RTTPair;

/**
 * Class that implements the depth-of-field post-processing effect, based on
 * the OpenSceneGraph 3.0 Cookbook reference (Chapter 6)
 */
class DepthOfField
{
public:
    /*--- Public constructors/destructor ---*/

    DepthOfField(osg::Camera* camera);
    ~DepthOfField();

    /*--- Public member functions ---*/

    void update();

    void setFocalDistance(float focalDistance);
    void setFocalRange(float focalRange);

private:
    int _width;
    int _height;

    osg::ref_ptr<osg::Geode> _screenQuad;
    osg::ref_ptr<osg::Camera> _orthoCamera;
    osg::Camera::RenderTargetImplementation _originalRenderTargetImpl;
    osg::ref_ptr<osg::Camera> _camera;
    RTTPair _pairs[2];
    osg::ref_ptr<osg::Texture2D> _textures[3];
};
}
}
}
#endif
