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

#include "CameraData.h"

#include "render/ViewStyleData.h"
#include "scene/CircuitScene.h"

#include <lunchbox/log.h>
#include <osg/Camera>

namespace bbp
{
namespace rtneuron
{
namespace core
{
CameraData::CameraData()
    : _sceneID(0)
{
}

CameraData::~CameraData()
{
}

CameraData* CameraData::getOrCreateCameraData(osg::Camera* camera)
{
    osg::Referenced* data = camera->getUserData();
    CameraData* cameraData = dynamic_cast<CameraData*>(camera->getUserData());
    if (data != 0 && cameraData == 0)
    {
        LBTHROW(
            std::runtime_error("Unexpected user data attached to "
                               "osg::Camera"));
    }
    else if (cameraData == 0)
    {
        cameraData = new CameraData();
        camera->setUserData(cameraData);
    }
    return cameraData;
}

void CameraData::setCircuitScene(CircuitScene* scene)
{
    if (scene == 0)
        _sceneID = 0;
    else
        _sceneID = scene->getID();
}
}
}
}
