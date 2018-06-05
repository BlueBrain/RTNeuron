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

#ifndef RTNEURON_CAMERADATA_H
#define RTNEURON_CAMERADATA_H

#include <lunchbox/types.h>

#include <osg/Viewport>

namespace osg
{
class Camera;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
class CircuitScene;
class ViewStyleData;

/**
   \brief Additional information of a camera such as:
   - View style data that can be shared between cameras.

   \todo
   - A reference to a CircuitScene.

   This object goes as the user data of the osg::Camera object returned
   by RenderInfo::getCurrentCamera
*/
class CameraData : public osg::Referenced
{
    /* Constructor */
public:
    CameraData();

    /* Destructor */
public:
    ~CameraData();

    /* Member functions */
public:
    static CameraData* getOrCreateCameraData(osg::Camera* camera);

    /**
       The ID of the circuit scene being rendered by the camera who owns
       these data.
    */
    uint32_t getCircuitSceneID() const { return _sceneID; }
    void setCircuitScene(CircuitScene* scene);

public:
    osg::ref_ptr<ViewStyleData> viewStyle;

protected:
    uint32_t _sceneID;
};
}
}
}
#endif
