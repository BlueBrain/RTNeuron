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

#ifndef RTNEURON_SCENE_CLIENTSCENEOBJECT_H
#define RTNEURON_SCENE_CLIENTSCENEOBJECT_H

#include "../SceneImpl.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   This class provides methods called during the client-side dispatching
   of the update operation queued by Scene::Object::update.
*/
class ClientSceneObject : public Scene::_Impl::BaseObject
{
public:
    ClientSceneObject(Scene::_Impl* parent)
        : BaseObject(parent)
    {
    }

    ClientSceneObject(Scene::_Impl* parent, unsigned int id)
        : BaseObject(parent, id)
    {
    }

    /** Realizes the object update at a subscene */
    virtual void update(SubScene& subscene, const AttributeMap& attributes) = 0;

    /** Realizes the object update at the main scene object. */
    virtual void update(Scene::_Impl& scene,
                        const AttributeMap& attributes) = 0;
};
}
}
}
#endif
