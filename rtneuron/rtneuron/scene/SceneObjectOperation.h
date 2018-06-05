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

#ifndef RTNEURON_SCENE_SCENEOBJECTOPERATION_H
#define RTNEURON_SCENE_SCENEOBJECTOPERATION_H

#include "../Scene.h"
#include "SceneOperation.h"

namespace bbp
{
namespace rtneuron
{
class Scene::ObjectOperation::Impl : public core::SceneOperation
{
public:
    virtual void operator()(core::SubScene& subScene) const = 0;

    virtual void operator()(rtneuron::Scene::_Impl& scene) const = 0;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int /* version */)
    {
        ar& boost::serialization::base_object<SceneOperation>(*this);
    }

protected:
    Impl(const Order order = Order::preUpdate)
        : SceneOperation(order)
    {
    }
};
}
}
#endif
