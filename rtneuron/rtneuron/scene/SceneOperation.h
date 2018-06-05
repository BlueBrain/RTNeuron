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

#ifndef RTNEURON_SCENE_SCENEOPERATION_H
#define RTNEURON_SCENE_SCENEOPERATION_H

#include "../Scene.h"
#include "../types.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class SubScene;

class SceneOperation
{
public:
    /** Indicates the moment at which this operation has to be applied during
        scene update. */
    enum class Order
    {
        /** Apply the operation before the subscenes are updated */
        preUpdate = 0,
        /** Apply the operation after the subscenes are updated */
        postUpdate
    };

    friend class boost::serialization::access;

    virtual ~SceneOperation(){};

    virtual void operator()(SubScene& subScene) const = 0;

    virtual void operator()(rtneuron::Scene::_Impl& scene) const = 0;

    Order getOperationOrder() const { return _order; }
protected:
    SceneOperation(const Order order = Order::preUpdate)
        : _order(order)
    {
    }

private:
    Order _order;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar& _order;
    }
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(bbp::rtneuron::core::SceneOperation)
}
}
}
#endif
