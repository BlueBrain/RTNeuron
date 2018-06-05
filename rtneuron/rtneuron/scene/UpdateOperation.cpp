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

#include "UpdateOperation.h"
#include "SceneObject.h"
#include "SubScene.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
UpdateOperation::UpdateOperation()
    : _objectID(-1)
{
}

UpdateOperation::UpdateOperation(const SceneObject& object)
    : _objectID(object.getID())
    , _attributes(object.getAttributes())
{
}

void UpdateOperation::operator()(SubScene& subScene) const
{
    Scene::_Impl::BaseObjectPtr object =
        subScene.getScene().getObjectByID(_objectID);
    if (!object)
    {
        /* Some operations can be applied at the scene level on temporary
           objects that are removed before the operations are applied at the
           subscene level.
           We don't give any warning in this case. */
        return;
    }
    static_cast<ClientSceneObject&>(*object).update(subScene, _attributes);
}

void UpdateOperation::operator()(Scene::_Impl& scene) const
{
    Scene::_Impl::BaseObjectPtr object = scene.getObjectByID(_objectID);
    if (!object)
    {
        LBDEBUG << "Cannot apply update operation on non existing object"
                << std::endl;
        return;
    }
    static_cast<ClientSceneObject&>(*object).update(scene, _attributes);
}

template <class Archive>
void UpdateOperation::serialize(Archive& ar, const unsigned int /* version */)
{
    ar& boost::serialization::base_object<SceneOperation>(*this);
    ar& _objectID;
    ar& _attributes;
}

template void UpdateOperation::serialize<net::DataOStreamArchive>(
    net::DataOStreamArchive&, const unsigned int);

template void UpdateOperation::serialize<net::DataIStreamArchive>(
    net::DataIStreamArchive&, const unsigned int);
}
}
}

#include <boost/serialization/export.hpp>

BOOST_CLASS_EXPORT(bbp::rtneuron::core::UpdateOperation)
