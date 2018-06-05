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

#include "SceneObject.h"
#include "SceneObjectOperation.h"
#include "UpdateOperation.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
SceneObject::SceneObject(Scene::_Impl* parent, const AttributeMap& attributes)
    : ClientSceneObject(parent)
    , Configurable(attributes)
    , _current(attributes)
    , _currentHash(_current.hash())
{
    _current.attributeMapChanged.connect(
        boost::bind(&SceneObject::applyAttribute, this, _1, _2));
}

void SceneObject::apply(const Scene::ObjectOperationPtr& operation)
{
    if (!operation)
        throw std::invalid_argument("Null pointer");

    if (!operation->accept(*this))
        throw std::runtime_error("Invalid operation on object.");

    lunchbox::ScopedWrite mutex(_parentLock);
    if (!_parent)
        throw std::runtime_error("Invalid object handler.");
    _parent->pushOperation(operation->getImplementation());
    _parent->dirty(true);
};

void SceneObject::update()
{
    lunchbox::ScopedWrite mutex(_parentLock);
    if (!_parent)
        throw std::runtime_error("Invalid object handler.");

    if (!preUpdateImplementation() && getAttributes().hash() == _currentHash)
        /* The attributes haven't changed */
        return;

    _currentHash = getAttributes().hash();
    _parent->pushOperation(SceneOperationPtr(new UpdateOperation(*this)));
    _parent->dirty(true);
}

void SceneObject::update(Scene::_Impl&, const AttributeMap& attributes)
{
    _current.merge(attributes);
}
}
}
}
