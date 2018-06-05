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

#ifndef RTNEURON_SCENE_SCENEOBJECT_H
#define RTNEURON_SCENE_SCENEOBJECT_H

#include "ClientSceneObject.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Base abstract interface for all scene objects
*/
class SceneObject : public ClientSceneObject,
                    public rtneuron::Scene::Object,
                    protected detail::Configurable
{
public:
    /*--- Public member functions ---*/

    AttributeMap& getAttributes() final
    {
        return Configurable::getAttributes();
    }

    const AttributeMap& getAttributes() const
    {
        return Configurable::getAttributes();
    }

    /** Queue a scene operation with the attribute map to be applied. */
    void update() final;

    void apply(const Scene::ObjectOperationPtr& operation) final;

    void onAttributeChangedImpl(const AttributeMap&,
                                const std::string&) override
    {
    }

    /* Used only for attribute verification.
       The implementation mustn't have any side effect. */
    virtual void onAttributeChangingImpl(
        const AttributeMap& attributes, const std::string& name,
        const AttributeMap::AttributeProxy& parameters) = 0;

    void update(SubScene&, const AttributeMap&) override {}
    /** Merge the input attributes with _current */
    void update(Scene::_Impl& scene, const AttributeMap& attributes) override;

    /** When the scene is reconstructed, all subset handlers are overriden
        by the parent handler attributes.
        This function needs to be called on all parent handlers to ensure
        that when update() is called in any subset handler the operation
        is effective. */
    virtual void dirtySubsets() {}
protected:
    /*--- Protected member attributes ---*/

    /* The attributes currently applied at the client side. */
    AttributeMap _current;
    uint32_t _currentHash;

    /*--- Protected constructors/destructor ---*/

    SceneObject(Scene::_Impl* parent, const AttributeMap& attributes);

    /*--- Protected member functions ---*/

    /** Called from update(Scene::_Impl&, AttributeMap&) to apply
        attributes one by one. */
    virtual void applyAttribute(const AttributeMap& /* attributes */,
                                const std::string& /* name */){};

    virtual bool preUpdateImplementation() { return false; }
};
}
}
}
#endif
