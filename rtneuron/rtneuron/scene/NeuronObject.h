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

#ifndef RTNEURON_SCENE_NEURON_OBJECT_H
#define RTNEURON_SCENE_NEURON_OBJECT_H

#include "SceneObject.h"
#include "data/SimulationDataMapper.h"
#include "render/NeuronColoring.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class NeuronObject : public SceneObject
{
public:
    class SubSet;

    /*--- Public constructors/destructor ---*/

    /* This constructor adds some required objects to the member attributes
       from the parent scene.
       The code is put here so it's closer to the complementary cleanup
       code. */
    NeuronObject(Neurons&& neurons2, const AttributeMap& attributes,
                 Scene::_Impl* parent);

    ~NeuronObject();

    /*--- Public member functions ---*/

    boost::any getObject() const final;
    const Neurons& getNeurons() const { return _neurons; }
    Neurons& getNeurons() { return _neurons; }
    void cleanup() final;

    void onAttributeChangedImpl(const AttributeMap& attributes,
                                const std::string& name) final;

    void onAttributeChangingImpl(
        const AttributeMap& attributes, const std::string& name,
        const AttributeMap::AttributeProxy& parameters) final;

    rtneuron::Scene::ObjectPtr query(const uint32_ts& ids,
                                     const bool checkIds) final;

    void update(Scene::_Impl&, const AttributeMap& attributes) final;

    using SceneObject::update;

    void dirtySubsets();

    /*--- Protected member functions ---*/
protected:
    void applyAttribute(const AttributeMap& attributes,
                        const std::string& name) final;

    bool preUpdateImplementation() final;

    void invalidateImplementation() final;

private:
    /*--- Private member attributes ---*/
    struct Helpers;

    Neurons _neurons;

    typedef std::shared_ptr<SubSet> SubSetPtr;
    typedef std::weak_ptr<SubSet> SubSetWeakPtr;
    typedef std::vector<SubSetWeakPtr> SubSets;
    SubSets _subsets;
    mutable std::mutex _subsetsMutex;

    RepresentationMode _restrictedMode;
    bool _subsetDirty;
    bool _issueLoadOnUpdate;

    /* Modified only on client side operations */
    struct ClientMembers
    {
        ClientMembers()
            : _coloringDirty(false)
        {
        }
        ClientMembers(const AttributeMap& attributes)
            : _coloringDirty(false)
            , _coloring(attributes)
        {
        }
        bool _coloringDirty;
        NeuronColoring _coloring;
    } _client;

    /*--- Private member functions ---*/
    void _invalidateQueriedHandlers();
};
}
}
}
#endif
