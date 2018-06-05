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

#ifndef RTNEURON_SCENE_SCENEOPERATIONS_H
#define RTNEURON_SCENE_SCENEOPERATIONS_H

#include "../SceneImpl.h"
#include "SceneOperation.h"

#include <co/object.h>

#include <boost/serialization/assume_abstract.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   A distributed object that stores a vector of callbacks to operate on scenes.

   This class is not thread-safe.
*/
class SceneOperations : public co::Object
{
public:
    /*--- Public member attributes ---*/

    bool isDirty() const { return _dirty; }
    void push(const SceneOperationPtr& operation)
    {
        assert(!isAttached() || isMaster());
        assert(operation);
        _operations.push_back(operation);
        _dirty = true;
    }

    void apply(SubScene& subscene)
    {
        for (const auto& operation : _operations)
            (*operation)(subscene);
    }

    void apply(rtneuron::Scene::_Impl& scene)
    {
        for (const auto& operation : _operations)
            (*operation)(scene);
    }

    /** Extract the operations that match the operation order given into
        a separate object.

        The matching operations are removed from this container. */
    void extract(SceneOperations& operations, SceneOperation::Order order);

    size_t size() const { return _operations.size(); }
    void clear() { _operations.clear(); }
    /*--- Public constructors/destructors ---*/

    SceneOperations()
        : _dirty(false)
    {
    }

protected:
    /*--- Protected member functions ---*/

    virtual ChangeType getChangeType() const { return INSTANCE; }
    virtual void getInstanceData(co::DataOStream& out);

    virtual void applyInstanceData(co::DataIStream& in);

private:
    /*--- Private member attributes ---*/
    typedef std::vector<SceneOperationPtr> Operations;
    Operations _operations;
    bool _dirty;
};
}
}
}
#endif
