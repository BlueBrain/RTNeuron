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

#ifndef RTNEURON_SPHERESET_H
#define RTNEURON_SPHERESET_H

#include "../render/SceneStyle.h"
#include "coreTypes.h"

#include <osg/Vec4>

namespace osg
{
class Node;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   A fast and compact class to render spheres.
*/
class SphereSet
{
public:
    /*--- Public declarations ---*/

    class SubSetID_;
    typedef SubSetID_* SubSetID;

    /*--- Public constructors/destructor ---*/

    /**
       @param styleType Defines the shader set that will be requested
       to SceneStyle when applyStyle is called.
    */
    SphereSet(SceneStyle::StateType styleType = SceneStyle::SPHERES,
              size_t maxSpheresPerSubset = 200000);

    ~SphereSet();

    SphereSet(const SphereSet&) = delete;
    SphereSet& operator=(const SphereSet&) = delete;

    /*--- Public member functions ---*/

    /**
       Add a sphere to this sphere set.

       Thread-safe with concurrent radius, color and attribute updates.
    */
    SubSetID addSphere(const osg::Vec3& position, float radius,
                       const osg::Vec4& color);

    /**
       Add a list of spheres to this sphere set.

       Thread-safe with concurrent radius, color and attribute updates.
    */
    SubSetID addSpheres(const std::vector<osg::Vec3>& positions, float radius,
                        const osg::Vec4& color);

    /**
        Update the radius of a sphere subset.

        Thread-safe with concurrent sphere additions.
    */
    void updateRadius(SubSetID id, float radius, bool dirty = true);

    /**
        Update the color of a sphere subset.

        Thread-safe with concurrent sphere additions.
    */
    void updateColor(SubSetID id, const osg::Vec4& color, bool dirty = true);

    /**
        Update the vertex attribute of the given index in a sphere set.

        If the vertex attribute array does not exist yet it is created.

        It's the user responsability to make sure all subsets from this
        SphereSet have a defined value for the attribute array.
        The attribute value will be undefined for spheres sub sets that share
        the underlying osg::Geometry with this one if updateAttribute is not
        called for them.

        Thread-safe with concurrent sphere additions.
    */
    template <typename T>
    void updateAttribute(SubSetID id, unsigned int index, const T& value,
                         bool dirty = true);

    void dirtyUpdatedArrays();

    /** Not thread-safe with concurrent updates and additions. */
    void remove(SubSetID id);

    /** Not thread-safe with concurrent updates and additions. */
    void clear();

    osg::Node* getNode();

    void applyStyle(const SceneStylePtr& style);

private:
    /*--- Private member variables ---*/
    class Impl;
    Impl* _impl;
};
}
}
}
#endif
