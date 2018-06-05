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

#ifndef RTNEURON_LODNEURONMODELDRAWABLE_H
#define RTNEURON_LODNEURONMODELDRAWABLE_H

#include <osg/Drawable>
#include <osg/Version>

#include "scene/LODNeuronModel.h"

namespace osgUtil
{
class CullVisitor;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! Proxy drawable provided LODNeuronModel
/** This proxy drawable contains a list of drawable objects which represent
    different LOD of the same neuron. The list of LODs is constructed by a
    LODNeuronModel as NeuronModels are added to it.

    During the cull traversal, an internal cull callback selects
    those drawables to pass to the render stage. The selection uses an
    approximation of the screen projected area of the bounding
    box of the candidate drawables. The visbility range for each LOD is
    established at the moment it is added.
    \sa NeuronModel::createNeuronModels implementation.
*/
class LODNeuronModel::Drawable : public osg::Drawable
{
public:
    /*--- Public declarations ---*/

    class CullCallback;
    class UpdateCallback;
    friend class LODNeuronModel;

    /*--- Public constructors and destructor ---*/

    Drawable();

    Drawable(const Drawable& drawable,
             const osg::CopyOp& op = osg::CopyOp::SHALLOW_COPY);

    META_Object(rtneuron, Drawable);

    virtual ~Drawable() {}
    const LODNeuronModel& getModel() const { return *_model; }
    /*--- Public member fuctions ---*/

    void drawImplementation(osg::RenderInfo&) const {};

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    virtual void releaseGLObjects(osg::State* state) const;

private:
    /*--- Private member attributes ---*/

    LODNeuronModel* _model;

    /*--- Private member functions ---*/

    void _setSphericalSomaVisibility(osgUtil::CullVisitor* visitor,
                                     bool visible);
};
}
}
}
#endif
