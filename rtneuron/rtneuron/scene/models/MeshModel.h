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

#ifndef RTNEURON_MESHMODEL_H
#define RTNEURON_MESHMODEL_H

#include "NeuronParts.h"
#include "SkeletonModel.h"
#include "render/DrawElementsPortions.h"

#include <osg/FrontFace>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
//! Mesh model for a neuron.
/** This is an internal implementation model of DetailedNeuronModel that
    handles a regular triangular mesh. */
class MeshModel : public model::SkeletonModel
{
public:
    /*--- Public constructors/destructor ---*/
    MeshModel(NeuronParts parts, const ConstructionData& data);

    /*--- Public member functions ---*/
    void clip(const Planes& planes);

    Skeleton::PortionRanges postProcess(NeuronSkeleton& skeleton);

    const osg::BoundingBox& getOrCreateBound(const Neuron& neuron) final;

    osg::StateSet* getModelStateSet(const bool subModel,
                                    const SceneStyle& style) const final;

    Drawable* instantiate(const SkeletonPtr& skeleton,
                          const CircuitSceneAttributes& sceneAttr);

private:
    /*--- Private member variables ---*/

    struct
    {
        Vector3fsPtr _vertices;
        Vector3fsPtr _normals;
        uint32_tsPtr _indices; /* May be a triangle soup or strip */
    } _custodians; /* This variable is used to keep a copy of the mesh
                      data when needed (conservative loading) */

    osg::ref_ptr<osg::Array> _normals;

    DrawElementsPortions::IndexArray _indices;
    size_t _primitiveLength;
    osg::PrimitiveSet::Mode _mode;
    Skeleton::PortionRanges _ranges;
    osg::ref_ptr<DrawElementsPortions> _primitiveEBOOwner;

    bool _useDefaultCull;

    /* Needed while v1 meshes are used (is it still the case?) */
    bool _clockWiseWinding;

    /*--- Private member functions ---*/
    void _createSomaMesh(const ConstructionData& data);
    void _createSomaDendriteMesh(const ConstructionData& data);
    void _createFullMesh(const ConstructionData& data);
    void _createFullMeshPrimitive(const ConstructionData& data);
};
}
}
}
}
#endif
