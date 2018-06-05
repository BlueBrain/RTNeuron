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

#ifndef RTNEURON_CYLINDERBASEDMODEL_H
#define RTNEURON_CYLINDERBASEDMODEL_H

#include "NeuronParts.h"
#include "SkeletonModel.h"
#include "render/DrawElementsPortions.h"
#include "utils.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
//! Pseudo-cylindrical based model
/** This is an internal implementation model of DetailedNeuronModel
    that represents each morphological segment with a screen-aligned quad
    shaded as generalized-cylinder.
*/
class CylinderBasedModel : public model::SkeletonModel
{
    /* Constructor */
public:
    CylinderBasedModel(NeuronParts parts, const ConstructionData& data,
                       const float maxSquaredDistanceToMorphology,
                       const bool supportCLODHint);

    /* Member functions */
public:
    virtual void clip(const Planes& planes);

    Skeleton::PortionRanges postProcess(NeuronSkeleton& skeleton);

    virtual osg::StateSet* getModelStateSet(const bool subModel,
                                            const SceneStyle& style) const;

    Drawable* instantiate(const SkeletonPtr& skeleton,
                          const CircuitSceneAttributes& sceneAttr);

    /* Member attributes */
protected:
    osg::ref_ptr<osg::Vec4Array> _tangentsAndThickness;
    boost::shared_array<GLuint> _indices; /* quad strip indices */
    size_t _primitiveLength;

    Skeleton::PortionRanges _ranges;
    osg::ref_ptr<DrawElementsPortions> _primitiveEBOOwner;

    bool _supportCLOD;

    void _create(const ConstructionData& data,
                 const float maxSquaredDistanceToMorphology, const bool noAxon);
    void _allocate(const size_t numPoints);
    void _createSection(const brain::neuron::Section& section,
                        const ConstructionData& data,
                        const float maxSquaredDistanceToMorphology,
                        util::SectionStartPointsMap& sectionStarts);

    size_t _startSection(const brain::neuron::Section& section,
                         const brain::Vector4fs& samples,
                         const brain::floats& relativeDistances,
                         const ConstructionData& data,
                         util::SectionStartPointsMap& sectionStarts);
};
}
}
}
}
#endif
