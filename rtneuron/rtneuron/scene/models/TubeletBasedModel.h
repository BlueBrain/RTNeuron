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

#ifndef RTNEURON_TUBELETBASEDMODEL_H
#define RTNEURON_TUBELETBASEDMODEL_H

#include "NeuronParts.h"
#include "SkeletonModel.h"
#include "render/DrawElementsPortions.h"
#include "utils.h"

#include <brain/neuron/types.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
//! Tubelet based model
/** This is an internal implementation model of DetailedNeuronModel
    that represents each morphological segment with a spherically-capped
    conical frustum. Segments of the same section or connected in seamless
    fashion.
*/
class TubeletBasedModel : public SkeletonModel
{
    /* Constructor */
public:
    TubeletBasedModel(NeuronParts parts, const ConstructionData& data);

    /* Member functions */
public:
    void clip(const Planes& planes);

    Skeleton::PortionRanges postProcess(NeuronSkeleton& skeleton);

    virtual osg::StateSet* getModelStateSet(const bool subModel,
                                            const SceneStyle& style) const;

    Drawable* instantiate(const SkeletonPtr& skeleton,
                          const CircuitSceneAttributes& attributes);

    /* Member attributes */
private:
    osg::ref_ptr<osg::FloatArray> _pointRadii;
    osg::ref_ptr<osg::Vec4Array> _cutPlanes;
    boost::shared_array<GLuint> _indices; /* indices */
    size_t _primitiveLength;

    Skeleton::PortionRanges _ranges;
    osg::ref_ptr<DrawElementsPortions> _primitiveEBOOwner;
    bool _useCLOD;

    void _create(NeuronParts parts, const ConstructionData& data);
    void _allocate(const size_t numPoints);
    void _createSection(const brain::neuron::Section& section,
                        const ConstructionData& data,
                        util::SectionStartPointsMap& sectionPoints);
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
