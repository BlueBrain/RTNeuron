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

#ifndef RTNEURON_NEURONSKELETON_H
#define RTNEURON_NEURONSKELETON_H

#include "NeuronParts.h"

#include "render/Skeleton.h"

#include <brain/neuron/types.h>

#include <osg/PrimitiveSet>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
/** \brief Derived class that implements the logic to extract a Skeleton from a
  bbp::Neuron.
*/
class NeuronSkeleton : public Skeleton
{
    friend std::ostream& operator<<(std::ostream& out,
                                    const NeuronSkeleton& skeleton);

public:
    /*--- Public constructors/destructor */

    NeuronSkeleton(const brain::neuron::Morphology& morphology,
                   const NeuronParts parts = NEURON_PARTS_FULL);

    NeuronSkeleton(const NeuronSkeleton& skeleton);

    /*--- Public member functions ---*/

    /**
       Augments the skeleton portions of a morphology using its mesh
       triangle strip and extracts the strip intervals for each
       portion in the skeleton. The input strip is supposed to be
       sorted by section and almost sorted by relative position (Given
       to contiguous vertices, if they belong to the same section they
       must belong to neighbour skeleton portions). Given this strip,
       the different section portions of the skeleton are enlarged
       with the triangles from the strip to make sure than each
       portion is well defined bounding volume of a linear piece of
       the triangle strip. Degeneraed triangles are not considered
       since they are not rendered.
     */
    PortionRanges postProcess(const NeuronMesh& mesh);

    /**
       A convenient overload of the function above. It does basically the same
       but using the arrays directly but allows finer control.
       The main difference is that with GL_LINES primitives it is possible to
       indicate whether the primitive is sorted by section or not. It is still
       a requirement that the primitive indices are sorted by relative position
       of their vertices within a particular section.
     */
    PortionRanges postProcess(const osg::Vec3* vertices,
                              const uint16_t* sections, const float* positions,
                              const uint32_t* primitive,
                              const osg::PrimitiveSet::Mode mode,
                              const size_t length);

public:
    /*--- Protected member functions ---*/

    void _computeBranchOrders(const brain::neuron::Morphology& morphology);

    PortionRanges _postProcessLines(const osg::Vec3* vertices,
                                    const uint16_t* sections,
                                    const float* positions,
                                    const uint32_t* primitive,
                                    const size_t length);

    PortionRanges _postProcessTriangles(const osg::Vec3* vertices,
                                        const uint16_t* sections,
                                        const float* positions,
                                        const uint32_t* primitive,
                                        const size_t length);

    PortionRanges _postProcessStrip(const osg::Vec3* vertices,
                                    const uint16_t* sections,
                                    const float* positions,
                                    const uint32_t* primitive,
                                    const size_t length);

    unsigned int _postProcessTriangle(
        const osg::Vec3* vertices, const uint16_t* sections,
        const float* positions, const uint32_t* triangle,
        uint16_t& currentSection, unsigned int& skeletonIndex, osg::Vec3& axis);

    /* sections must be sorted by section ID. */
    void _extract(
        const brain::neuron::Soma& soma,
        const brain::neuron::Sections& sections,
        const unsigned int maxPortionsPerSection = MAX_CAPSULES_PER_SECTION);

    Portions _extractPortions(const brain::neuron::Sections& sections,
                              const unsigned int maxPortionsPerSection);

    void _addSomaCapsule(const brain::neuron::Soma& soma);

    void _addCapsule(const Portion& portion, const brion::Vector4fs& samples,
                     const brion::floats& relativeDistances, const size_t index,
                     const size_t portionIndex, const size_t SectionID,
                     const bool sectionStart, size_t& blockIndex,
                     size_t& accumulatedSections);
};

std::ostream& operator<<(std::ostream& out, const NeuronSkeleton& skeleton);
}
}
}
}
#endif
