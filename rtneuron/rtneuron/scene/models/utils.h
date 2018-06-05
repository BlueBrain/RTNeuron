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

#ifndef RTNEURON_MODELUITLS_H
#define RTNEURON_MODELUITLS_H

#include <brain/neuron/types.h>
#include <brain/types.h>

#include <osg/PrimitiveSet>
#include <osg/Vec3>

#include <boost/shared_array.hpp>

#include <map>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class NeuronMesh;

namespace model
{
/**
 * Shared functionality which is needed for the creation of the different models
 */
namespace util
{
struct SectionStartPoint
{
    /** Return the last sample whose relative position along the polyline
        is <= position.
        Return the adjusted radius for the virtual section start by
        interpolating the radius and that sample and the radius at the
        following sample according to the section start position.
    */
    size_t findSampleAfterSectionStart(const brain::Vector4fs& samples,
                                       const brain::floats& relativeDistances,
                                       float sectionLength,
                                       float& correctedInitialRadius) const;

    osg::Vec3 point;
    float position;
};

typedef std::vector<bool> bools;
typedef boost::shared_array<unsigned int> Indices;
typedef std::vector<osg::Vec4d> Planes;
typedef std::map<uint16_t, SectionStartPoint> SectionStartPointsMap;

/** Sort sections by ID */
void sortSections(brain::neuron::Sections& sections);

size_t numberOfPoints(const brain::neuron::Sections& sections);

brain::floats computeRelativeDistances(const brain::Vector4fs& samples,
                                       float totalLength);

void removeZeroLengthSegments(brain::Vector4fs& samples,
                              brain::floats& relativeDistances);

bool isInCore(const brain::neuron::Morphology& morphology,
              const uint16_t* sections, const float* positions,
              const uint32_t index);

SectionStartPointsMap extractCellCoreSectionStartPoints(
    const brain::neuron::Morphology& morphology, const NeuronMesh& mesh);

/** Return the main child section that can be considered the main branch
    continuation of a given one.
    When the section is terminal return it instead.
*/
brain::neuron::Section mainBranchChildSection(
    const brain::neuron::Section& section);

void clipPrimitive(Indices& indices, size_t& length,
                   const osg::PrimitiveSet::Mode mode,
                   const osg::Vec3* vertices, const Planes& planes,
                   bools& referenced);

void clipAxonFromPrimitive(Indices& indices, size_t& length,
                           const osg::PrimitiveSet::Mode mode,
                           const uint16_t* sections,
                           const brain::neuron::Morphology& morphogy,
                           bools& referenced);
}
}
}
}
}

#endif
