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

#include "SkeletonModel.h"

#include "../../AttributeMap.h"
#include "data/Neuron.h"
#include "util/vec_to_vec.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
namespace
{
osg::Array* _createAxonDendriteColorArray(
    const brain::neuron::Morphology& morph, const uint16_t* sections,
    const osg::Vec4& dendriteColor, const osg::Vec4& axonColor,
    const size_t length)
{
    osg::Vec4Array* colors = new osg::Vec4Array();
    colors->reserve(length);
    colors->setDataVariance(osg::Object::STATIC);

    for (size_t i = 0; i < length; ++i)
    {
        const uint16_t sectionID = sections[i];
        if (sectionID == 0)
        {
            colors->push_back(dendriteColor);
            continue;
        }
        const brain::neuron::Section& section = morph.getSection(sectionID);
        const osg::Vec4 color =
            section.getType() == brain::neuron::SectionType::axon
                ? axonColor
                : dendriteColor;
        colors->push_back(color);
    }
    return colors;
}

osg::Array* _perVertexWidths(const brain::neuron::Morphology& morph,
                             const osg::Vec3* vertices,
                             const boost::uint16_t* sections,
                             const float* positions, const size_t size)
{
    osg::FloatArray* widths = new osg::FloatArray();
    widths->resize(size);
    widths->setDataVariance(osg::Object::STATIC);

    const auto& soma = morph.getSoma();

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        const boost::uint16_t sectionID = sections[i];
        const float position = std::max(0.0f, std::min(1.0f, positions[i]));
        float width;
        /* Setion 0 is assumed to be the soma. */
        if (sectionID == 0)
        {
            const float trunkWidth = 4;
            const float diameter = soma.getMeanRadius() * 2;
            const float distance =
                (vec_to_vec(vertices[i]) - soma.getCentroid()).length() -
                diameter * 0.5;
            if (distance < 0)
                width = diameter;
            else if (distance > 2)
                width = trunkWidth;
            else
                width = ((1.f - distance * .5f) * diameter +
                         distance * .5f * trunkWidth);
        }
        else
        {
            const auto& section = morph.getSection(sectionID);
            width = section.getSamples({position})[0][3];
        }

        (*widths)[i] = width;
    }
    return widths;
}

osg::Array* _perVertexDistancesToSoma(const brain::neuron::Morphology& morph,
                                      const boost::uint16_t* sections,
                                      const float* positions, size_t size)
{
    osg::FloatArray* values = new osg::FloatArray();
    values->reserve(size);
    values->setDataVariance(osg::Object::STATIC);

    for (size_t i = 0; i < size; ++i)
    {
        const boost::uint16_t sectionID = sections[i];
        const float position = std::max(0.0f, std::min(1.0f, positions[i]));
        float distance;
        /* Setion 0 is assumed to be the soma. */
        if (sectionID == 0)
        {
            distance = 0;
        }
        else
        {
            const auto& section = morph.getSection(sectionID);
            const auto length = section.getLength();
            const auto distanceToSoma = section.getDistanceToSoma();
            distance = distanceToSoma + length * position;
        }

        values->push_back(distance);
    }
    return values;
}

/**
   Creates the array of per vertex attributes required for a coloring scheme.

   The output array will be a osg::FloatArray for by-width and
   by-distance-to-soma modes; in this cases the actual color are given
   by a separate colormap indexed by the attribute array. For the rest of the
   coloring schemes this functions returns an osg::Vec4Array with the colors.
*/
osg::ref_ptr<osg::Array> _createPerVertexColoringArray(
    const NeuronColoring& coloring, const brain::neuron::Morphology& morph,
    const osg::Vec3* vertices, const uint16_t* sections, const float* positions,
    size_t length)
{
    switch (coloring.getScheme())
    {
    case ColorScheme::BY_WIDTH:
    {
        return _perVertexWidths(morph, vertices, sections, positions, length);
    }
    case ColorScheme::BY_DISTANCE_TO_SOMA:
    {
        return _perVertexDistancesToSoma(morph, sections, positions, length);
    }
    case ColorScheme::BY_BRANCH_TYPE:
    {
        osg::Vec4 dendrite = coloring.getPrimaryColor();
        osg::Vec4 axon = coloring.getSecondaryColor();
        return _createAxonDendriteColorArray(morph, sections, dendrite, axon,
                                             length);
    }
    default:
        LBTHROW(std::runtime_error("Unkown color map type"));
        abort();
    }
}
}

SkeletonModel::~SkeletonModel()
{
}

osg::ref_ptr<osg::Array> SkeletonModel::getOrCreateColorArray(
    const NeuronColoring& coloring, const brain::neuron::Morphology& morphology)
{
    /* When shared models are used, this function may be called concurrently
       from CircuitScene::addNeuronList. The most reasonable solution seems
       to be to lock the the whole function call. This is not a contention
       problem in circuits with many unique morphologies. */
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    /* Returning the cached array if valid. */
    if (_coloring.areVertexAttributesCompatible(coloring) && _colors &&
        _colors->getNumElements() == _length)
    {
        return _colors;
    }

    assert(_vertices);
    if (coloring.getScheme() == ColorScheme::SOLID)
    {
        osg::ref_ptr<osg::Vec4Array> array(new osg::Vec4Array(1));
        array->setDataVariance(osg::Object::STATIC);
        array->setVertexBufferObject(_vertices->getVertexBufferObject());
        return array;
    }

    _colors =
        _createPerVertexColoringArray(coloring, morphology,
                                      (osg::Vec3*)_vertices->getDataPointer(),
                                      _sections->data(), _positions->data(),
                                      _length);
    _coloring = coloring;
    _colors->setDataVariance(osg::Object::STATIC);
    /* Using a separate VBO for this array because even with shared
       morphologies, different cells may be updated to use different
       parameters (actually, this only happens for by-branch). */
    _colors->setVertexBufferObject(_vertices->getVertexBufferObject());
    return _colors;
}

const osg::BoundingBox& SkeletonModel::getOrCreateBound(const Neuron& neuron)
{
    if (!_bbox.valid())
    {
        brain::Vector3f min, max;
        neuron.getMorphology()->getBoundingBox(min, max);
        _bbox.set(vec_to_vec(min), vec_to_vec(max));
    }
    return _bbox;
}
}
}
}
}
