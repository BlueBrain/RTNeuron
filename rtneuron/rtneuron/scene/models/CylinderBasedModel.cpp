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

#include "CylinderBasedModel.h"

#include "ConstructionData.h"

#include "config/constants.h"
#include "render/SceneStyle.h"
#include "scene/CircuitSceneAttributes.h"
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
inline osg::Vec3 _center(const brain::Vector4f& sample)
{
    return osg::Vec3(sample[0], sample[1], sample[2]);
}
inline float _radius(const brain::Vector4f& sample)
{
    return sample[3] * 0.5;
}

bool _skipPoint(const osg::Vec3& point, const osg::Vec3& previousPoint,
                const osg::Vec3& previousTangent,
                const float maxSquaredDistanceToMorphology)
{
    if (maxSquaredDistanceToMorphology == 0)
        return false;

    const osg::Vec3 p = point - previousPoint;
    const float p_t = previousTangent * p;
    const float p_p = p * p;
    return p_p - p_t * p_t < maxSquaredDistanceToMorphology;
}
}

CylinderBasedModel::CylinderBasedModel(
    const NeuronParts parts, const ConstructionData& data,
    const float maxSquaredDistanceToMorphology, const bool supportCLODHint)
{
    if (parts == NEURON_PARTS_SOMA)
        throw std::runtime_error(
            "Pseudo cylinder neuron models cannot be soma only");
    const bool noAxon = parts == NEURON_PARTS_SOMA_DENDRITES;

    _create(data, maxSquaredDistanceToMorphology, noAxon);

    _supportCLOD = (supportCLODHint && data.sceneAttr.neuronLODs &&
                    (*data.sceneAttr.neuronLODs)("clod", false));

    osg::ref_ptr<osg::VertexBufferObject> vbo = new osg::VertexBufferObject();
    _vertices->setVertexBufferObject(vbo);
    _tangentsAndThickness->setVertexBufferObject(vbo);
}

void CylinderBasedModel::clip(const std::vector<osg::Vec4d>& planes)
{
    /* Annotating vertices that are visible and creating new primitives
       with the new vertex indices. */
    std::vector<bool> referenced(_length, false);
    util::clipPrimitive(_indices, _primitiveLength, osg::PrimitiveSet::LINES,
                        (osg::Vec3f*)_vertices->getDataPointer(), planes,
                        referenced);

    /* Computing final size of per vertex arrays and annotating the indices
       of the vertices to remove */
    _length = 0;
    for (size_t i = 0; i < referenced.size(); ++i)
    {
        if (referenced[i])
            ++_length;
    }

    /* Creating new per vertex arrays */
    osg::ref_ptr<osg::Vec3Array> vertices(new osg::Vec3Array());
    vertices->reserve(_length);
    osg::ref_ptr<osg::Vec4Array> tangentsAndThickness(new osg::Vec4Array());
    tangentsAndThickness->reserve(_length);
    osg::ref_ptr<osg::Vec4Array> colors;
    if (_colors)
    {
        colors = new osg::Vec4Array();
        colors->reserve(_length);
    }
    uint16_tsPtr sections(new uint16_ts());
    sections->reserve(_length);
    floatsPtr positions(new floats());
    positions->reserve(_length);

    for (size_t i = 0; i < referenced.size(); ++i)
    {
        if (!referenced[i])
            continue;
        vertices->push_back(((osg::Vec3f*)_vertices->getDataPointer())[i]);
        tangentsAndThickness->push_back(
            ((osg::Vec4f*)_tangentsAndThickness->getDataPointer())[i]);
        sections->push_back((*_sections)[i]);
        positions->push_back((*_positions)[i]);
        if (_colors)
            colors->push_back(((osg::Vec4f*)_colors->getDataPointer())[i]);
    }
    _vertices = vertices;
    _tangentsAndThickness = tangentsAndThickness;
    _colors = colors;
    _sections = sections;
    _positions = positions;
}

Skeleton::PortionRanges CylinderBasedModel::postProcess(
    NeuronSkeleton& skeleton)
{
    return skeleton.postProcess((osg::Vec3*)_vertices->getDataPointer(),
                                _sections->data(), _positions->data(),
                                (uint32_t*)_indices.get(),
                                osg::PrimitiveSet::LINES, _primitiveLength);
}

osg::StateSet* CylinderBasedModel::getModelStateSet(
    const bool subModel, const SceneStyle& style) const
{
    if (!subModel)
        return 0;

    AttributeMap extra;
    if (_supportCLOD)
        extra.set("clod", _supportCLOD);
    return style.getStateSet(SceneStyle::NEURON_PSEUDO_CYLINDERS, extra);
}

DetailedNeuronModel::Drawable* CylinderBasedModel::instantiate(
    const SkeletonPtr& skeleton, const CircuitSceneAttributes& sceneAttr)
{
    Drawable* geometry = new Drawable();
    if (_primitiveLength == 0)
        return geometry;

    /* Setting up the geometry object */
    geometry->setVertexArray(_vertices);
    geometry->setVertexAttribArray(TANGENT_AND_THICKNESS_ATTRIB_NUM,
                                   _tangentsAndThickness);
    geometry->setVertexAttribBinding(TANGENT_AND_THICKNESS_ATTRIB_NUM,
                                     osg::Geometry::BIND_PER_VERTEX);

    /* Creating the primitive */
    DrawElementsPortions* primitive =
        new DrawElementsPortions(osg::PrimitiveSet::LINES, _indices,
                                 _primitiveLength);
    /* Checking if there is a DrawElementsPortions holding a EBO for this
       neuron and taking the EBO from it. Creating a new EBO and selecting this
       primitive as the owner otherwise */
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        /* This may be called concurrently from CircuitScene::addNeuronList */
        if (!_primitiveEBOOwner)
        {
            _primitiveEBOOwner = primitive;
            _ranges = postProcess(*skeleton);
        }
        else
        {
            primitive->setEBOOwner(_primitiveEBOOwner);
        }
    }

#ifdef USE_CUDA
    /* Setting up the skeleton instance for this model and the
       cull callback that will perform skeletal based culling. */
    if (sceneAttr.useCUDACulling)
        geometry->setCullCallback(Skeleton::createCullCallback(skeleton));
#else
    (void)sceneAttr;
#endif
    primitive->setSkeleton(skeleton);
    primitive->setPortionRanges(_ranges);

    /* No other primitive set should be added to these geometry objects
       unless we want to screw the EBO that will be created and shared
       internally. */
    geometry->addPrimitiveSet(primitive);

    geometry->setUseDisplayList(false);
    {
        /* Protecting concurrent access to the VBO during
           CircuitScene::addNeuronList */
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        geometry->setUseVertexBufferObjects(true);
    }

    return geometry;
}

void CylinderBasedModel::_create(const ConstructionData& data,
                                 const float maxSquaredDistanceToMorphology,
                                 const bool noAxon)
{
    const auto& morphology = data.morphology;

    /** Special points to connect first order sections if required */
    util::SectionStartPointsMap sectionStarts;
    if (data.sceneAttr.connectFirstOrderBranches &&
        maxSquaredDistanceToMorphology == 0 && data.mesh)
    {
        sectionStarts =
            util::extractCellCoreSectionStartPoints(morphology, *data.mesh);
    }

    using S = brain::neuron::SectionType;
    std::vector<S> types{S::dendrite, S::apicalDendrite, S::axon};
    if (noAxon)
        types.pop_back();
    auto sections = morphology.getSections(types);
    util::sortSections(sections);

    const size_t numPoints = util::numberOfPoints(sections);
    _allocate(numPoints);
    std::vector<GLuint> indices;
    indices.reserve((numPoints - sections.size()) * 2);

    GLuint primitiveIndex = 0;
    size_t pointCount = 0;
    osg::Vec3Array& vertices = static_cast<osg::Vec3Array&>(*_vertices);

    for (const auto& section : sections)
    {
        _createSection(section, data, maxSquaredDistanceToMorphology,
                       sectionStarts);

        for (size_t i = 0; i < vertices.size() - pointCount - 1;
             ++i, ++primitiveIndex)
        {
            indices.push_back(primitiveIndex);
            indices.push_back(primitiveIndex + 1);
        }
        pointCount = vertices.size();
        ++primitiveIndex;
    }

    _vertices->trim();
    _tangentsAndThickness->trim();
    _length = static_cast<osg::Vec3Array&>(*_vertices).size();
    _primitiveLength = indices.size();

    _indices.reset(new GLuint[_primitiveLength]);
    memcpy(&_indices[0], &indices[0], sizeof(GLuint) * _primitiveLength);
}

void CylinderBasedModel::_allocate(const size_t numPoints)
{
    _vertices = new osg::Vec3Array();
    _vertices->setDataVariance(osg::Object::STATIC);
    _tangentsAndThickness = new osg::Vec4Array();
    _tangentsAndThickness->setDataVariance(osg::Object::STATIC);
    _sections.reset(new uint16_ts());
    _positions.reset(new floats());

    static_cast<osg::Vec3Array&>(*_vertices).reserve(numPoints);
    _tangentsAndThickness->reserve(numPoints);
    _sections->reserve(numPoints);
    _positions->reserve(numPoints);
}

void CylinderBasedModel::_createSection(
    const brain::neuron::Section& section, const ConstructionData& data,
    const float maxSquaredDistanceToMorphology,
    util::SectionStartPointsMap& sectionStarts)
{
    auto samples = section.getSamples();
    auto relativeDistances =
        util::computeRelativeDistances(samples, section.getLength());
    util::removeZeroLengthSegments(samples, relativeDistances);

    auto& vertices = static_cast<osg::Vec3Array&>(*_vertices);
    size_t index =
        _startSection(section, samples, relativeDistances, data, sectionStarts);

    assert(index > 0);
    auto previousPoint = vertices.back();
    const auto t = _tangentsAndThickness->back();
    osg::Vec3 previousTangent{t[0], t[1], t[2]};

    for (; index != samples.size() - 1; ++index)
    {
        /* Getting the information for the first point of the segment. */
        const auto& sample = samples[index];
        const auto point = _center(sample);
        const auto radius = _radius(sample);

        /* Skipping this point if it's closer from the previous tangent
           vector more than the user given distance */
        if (_skipPoint(point, previousPoint, previousTangent,
                       maxSquaredDistanceToMorphology))
            continue;

        /* Computing tangent as the central difference vector. */
        const auto next = _center(samples[index + 1]);
        auto tangent = next - previousPoint;
        tangent.normalize();

        /* Adding vertex, tangent and other info to geometry arrays. */
        vertices.push_back(point);
        _sections->push_back(section.getID());
        _positions->push_back(relativeDistances[index]);
        _tangentsAndThickness->push_back(osg::Vec4{tangent, radius});

        previousTangent = tangent;
        previousPoint = point;
    }

    const auto& sample = *samples.rbegin();
    const auto point = _center(sample);
    const auto radius = _radius(sample);

    const auto child = model::util::mainBranchChildSection(section);
    assert(child.getID() == section.getID() || child.getNumSamples() > 1);
    const auto nextPoint =
        child.getID() == section.getID() ? point : _center(child[1]);
    auto tangent = (nextPoint - previousPoint);
    tangent.normalize();

    vertices.push_back(point);
    _sections->push_back(section.getID());
    _positions->push_back(1.f);
    _tangentsAndThickness->push_back(osg::Vec4(tangent, radius));
}

size_t CylinderBasedModel::_startSection(
    const brain::neuron::Section& section, const brain::Vector4fs& samples,
    const brain::floats& relativeDistances, const ConstructionData& data,
    util::SectionStartPointsMap& sectionStarts)
{
    const bool connectToSoma = data.sceneAttr.connectFirstOrderBranches;

    size_t index = 1;
    auto point = _center(samples[0]);
    auto radius = _radius(samples[0]);
    float position = 0;

    /* Previous point for tangent calculation */
    osg::Vec3 previousPoint;

    if (!section.hasParent())
    {
        if (connectToSoma && !sectionStarts.empty())
        {
            const auto& sectionStart = sectionStarts[section.getID()];
            index = sectionStart.findSampleAfterSectionStart(
                samples, relativeDistances, section.getLength(), radius);
            point = sectionStart.point;
            position = sectionStart.position;
            previousPoint = point;
            /* Shifting the point slightly backwards because otherwise there
               is a visible gap between the mesh and the pseudocylinder. */
            auto u = _center(samples[index]) - point;
            u.normalize();
            point -= u * (radius * 0.5f);
        }
        else if (connectToSoma)
        {
            const auto& soma = data.morphology.getSoma();
            const auto somaCenter = _center(soma.getCentroid());
            const float somaRadius = soma.getMeanRadius();

            radius *= 1.1f;
            /* The following computes a normalized radius from soma center. */
            (point = point - somaCenter).normalize();
            /* Now we place the section start using the maximal soma radius. */
            point = somaCenter + point * somaRadius * 0.80;
            /* Previous point for tangent calculation */
            previousPoint = somaCenter;
        }
        else
            previousPoint = point;
    }
    else
    {
        const auto& parent = section.getParent();
        assert(parent.getNumSamples() > 1);
        const auto& bestChild = util::mainBranchChildSection(parent);

        if (bestChild.getID() == section.getID())
        {
            previousPoint = _center(parent[-2]);
            radius = _radius(parent[-1]);
        }
        else
        {
            previousPoint = point;
        }
    }

    /* Computing tangent as the central difference vector. */
    const auto next = _center(samples[index]);
    auto tangent = next - previousPoint;
    tangent.normalize();

    /* Adding vertex, tangent and other info to geometry arrays. */
    osg::Vec3Array& vertices = static_cast<osg::Vec3Array&>(*_vertices);
    vertices.push_back(point);
    _sections->push_back(section.getID());
    _positions->push_back(position);
    _tangentsAndThickness->push_back(osg::Vec4(tangent, radius));

    return index;
}
}
}
}
}
