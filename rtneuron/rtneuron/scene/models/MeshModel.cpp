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

#include "MeshModel.h"

#include "ConstructionData.h"

#include "data/NeuronMesh.h"
#include "render/SceneStyle.h"
#include "render/ShallowArray.h"
#include "scene/CircuitSceneAttributes.h"
#include "util/vec_to_vec.h"
#include "utils.h"

inline void shared_array_non_deleter(const void*)
{
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
DrawElementsPortions::IndexArray _unrollStrip(const uint32_t* strip,
                                              size_t length)
{
    std::vector<size_t> indices;
    indices.reserve(length * 3);
    for (size_t i = 0; i < length - 2; ++i)
    {
        if (strip[i] == strip[i + 1] || strip[i + 1] == strip[i + 2] ||
            strip[i] == strip[i + 2])
        {
            continue;
        }
        if (i % 2 == 0)
        {
            indices.push_back(strip[i]);
            indices.push_back(strip[i + 1]);
            indices.push_back(strip[i + 2]);
        }
        else
        {
            indices.push_back(strip[i]);
            indices.push_back(strip[i + 2]);
            indices.push_back(strip[i + 1]);
        }
    }
    assert(sizeof(GLuint) == sizeof(size_t));
    DrawElementsPortions::IndexArray outIndices(new GLuint[indices.size()]);
    memcpy(&outIndices[0], &indices[0], sizeof(size_t) * indices.size());
    return outIndices;
}

MeshModel::MeshModel(const NeuronParts parts, const ConstructionData& data)
{
    _useDefaultCull = parts == NEURON_PARTS_SOMA;

    switch (parts)
    {
    case NEURON_PARTS_SOMA:
        _createSomaMesh(data);
        break;
    case NEURON_PARTS_SOMA_DENDRITES:
        _createSomaDendriteMesh(data);
        break;
    case NEURON_PARTS_FULL:
        _createFullMesh(data);
        break;
    }

    osg::ref_ptr<osg::VertexBufferObject> vbo = new osg::VertexBufferObject();
    _vertices->setVertexBufferObject(vbo);
    _normals->setVertexBufferObject(vbo);
}

osg::StateSet* MeshModel::getModelStateSet(const bool subModel,
                                           const SceneStyle& style) const
{
    if (subModel)
    {
        AttributeMap extra;
        if (_clockWiseWinding)
            extra.set("cw_winding", true);
        return style.getStateSet(SceneStyle::NEURON_MESH, extra);
    }

    /* is this codepath called? */
    if (_clockWiseWinding)
    {
        osg::StateSet* stateSet = new osg::StateSet();
        stateSet->setAttributeAndModes(
            new osg::FrontFace(osg::FrontFace::CLOCKWISE));
        return stateSet;
    }
    return 0;
}

void MeshModel::clip(const Planes& planes)
{
    /* Annotating vertices that are visible and creating new primitives
       with the new vertex indices. */
    std::vector<bool> referenced(_length, false);
    util::clipPrimitive(_indices, _primitiveLength, _mode,
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
    osg::ref_ptr<osg::Vec3Array> normals(new osg::Vec3Array());
    normals->reserve(_length);
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
        normals->push_back(((osg::Vec3f*)_normals->getDataPointer())[i]);
        sections->push_back((*_sections)[i]);
        positions->push_back((*_positions)[i]);
        if (_colors)
            colors->push_back(((osg::Vec4f*)_colors->getDataPointer())[i]);
    }

    _vertices = vertices;
    _normals = normals;
    _colors = colors;
    _sections = sections;
    _positions = positions;

    osg::ref_ptr<osg::VertexBufferObject> vbo = new osg::VertexBufferObject();
    _vertices->setVertexBufferObject(vbo);
    _normals->setVertexBufferObject(vbo);
    if (_colors)
        _colors->setVertexBufferObject(vbo);

    /* After hard clipping the backup of the mesh data is not needed
       anymore. */
    _custodians._vertices.reset();
    _custodians._normals.reset();
    _custodians._indices.reset();
}

Skeleton::PortionRanges MeshModel::postProcess(NeuronSkeleton& skeleton)
{
    return skeleton.postProcess((osg::Vec3*)_vertices->getDataPointer(),
                                _sections->data(), _positions->data(),
                                (uint32_t*)_indices.get(), _mode,
                                _primitiveLength);
}

const osg::BoundingBox& MeshModel::getOrCreateBound(const Neuron&)
{
    if (!_bbox.valid())
    {
        osg::Vec3 min(std::numeric_limits<size_t>::max(),
                      std::numeric_limits<size_t>::max(),
                      std::numeric_limits<size_t>::max());
        osg::Vec3 max(-min);
        const osg::Vec3* vertices =
            static_cast<const osg::Vec3*>(_vertices->getDataPointer());
        for (size_t i = 0; i < _length; ++i)
        {
            for (int k = 0; k < 3; ++k)
            {
                max[k] = std::max(max[k], vertices[i][k]);
                min[k] = std::min(min[k], vertices[i][k]);
            }
        }
        _bbox.set(min, max);
    }
    return _bbox;
}

DetailedNeuronModel::Drawable* MeshModel::instantiate(
    const SkeletonPtr& skeleton, const CircuitSceneAttributes& sceneAttr)
{
    Drawable* geometry = new Drawable();
    if (_primitiveLength == 0)
        return geometry;

    /* Setting up vertices and normals */
    geometry->setVertexArray(_vertices);
    geometry->setNormalArray(_normals);
    geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    /* Creating the primitive to be drawn by this osg::Geometry */
    DrawElementsPortions* primitive =
        new DrawElementsPortions(_mode, _indices, _primitiveLength);
    /* Checking if there is a DrawElementsPortions holding a EBO for this
       neuron and taking the EBO from it. Creating a new EBO and selecting
       this primitive as the owner otherwise */

    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        /* This may be called concurrently from CircuitScene::addNeuronList */
        if (!_primitiveEBOOwner)
        {
            _primitiveEBOOwner = primitive;
            if (!_useDefaultCull)
                _ranges = postProcess(*skeleton);
        }
        else
        {
            primitive->setEBOOwner(_primitiveEBOOwner.get());
        }
    }

    if (!_useDefaultCull)
    {
/* Setting up the skeleton instance for this model and the
   cull callback that will perform skeletal based culling. */
#ifdef USE_CUDA
        if (sceneAttr.useCUDACulling)
            geometry->setCullCallback(Skeleton::createCullCallback(skeleton));
#else
        (void)sceneAttr;
#endif
        primitive->setSkeleton(skeleton);
        primitive->setPortionRanges(_ranges);
    }

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

#define IS_IN_CORE(i) \
    util::isInCore(morphology, sections, positions, (*indices)[i])
void MeshModel::_createSomaMesh(const ConstructionData& data)
{
    const auto& morphology = data.morphology;
    const auto& mesh = *data.mesh;

    /* Counting the number of polygons the core consists of an creating
       the translation table. */
    const unsigned int UNDEFINED = std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> translation(mesh.getVertexCount(), UNDEFINED);

    uint32_tsPtr indices;
    size_t step;

    /* Preferring the strip over the soup to maintain the consistency */
    if (data.sceneAttr.primitiveOptions.strips && mesh.getTriangleStrip())
    {
        step = 1;
        indices = mesh.getTriangleStrip();
    }
    else if (mesh.getTriangles())
    {
        step = 3;
        indices = mesh.getTriangles();
    }
    else
    {
        /* No geometry to process found */
        return;
    }

    const float* positions = mesh.getVertexRelativeDistances()->data();
    const uint16_t* sections = mesh.getVertexSections()->data();

    size_t numPolygons = 0;
    _length = 0;
    for (size_t i = 0; i < indices->size() - 2; i += step)
    {
        /* Skiping degenerated triangles when we are in a strip */
        if (step == 1 && ((*indices)[i + 2] == (*indices)[i] ||
                          (*indices)[i] == (*indices)[i + 1] ||
                          (*indices)[i + 1] == (*indices)[i + 2]))
        {
            continue;
        }

        /* A triangle is visible if any of its vertices is visible */
        if (IS_IN_CORE(i) || IS_IN_CORE(i + 1) || IS_IN_CORE(i + 2))
        {
            /* Storing tranlation of the indices */
            for (size_t j = 0; j < 3; ++j)
            {
                const unsigned int index = (*indices)[i + j];
                if (translation[index] == UNDEFINED)
                    translation[index] = _length++;
            }
            ++numPolygons;
        }
    }

    /* Creating the vertex, normal, sections and positions arrays */
    _vertices = new osg::Vec3Array();
    osg::Vec3Array& vertices = static_cast<osg::Vec3Array&>(*_vertices);
    vertices.resize(_length);
    vertices.setDataVariance(osg::Object::STATIC);
    _normals = new osg::Vec3Array();
    osg::Vec3Array& normals = static_cast<osg::Vec3Array&>(*_normals);
    normals.resize(_length);
    normals.setDataVariance(osg::Object::STATIC);
    _sections.reset(new uint16_ts(_length));
    _positions.reset(new floats(_length));
    const brain::Vector3f* sourceVertices = mesh.getVertices()->data();
    const brain::Vector3f* sourceNormals = mesh.getNormals()->data();
    for (unsigned int i = 0; i < mesh.getVertexCount(); ++i)
    {
        if (translation[i] != UNDEFINED)
        {
            const unsigned int index = translation[i];
            vertices[index] = vec_to_vec(sourceVertices[i]);
            normals[index] = vec_to_vec(sourceNormals[i]);
            (*_sections)[index] = sections[i];
            (*_positions)[index] = positions[i];
        }
    }

    /* Creating the primitive index array */
    _primitiveLength = numPolygons * 3;
    _indices.reset(new GLuint[_primitiveLength]);

    /* The loop is the same style than one the used for counting the polygons */
    bool reverseWinding = mesh.getVersion() == brion::MESH_VERSION_2;
    for (size_t i = 0, j = 0; i < (*indices).size() - 2; i += step)
    {
        if (step == 1)
        {
            reverseWinding = !reverseWinding;
            /* Skipping degenerate triangles */
            if ((*indices)[i + 2] == (*indices)[i] ||
                (*indices)[i] == (*indices)[i + 1] ||
                (*indices)[i + 1] == (*indices)[i + 2])
            {
                continue;
            }
        }
        if (IS_IN_CORE(i) || IS_IN_CORE(i + 1) || IS_IN_CORE(i + 2))
        {
            if (step == 3 || !reverseWinding)
            {
                _indices[j] = translation[(*indices)[i]];
                _indices[j + 1] = translation[(*indices)[i + 1]];
                _indices[j + 2] = translation[(*indices)[i + 2]];
            }
            else
            {
                _indices[j] = translation[(*indices)[i]];
                _indices[j + 2] = translation[(*indices)[i + 1]];
                _indices[j + 1] = translation[(*indices)[i + 2]];
            }
            j += 3;
        }
    }
    _clockWiseWinding = false;

    _mode = osg::PrimitiveSet::TRIANGLES;
}
#undef IS_IN_CORE

void MeshModel::_createSomaDendriteMesh(const ConstructionData& data)
{
    const auto& morphology = data.morphology;
    const auto& mesh = *data.mesh;

    /* Creating and clipping the primitive */
    _createFullMeshPrimitive(data);
    std::vector<bool> referenced(mesh.getVertexCount(), false);
    util::clipAxonFromPrimitive(_indices, _primitiveLength, _mode,
                                mesh.getVertexSections()->data(), morphology,
                                referenced);

    /* Computing final per vertex array sizes */
    _length = 0;
    for (size_t i = 0; i < referenced.size(); ++i)
    {
        if (referenced[i])
            ++_length;
    }

    /* Allocating data sets */
    osg::ref_ptr<osg::Vec3Array> vertices(new osg::Vec3Array());
    vertices->reserve(_length);
    osg::ref_ptr<osg::Vec3Array> normals(new osg::Vec3Array());
    normals->reserve(_length);
    uint16_tsPtr sections(new uint16_ts());
    sections->reserve(_length);
    floatsPtr positions(new floats());
    positions->reserve(_length);

    /* Copying filtered vertex data. */
    const auto& allVertices = *mesh.getVertices();
    const auto& allNormals = *mesh.getNormals();
    const auto& allSections = *mesh.getVertexSections();
    const auto& allPositions = *mesh.getVertexRelativeDistances();

    for (size_t i = 0; i < referenced.size(); ++i)
    {
        if (!referenced[i])
            continue;
        vertices->push_back(vec_to_vec(allVertices[i]));
        normals->push_back(vec_to_vec(allNormals[i]));
        sections->push_back(allSections[i]);
        positions->push_back(allPositions[i]);
    }

    _vertices = vertices;
    _normals = normals;
    _sections = sections;
    _positions = positions;
}

void MeshModel::_createFullMesh(const ConstructionData& data)
{
    const auto& mesh = *data.mesh;
    if (data.sceneAttr.generateMeshes)
    {
        _custodians._vertices = mesh.getVertices();
        _custodians._normals = mesh.getNormals();
    }
    const auto& vertices = *mesh.getVertices();
    const auto& normals = *mesh.getNormals();
    const auto count = mesh.getVertexCount();
    _vertices = new Vec3ShallowArray((osg::Vec3*)vertices.data(), count);
    _vertices->setDataVariance(osg::Object::STATIC);
    _normals = new Vec3ShallowArray((osg::Vec3*)normals.data(), count);
    _normals->setDataVariance(osg::Object::STATIC);
    _sections = mesh.getVertexSections();
    _positions = mesh.getVertexRelativeDistances();
    _length = mesh.getVertexCount();

    _createFullMeshPrimitive(data);
}

void MeshModel::_createFullMeshPrimitive(const ConstructionData& data)
{
    const auto& mesh = *data.mesh;
    const auto& sceneAttr = data.sceneAttr;

    auto strip = mesh.getTriangleStrip();
    if (!sceneAttr.primitiveOptions.strips || !strip)
    {
        auto triangles = mesh.getTriangles();
        if (!triangles || triangles->empty())
        {
            LBTHROW(
                std::runtime_error("Creating neuron mesh model: triangle"
                                   " indices not available"));
        }
        if (data.sceneAttr.generateMeshes)
            _custodians._indices = triangles;

        _indices =
            DrawElementsPortions::IndexArray((GLuint*)(triangles->data()),
                                             shared_array_non_deleter);
        _primitiveLength = triangles->size();
        _mode = osg::PrimitiveSet::TRIANGLES;
        if (mesh.getVersion() == brion::MESH_VERSION_1)
            _useDefaultCull = true;
        _clockWiseWinding = false;
    }
    else
    {
        if (strip->empty())
        {
            LBTHROW(
                std::runtime_error("Creating neuron mesh model: triangle strip"
                                   " indices not available"));
        }
        if (sceneAttr.primitiveOptions.unrollStrips)
        {
            _indices = _unrollStrip(strip->data(), strip->size());
            _primitiveLength = strip->size() * 3;
            _mode = osg::PrimitiveSet::TRIANGLES;
        }
        else
        {
            if (data.sceneAttr.generateMeshes)
                _custodians._indices = strip;

            _indices =
                DrawElementsPortions::IndexArray((GLuint*)strip->data(),
                                                 shared_array_non_deleter);
            _primitiveLength = strip->size();
            _mode = osg::PrimitiveSet::TRIANGLE_STRIP;
        }
        _clockWiseWinding = mesh.getVersion() == brion::MESH_VERSION_1;
    }
}
}
}
}
}
