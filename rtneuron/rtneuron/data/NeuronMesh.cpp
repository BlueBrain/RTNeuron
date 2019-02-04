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

#include "NeuronMesh.h"
#include "data/Neuron.h"

#include <brion/mesh.h>

#if RTNEURON_USE_NEUMESH
#include <BBP/Morphology.h>

#include <brain/neuron/morphology.h>
#include <neumesh/MeshConstruction.h>
#include <neumesh/Morphology.h>
#endif

#include <brain/circuit.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <vmmlib/vector.hpp>

#include <lunchbox/debug.h>

namespace bbp
{
#if RTNEURON_USE_NEUMESH
/* This is a gross hack to support on-the-fly mesh creation without
   modifying BBPSDK */
namespace detail
{
class MorphologyReader
{
public:
    static std::shared_ptr<bbp::Morphology> _createLegacyMorphology(
        const brain::neuron::Morphology& morphology);
};

std::shared_ptr<bbp::Morphology> MorphologyReader::_createLegacyMorphology(
    const brain::neuron::Morphology& morphology)
{
    /* Code taken from BBPSDK morphology reader. */
    const auto points = morphology.getPoints();
    const auto sections = morphology.getSections();
    const auto types = morphology.getSectionTypes();

    auto out = std::make_shared<bbp::Morphology>();

    /** First pass creates all sections unconnected */
    for (size_t sectionID = 0; sectionID < sections.size(); ++sectionID)
    {
        const int32_t startPoint = sections[sectionID][0];
        const size_t numberOfPoints =
            sectionID < sections.size() - 1
                ? sections[sectionID + 1][0] - startPoint
                : points.size() - startPoint;

        size_t numberOfSegments = numberOfPoints;
        /** A segment needs at least two points */
        if (numberOfSegments < 2)
            numberOfSegments = 0;

        const SectionType sectionType = (SectionType)types[sectionID];
        bbp::Section* newSection;
        if (sectionType == SECTION_SOMA)
        {
            LBASSERT(sectionID == 0);
            newSection = new Soma;
            /** Copy soma surface points */

            Soma* newSoma = static_cast<Soma*>(newSection);
            const auto& first = points[startPoint];
            if (numberOfPoints)
                newSoma->surface_points().insert(first.get_sub_vector<3, 0>());

            if (numberOfPoints == 1)
            {
                newSoma->single_point_radius(first[3]);
            }
            else
            {
                for (size_t pointID = 1; pointID < numberOfPoints; ++pointID)
                {
                    const auto& point =
                        points[startPoint + pointID].get_sub_vector<3, 0>();
                    newSoma->surface_points().insert(point);
                }
            }
        }
        else
        {
            newSection = new bbp::Section(sectionType, numberOfSegments);
            /** Copy segments */
            for (size_t crossSectionID = 0; crossSectionID < numberOfPoints;
                 ++crossSectionID)
            {
                const Vector4f& point = points[startPoint + crossSectionID];
                newSection->grow(
                    Cross_Section(point.get_sub_vector<3, 0>(), point.w()));
            }
        }
        newSection->morphology(out.get());
        newSection->register_in_morphology(sectionID);
        // cppcheck-suppress memleak newSection
    }

    /** Second pass connects all sections to their parent */
    for (Sections::iterator i = out->sections().begin();
         i != out->sections().end(); ++i)
    {
        const int32_t parentID = sections[i->id()][1];
        if (uint16_t(parentID) != LB_UNDEFINED_UINT16)
            i->parent(out->section(parentID));
    }
    return out;
}
}
#endif

namespace rtneuron
{
namespace core
{
namespace
{
const char* _meshSubPath = "high/TXT";
NeuronMeshPtr _createMesh(const std::string& path)
{
    return std::make_shared<NeuronMesh>(path);
}
}

NeuronMesh::NeuronMesh(const std::string& path)
{
    const brion::Mesh mesh(path);
    _vertices = mesh.readVertices();
    _normals = mesh.readNormals();
    _triangleStrip = mesh.readTriStrip();
    if (!_triangleStrip || _triangleStrip->empty())
    {
        _triangleStrip.reset();
        _triangles = mesh.readTriangles();
    }
    _sections = mesh.readVertexSections();
    _relativeDistances = mesh.readVertexDistances();
    _version = mesh.getVersion();
    if (!_normals || _normals->empty())
    {
        switch (_computeNormals())
        {
        case 1:
            LBWARN << "Cannot calculate normals for mesh " << path
                   << "; no triangles or triangle strips found" << std::endl;
            break;
        case 2:
            LBWARN << "Found degenerated triangle(s) in mesh " << path
                   << std::endl;
            break;
        default:;
        }
    }
}

#if RTNEURON_USE_NEUMESH
NeuronMesh::NeuronMesh(const brain::neuron::Morphology& morphology)
{
    const auto oldMorphology =
        detail::MorphologyReader::_createLegacyMorphology(morphology);
    neumesh::Morphology neuMeshMorhology(*oldMorphology);
    neumesh::MeshConstruction constructor(neuMeshMorhology);
    constructor.setSmoothing(true);
    constructor.setSubdivide(true);
    constructor.setTesselationFactor(1.0);
    const auto mesh = constructor.generateSurfaceMesh();
    const auto legacyMesh =
        constructor.convertToBBPMesh(*mesh, *oldMorphology, true, true);
    _vertices = legacyMesh->getVertices();
    _normals = legacyMesh->getNormals();
    _triangleStrip = legacyMesh->getTriangleStrip();
    _triangles = legacyMesh->getTriangles();
    _sections = legacyMesh->getVertexSections();
    _relativeDistances = legacyMesh->getVertexRelativeDistances();
    _version = legacyMesh->getVersion();
}
#endif

NeuronMeshes NeuronMesh::load(const Strings& labels, const std::string& prefix,
                              MeshCache& cache)
{
    boost::filesystem::path prefixPath{prefix};
    if (boost::filesystem::is_directory(prefixPath / _meshSubPath))
        prefixPath /= _meshSubPath;
    NeuronMeshes meshes;
    for (const auto& label : labels)
    {
        meshes.push_back(
            cache.getOrCreate(label, _createMesh,
                              (prefixPath / (label + ".bin")).string()));
    }
    return meshes;
}

NeuronMeshPtr NeuronMesh::load(const std::string& name,
                               const std::string& prefix)
{
    /* Not optimal for loading a set of meshes from a collection of neurons. */
    boost::filesystem::path path{prefix};
    if (boost::filesystem::is_directory(path / _meshSubPath))
        path /= _meshSubPath;
    path /= name + ".bin";
    return std::make_shared<NeuronMesh>(path.string());
}

int NeuronMesh::_computeNormals()
{
    using namespace brion;
    using namespace vmml;
    const Vector3fs& vertices = *_vertices;

    /* Allocating and initializing array */
    Vector3fsPtr normals(new Vector3fs(vertices.size()));

    if (_triangles)
    {
        /* Computing normals from a triangle soup */
        const uint32_ts& triangles = *_triangles;
        assert(triangles.size() % 3 == 0);
        /* Not parallelized because this function is called from an outer
           OpenMP loop and spawning more threads here causes resource
           contention (cache?) */
        for (size_t i = 0; i < triangles.size(); i += 3)
        {
            assert(triangles[i] < vertices.size() &&
                   triangles[i + 1] < vertices.size() &&
                   triangles[i + 2] < vertices.size());
            const Vector3f& u = vertices[triangles[i]];
            const Vector3f& v = vertices[triangles[i + 1]];
            const Vector3f& w = vertices[triangles[i + 2]];
            const Vector3f vu = v - u;
            const Vector3f wu = w - u;
            const Vector3f faceNormal = vmml::cross(vu, wu);
            (*normals)[triangles[i]] += faceNormal;
            (*normals)[triangles[i + 1]] += faceNormal;
            (*normals)[triangles[i + 2]] += faceNormal;
        }
    }
    else if (_triangleStrip)
    {
        /* Computing normals from a triangle strip */
        const uint32_ts& triStrip = *_triangleStrip;
        bool positive = _version == brion::MESH_VERSION_2;
        for (size_t i = 0; i < triStrip.size() - 2; ++i, positive = !positive)
        {
            if (triStrip[i] == triStrip[i + 1] ||
                triStrip[i] == triStrip[i + 2] ||
                triStrip[i + 1] == triStrip[i + 2])
            {
                /* Skipping degenerated triangle */
                continue;
            }
            assert(triStrip[i] < vertices.size() &&
                   triStrip[i + 1] < vertices.size() &&
                   triStrip[i + 2] < vertices.size());
            const Vector3f& u = vertices[triStrip[i]];
            const Vector3f& v = positive ? vertices[triStrip[i + 1]]
                                         : vertices[triStrip[i + 2]];
            const Vector3f& w = positive ? vertices[triStrip[i + 2]]
                                         : vertices[triStrip[i + 1]];
            Vector3f faceNormal = v - u;
            faceNormal = faceNormal.cross(w - u);
            (*normals)[triStrip[i]] += faceNormal;
            (*normals)[triStrip[i + 1]] += faceNormal;
            (*normals)[triStrip[i + 2]] += faceNormal;
        }
    }
    else
    {
        return 1;
    }

    bool degenerate = false;
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        if ((*normals)[i].length() != 0)
            (*normals)[i].normalize();
        else
            degenerate = true;
    }
    _normals = normals;
    return degenerate ? 2 : 0;
}
}
}
}
