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

#ifndef RTNEURON_NEURONMESH_H
#define RTNEURON_NEURONMESH_H

#include "coreTypes.h"
#include "util/ObjectCache.h"

#include <brain/neuron/types.h>
#include <brion/types.h>

#include <boost/filesystem/path.hpp>

namespace brain
{
class Circuit;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
using MeshCache = ObjectCache<NeuronMesh, std::string>;

class NeuronMesh
{
public:
    /*--- Public constructors/destructor ---*/

    explicit NeuronMesh(const std::string& path);

#if RTNEURON_USE_NEUMESH
    /** Create a mesh from the given morphology. */
    explicit NeuronMesh(const brain::neuron::Morphology& morphology);
#endif

    /*--- Public member functions ---*/

    static NeuronMeshes load(const Strings& labels, const std::string& prefix,
                             MeshCache& cache);
    static NeuronMeshPtr load(const std::string& name,
                              const std::string& prefix);

    size_t getVertexCount() const { return _vertices ? _vertices->size() : 0; }
    size_t getTriangleCount() const
    {
        return _triangles ? _triangles->size() / 3 : 0;
    }

    size_t getTriangleStripLength() const
    {
        return _triangleStrip ? _triangleStrip->size() : 0;
    }

    /**
        Get the mesh version
        brion::MESH_VERSION_1 meshes have negative orientation in the first
        triangle strip triangle, clockwise winding for the triangle soups.
    */
    brion::MeshVersion getVersion() const { return _version; }
    brion::Vector3fsPtr getVertices() const { return _vertices; }
    brion::Vector3fsPtr getNormals() const { return _normals; }
    brion::uint16_tsPtr getVertexSections() const { return _sections; }
    brion::floatsPtr getVertexRelativeDistances() const
    {
        return _relativeDistances;
    }

    brion::uint32_tsPtr getTriangles() const { return _triangles; }
    brion::uint32_tsPtr getTriangleStrip() const { return _triangleStrip; }
private:
    /*--- Private member variables ---*/

    brion::Vector3fsPtr _vertices;
    brion::Vector3fsPtr _normals;
    brion::uint16_tsPtr _sections;
    brion::floatsPtr _relativeDistances;
    brion::uint32_tsPtr _triangles;
    brion::uint32_tsPtr _triangleStrip;
    brion::MeshVersion _version;

    /*--- Private member functions ---*/

    /** Compute per vertex normals.
        @return 0 in case of success, 1 if triangles are missing, 2 if
        degenerate triangles were found.
    */
    int _computeNormals();
};
}
}
}
#endif
