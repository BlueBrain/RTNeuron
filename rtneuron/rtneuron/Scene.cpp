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

#include "detail/Configurable.h"

#include "InitData.h"
#include "SceneImpl.h"

#include "Camera.h"
#include "Scene.h"

#include <brain/circuit.h>

#include <vmmlib/ray.hpp>
#include <vmmlib/vector.hpp>

namespace bbp
{
namespace rtneuron
{
/*
  Scene::Object
*/
Scene::Object::~Object()
{
}

Scene::ObjectPtr Scene::Object::query(const uint32_ts&, const bool)
{
    LBTHROW(std::runtime_error("Subset query unimplemented"));
}

/*
  Scene::ObjectOperation
*/
Scene::ObjectOperation::~ObjectOperation()
{
}

/*
  Constructors/destructor
*/
Scene::Scene(const RTNeuronPtr& application, const AttributeMap& attributes)
    : _impl(new _Impl(attributes, this))
    , _application(application)
{
}

Scene::~Scene()
{
    _invalidate();
}

/*
  Member functions
*/
AttributeMap& Scene::getAttributes()
{
    _checkValid();
    return _impl->getAttributes();
}

void Scene::setCircuit(const CircuitPtr& circuit)
{
    return _impl->setCircuit(circuit);
}

CircuitPtr Scene::getCircuit() const
{
    return _impl->getCircuit();
}

Scene::ObjectPtr Scene::addNeurons(const GIDSet& gids,
                                   const AttributeMap& attributes)
{
    _checkValid();
    return _impl->addNeurons(gids, attributes);
}

Scene::ObjectPtr Scene::addEfferentSynapses(const brain::Synapses& synapses,
                                            const AttributeMap& attributes)
{
    _checkValid();
    return _impl->addEfferentSynapses(synapses, attributes);
}

Scene::ObjectPtr Scene::addAfferentSynapses(const brain::Synapses& synapses,
                                            const AttributeMap& attributes)

{
    _checkValid();
    return _impl->addAfferentSynapses(synapses, attributes);
}

Scene::ObjectPtr Scene::addModel(const char* filename,
                                 const Matrix4f& transform,
                                 const AttributeMap& attributes)
{
    _checkValid();
    return _impl->addModel(filename, transform, attributes);
}

Scene::ObjectPtr Scene::addModel(const char* filename, const char* transform,
                                 const AttributeMap& attributes)
{
    _checkValid();
    return _impl->addModel(filename, transform, attributes);
}

Scene::ObjectPtr Scene::addGeometry(
    const osg::ref_ptr<osg::Array>& vertices,
    const osg::ref_ptr<osg::DrawElementsUInt>& primitive,
    const osg::ref_ptr<osg::Vec4Array>& colors,
    const osg::ref_ptr<osg::Vec3Array>& normals, const AttributeMap& attributes)
{
    _checkValid();
    return _impl->addGeometry(vertices, primitive, colors, normals, attributes);
}

void Scene::update()
{
    _checkValid();
    _impl->update();
}

std::vector<Scene::ObjectPtr> Scene::getObjects()
{
    _checkValid();
    return _impl->getObjects();
}

void Scene::remove(const Scene::ObjectPtr& object)
{
    _checkValid();
    _impl->remove(object);
}

void Scene::clear()
{
    _checkValid();
    _impl->clear();
}

void Scene::highlight(const GIDSet& gids, bool on)
{
    _checkValid();
    _impl->highlight(gids, on);
}

void Scene::setNeuronSelectionMask(const GIDSet& gids)
{
    _checkValid();
    _impl->setNeuronSelectionMask(gids);
}

const GIDSet& Scene::getNeuronSelectionMask() const
{
    _checkValid();
    return _impl->getNeuronSelectionMask();
}

const GIDSet& Scene::getHighlightedNeurons() const
{
    _checkValid();
    return _impl->getHighlightedNeurons();
}

void Scene::setSimulation(const CompartmentReportPtr& report)
{
    _checkValid();
    _impl->setSimulation(report);
}

void Scene::setSimulation(const SpikeReportReaderPtr& report)
{
    _checkValid();
    _impl->setSimulation(report);
}

void Scene::pick(const Vector3f& origin, const Vector3f& direction) const
{
    _checkValid();
    const Rayf ray(origin, direction);
    _impl->pick(ray);
}

void Scene::pick(const View& view, const float left, const float right,
                 const float bottom, const float top) const
{
    _checkValid();
    Camera& camera = *view.getCamera();
    Vector3f position;
    Orientation orientation;
    camera.getView(position, orientation);

    Planes planes;
    Vector4f frustum;
    float near;
    /* Adjusting the camera planes to the input rectangle */
    if (camera.isOrtho())
        camera.getProjectionOrtho(frustum[0], frustum[1], frustum[2],
                                  frustum[3]);
    else
        camera.getProjectionFrustum(frustum[0], frustum[1], frustum[2],
                                    frustum[3], near);
    const float width = frustum[1] - frustum[0];
    const float height = frustum[3] - frustum[2];
    const float l = frustum[0] + width * left;
    const float r = frustum[0] + width * right;
    const float b = frustum[2] + height * bottom;
    const float t = frustum[2] + height * top;

    const Vector3f axis(orientation[0], orientation[1], orientation[2]);
    const Matrix4f rotation(vmml::Quaternionf(orientation[3] * M_PI / 180,
                                              axis),
                            Vector3f());

    /* Computing camera planes.
       Each plane is given by its normal and the value d = -N·P, where P
       is a point on the plane. */
    if (camera.isOrtho())
    {
        /* Correcting the projection matrix with the model scale. */
        const float scale = (double)view.getAttributes()("model_scale", 1.0);

        /* Computing the horizontal (u) and vertical (v) normal vectors.
           Note that the selections lie in the negative hemispace of the
           planes, so each plane will have a different normal. */
        const Vector3f u(rotation * Vector3f(1, 0, 0));
        const Vector3f v(rotation * Vector3f(0, 1, 0));

        /* Left plane */
        planes.push_back(Vector4f(-u, u.dot(position + (u * l * scale))));
        /* Right plane */
        planes.push_back(Vector4f(u, -u.dot(position + (u * r * scale))));
        /* Bottom plane */
        planes.push_back(Vector4f(-v, v.dot(position + (v * b * scale))));
        /* Top plane */
        planes.push_back(Vector4f(v, -v.dot(position + (v * t * scale))));

        /* Adding an extra plane to not select objects behind the camera
           position */
        const Vector3f z(rotation * Vector3f(0, 0, 1));
        planes.push_back(Vector4f(z, -z.dot(position)));
    }
    else
    {
        /* In orthographic projections all planes contain the camera
           position. */
        Vector3f normals[4] = {
            Vector3f(rotation * Vector3f(-near, 0, -l)), /* left */
            Vector3f(rotation * Vector3f(near, 0, r)),   /* right */
            Vector3f(rotation * Vector3f(0, -near, -b)), /* bottom */
            Vector3f(rotation * Vector3f(0, near, t))};  /* top */
        for (size_t i = 0; i != 4; ++i)
        {
            normals[i].normalize();
            planes.push_back(Vector4f(normals[i], -normals[i].dot(position)));
        }
    }

    _impl->pick(planes);
}

void Scene::setClipPlane(unsigned int index, const Vector4f& plane)
{
    _checkValid();
    _impl->setClipPlane(index, plane);
}

Vector4f Scene::getClipPlane(unsigned int index) const
{
    _checkValid();
    return _impl->getClipPlane(index);
}

void Scene::clearClipPlanes()
{
    _checkValid();
    _impl->clearClipPlanes();
}

void Scene::mapSimulation(const uint32_t frameNumber, const float milliseconds)
{
    _checkValid();
    _impl->mapSimulation(frameNumber, milliseconds);
}

unsigned int Scene::prepareSimulation(const uint32_t frameNumber,
                                      const float milliseconds)
{
    _checkValid();
    return _impl->prepareSimulation(frameNumber, milliseconds);
}

const CompartmentReportPtr& Scene::getCompartmentReport() const
{
    _checkValid();
    return _impl->getCompartmentReport();
}

const core::SpikeReportPtr& Scene::getSpikeReport() const
{
    _checkValid();
    return _impl->getSpikeReport();
}

std::shared_ptr<osgEq::Scene> Scene::getSceneProxy()
{
    _checkValid();
    return _impl;
}

osg::BoundingSphere Scene::getCircuitSceneBoundingSphere() const
{
    _checkValid();
    return _impl->getSomasBoundingSphere();
}

osg::BoundingSphere Scene::getSomasBoundingSphere() const
{
    _checkValid();
    return _impl->getSomasBoundingSphere();
}

osg::BoundingSphere Scene::getSynapsesBoundingSphere() const
{
    _checkValid();
    return _impl->getSynapsesBoundingSphere();
}

osg::BoundingSphere Scene::getExtraModelsBoundingSphere() const
{
    _checkValid();
    return _impl->getExtraModelsBoundingSphere();
}

void Scene::_invalidate()
{
    if (_impl)
    {
        _impl->_invalidateParent();
        _impl.reset();
    }
}

void Scene::_checkValid() const
{
    if (!_impl)
        LBTHROW(std::runtime_error("Invalid operation on Scene object."));
}
}
}
