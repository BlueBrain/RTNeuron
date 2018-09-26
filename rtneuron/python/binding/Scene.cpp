/* Copyright (c) 2006-2018, Ecole Polytechnique Federale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politecnica de Madrid (UPM)
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

#include "RawArray.h"
#include "boost_signal_connect_wrapper.h"
#include "docstrings.h"
#include "gil.h"
#include "helpers.h"

#include "rtneuron/Camera.h"
#include "rtneuron/Scene.h"
#include "rtneuron/types.h"

#include <brain/circuit.h>
#include <brain/spikeReportReader.h>
#include <brain/synapse.h>
#include <brain/synapses.h>

#include <brain/python/arrayHelpers.h>
#include <brain/python/helpers.h>

#include <osg/BoundingSphere>

#include <boost/python.hpp>
#include <boost/signals2/signal.hpp>

using namespace boost::python;
using namespace bbp::rtneuron;

static AttributeMap defaultAttributeMap;

// export_Scene ----------------------------------------------------------------

/* Needed to wrap the Scene::cellSetSelected signal parameters */
struct GIDSetToNumpy
{
    static PyObject* convert(const brain::GIDSet& gids)
    {
        boost::python::object object = brain_python::toNumpy(
            std::vector<uint32_t>(gids.begin(), gids.end()));
        return boost::python::incref(object.ptr());
    }
};

list Scene_getObjects(Scene* scene)
{
    std::vector<Scene::ObjectPtr> objects = scene->getObjects();
    list l;
    for (std::vector<Scene::ObjectPtr>::const_iterator i = objects.begin();
         i != objects.end(); ++i)
        l.append(*i);
    return l;
}

tuple Scene_getSomasBoundingSphere(Scene* scene)
{
    const osg::BoundingSphere sphere = scene->getSomasBoundingSphere();
    list l;
    l.append(sphere.center().x());
    l.append(sphere.center().y());
    l.append(sphere.center().z());
    return make_tuple(l, sphere.radius());
}

tuple Scene_getCircuitSceneBoundingSphere(Scene* scene)
{
    return Scene_getSomasBoundingSphere(scene);
}

tuple Scene_getSynapsesBoundingSphere(Scene* scene)
{
    const osg::BoundingSphere sphere = scene->getSynapsesBoundingSphere();
    list l;
    l.append(sphere.center().x());
    l.append(sphere.center().y());
    l.append(sphere.center().z());
    return make_tuple(l, sphere.radius());
}

object Object_getObject(Scene::Object* object)
{
    boost::any o = object->getObject();
    if (o.empty())
        return boost::python::object();

    try
    {
        auto&& gids = boost::any_cast<uint32_ts&&>(o);
        return boost::python::object(brain_python::toNumpy(std::move(gids)));
    }
    catch (...)
    {
    }
    try
    {
        auto&& synapses = boost::any_cast<brain::Synapses&&>(o);
        return boost::python::object(std::move(synapses));
    }
    catch (...)
    {
    }
    try
    {
        auto&& name = boost::any_cast<std::string&&>(o);
        return boost::python::object(std::move(name));
    }
    catch (...)
    {
    }

    PyErr_SetString(PyExc_RuntimeError,
                    "Unknown type returned from Scene::Object::getObject");
    return boost::python::object();
}

AttributeMap& Scene_getAttributes(Scene& view)
{
    AttributeMap& attributes = view.getAttributes();
    /** \todo Move this assignment to the class constructor so it's only
        done once */
    attributes.setExtraDocstring(DOXY_FN(bbp::rtneuron::Scene::getAttributes));
    return attributes;
}

/* All the GIL releases below are needed to avoid deadlocks between the
   scene signals that need to lock the GIL and may be called within
   critical sections inside Scene::Impl and any other Scene method invoked
   from the interpreter. */

Scene::ObjectPtr Scene_addNeurons(Scene& scene, object& gids,
                                  const AttributeMap& attributes)
{
    const auto gidset = brain_python::gidsFromPython(gids);
    ReleaseGIL release;
    return scene.addNeurons(gidset, attributes);
}

Scene::ObjectPtr Scene_addAfferentSynapses(Scene& scene,
                                           const brain::Synapses& synapses,
                                           const AttributeMap& attributes)
{
    ReleaseGIL release;
    return scene.addAfferentSynapses(synapses, attributes);
}

Scene::ObjectPtr Scene_addEfferentSynapses(Scene& scene,
                                           const brain::Synapses& synapses,
                                           const AttributeMap& attributes)
{
    ReleaseGIL release;
    return scene.addEfferentSynapses(synapses, attributes);
}

Scene::ObjectPtr Scene_addModel1(Scene& scene, const char* filename,
                                 const object& transform,
                                 const AttributeMap& attributes)
{
    auto matrix = brain_python::fromNumpy<brain::Matrix4f>(transform);
    ReleaseGIL release;
    return scene.addModel(filename, matrix, attributes);
}

Scene::ObjectPtr Scene_addModel2(Scene& scene, const char* filename,
                                 const char* transform,
                                 const AttributeMap& attributes)
{
    ReleaseGIL release;
    return scene.addModel(filename, transform, attributes);
}

Scene::ObjectPtr Scene_addGeometry(Scene& scene, object vertices_,
                                   object primitive_, object colors_,
                                   object normals_,
                                   const AttributeMap& attributes)
{
    RawArray<float> vertices(vertices_);
    if (vertices.shape.size() != 2 ||
        (vertices.shape[1] != 3 && vertices.shape[1] != 4))
    {
        const char* message = "Invalid vertex array shape";
        PyErr_SetString(PyExc_ValueError, message);
        throw_error_already_set();
    }

    std::string modeStr;
    if (attributes.get("mode", modeStr) == 1)
        LBWARN << "Scene::addGeometry: mode attribute is deprecated and no "
                  "longer needed"
               << std::endl;

    RawArray<unsigned int> indices(primitive_);
    if (!indices.shape.empty() &&
        (indices.shape.size() != 2 ||
         (indices.shape[1] != 2 && indices.shape[1] != 3)))
    {
        const char* message = "Invalid primitive array shape";
        PyErr_SetString(PyExc_ValueError, message);
        throw_error_already_set();
    }
    const size_t indexCount = indices.size();
    const size_t vertexCount = vertices.shape[0];
    for (size_t i = 0; i < indexCount; ++i)
    {
        if (indices.array[i] >= vertexCount)
        {
            const char* message = "Vertex index out of bounds";
            PyErr_SetString(PyExc_ValueError, message);
            throw_error_already_set();
        }
    }

    RawArray<float> colors(colors_);
    if (!colors.shape.empty() &&
        !(
            /* Either the colors array is a single color */
            (colors.shape.size() == 1 && colors.shape[0] == 4) ||
            /* Or the same size as the vertices */
            (colors.shape.size() == 2 && colors.shape[0] == vertices.shape[0] &&
             colors.shape[1] == 4)))
    {
        const char* message = "Invalid color array shape";
        PyErr_SetString(PyExc_ValueError, message);
        throw_error_already_set();
    }

    RawArray<float> normals;
    if (!normals.shape.empty() &&
        !(normals.shape.size() == 2 && normals.shape[0] == vertices.shape[0] &&
          normals.shape[1] == 3))
    {
        const char* message = "Invalid normal array shape";
        PyErr_SetString(PyExc_ValueError, message);
        throw_error_already_set();
    }

    osg::ref_ptr<osg::Array> vertexArray;
    if (vertices.shape[1] == 3)
    {
        vertexArray =
            new osg::Vec3Array((osg::Vec3f*)&vertices.array[0],
                               (osg::Vec3f*)&vertices.array[vertices.size()]);
    }
    else
    {
        vertexArray =
            new osg::Vec4Array((osg::Vec4f*)&vertices.array[0],
                               (osg::Vec4f*)&vertices.array[vertices.size()]);
    }

    osg::ref_ptr<osg::DrawElementsUInt> primitiveArray;
    if (!indices.shape.empty())
    {
        const osg::PrimitiveSet::Mode mode = indices.shape[1] == 3
                                                 ? osg::PrimitiveSet::TRIANGLES
                                                 : osg::PrimitiveSet::LINES;
        primitiveArray = new osg::DrawElementsUInt(mode, &indices.array[0],
                                                   &indices.array[indexCount]);
    }

    osg::ref_ptr<osg::Vec4Array> colorArray =
        colors.shape.empty()
            ? 0
            : new osg::Vec4Array((osg::Vec4f*)&colors.array[0],
                                 (osg::Vec4f*)&colors.array[colors.size()]);

    osg::ref_ptr<osg::Vec3Array> normalArray =
        normals.shape.empty()
            ? 0
            : new osg::Vec3Array((osg::Vec3f*)&normals.array[0],
                                 (osg::Vec3f*)&normals.array[normals.size()]);

    ReleaseGIL release;
    return scene.addGeometry(vertexArray, primitiveArray, colorArray,
                             normalArray, attributes);
}

void Scene_remove(Scene& scene, const Scene::ObjectPtr& object)
{
    ReleaseGIL release;
    scene.remove(object);
}

void Scene_clear(Scene& scene)
{
    ReleaseGIL release;
    scene.clear();
}

void Scene_update(Scene& scene)
{
    ReleaseGIL release;
    scene.update();
}

void Scene_highlight(Scene& scene, const object& target, bool on)
{
    const auto gidset = brain_python::gidsFromPython(target);
    ReleaseGIL release;
    scene.highlight(gidset, on);
}

void Scene_setSimulation1(Scene& scene, const CompartmentReportPtr& report)
{
    ReleaseGIL release;
    scene.setSimulation(report);
}

void Scene_setSimulation2(Scene& scene, const SpikeReportReaderPtr& report)
{
    ReleaseGIL release;
    scene.setSimulation(report);
}

void Scene_clearSimulation(Scene& scene)
{
    ReleaseGIL release;
    scene.setSimulation(SpikeReportReaderPtr());
    scene.setSimulation(CompartmentReportPtr());
}

object Scene_getHighlightedNeurons(Scene& scene)
{
    return brain_python::toNumpy(
        brain_python::toVector(scene.getHighlightedNeurons()));
}

object Scene_getNeuronSelectionMask(Scene& scene)
{
    return brain_python::toNumpy(
        brain_python::toVector(scene.getNeuronSelectionMask()));
}

void Scene_setNeuronSelectionMask(Scene& scene, const object& target)
{
    scene.setNeuronSelectionMask(brain_python::gidsFromPython(target));
}

void Scene_pick(const Scene& scene, object origin, object direction)
{
    scene.pick(extract_Vector3(origin), extract_Vector3(direction));
}

void Scene_setClipPlane(Scene& scene, const unsigned int index, object plane)
{
    scene.setClipPlane(index, extract_Vector4(plane));
}

list Scene_getClipPlane(Scene& scene, const unsigned int index)
{
    list result;
    auto plane = scene.getClipPlane(index);
    for (int i = 0; i != 4; ++i)
        result.append(plane[i]);
    return result;
}

void Scene_Object_apply(Scene::Object& object,
                        const Scene::ObjectOperationPtr& operation)
{
    ReleaseGIL release;
    object.apply(operation);
}

void Scene_Object_update(Scene::Object& object)
{
    ReleaseGIL release;
    object.update();
}

Scene::ObjectPtr Scene_Object_query(Scene::Object& object,
                                    boost::python::object list,
                                    const bool checkIds = false)
{
    brion::uint32_ts ids;
    if (brain_python::isArray(list))
        brain_python::gidsFromNumpy(list, ids);
    else
    {
        ids.reserve(len(list));
        stl_input_iterator<uint32_t> i(list), end;
        while (i != end)
            ids.push_back(*i++);
    }
    ReleaseGIL release;
    return object.query(ids, checkIds);
}

void export_Scene()
// clang-format off
{

class_<Scene, ScenePtr, boost::noncopyable> sceneWrapper(
    "Scene", DOXY_CLASS(bbp::rtneuron::Scene), no_init);

scope sceneScope = sceneWrapper;

/* Nested classes */

WRAP_MEMBER_SIGNAL(Scene, ProgressSignal)
WRAP_MEMBER_SIGNAL(Scene, CellSelectedSignal)
WRAP_MEMBER_SIGNAL(Scene, CellSetSelectedSignal)
WRAP_MEMBER_SIGNAL(Scene, SynapseSelectedSignal)

class_<Scene::ObjectOperation, Scene::ObjectOperationPtr, boost::noncopyable>
("ObjectOperation", no_init)
;

class_<Scene::Object, Scene::ObjectPtr, boost::noncopyable>
("Object", no_init)
    .add_property("attributes", make_function(&Scene::Object::getAttributes,
                                              return_internal_reference<>()))
    .def("update", Scene_Object_update,
         DOXY_FN(bbp::rtneuron::Scene::Object::update))
    .def("apply", Scene_Object_apply, (arg("operation")),
         DOXY_FN(bbp::rtneuron::Scene::Object::apply))
    .def("query", Scene_Object_query, (arg("ids"), arg("check_ids") = false),
         DOXY_FN(bbp::rtneuron::Scene::Object::query))
    .add_property("object", Object_getObject,
         DOXY_FN(bbp::rtneuron::Scene::Object::getObject))
;

boost::python::to_python_converter<brain::GIDSet, GIDSetToNumpy>();

/* Scene declarations */
sceneWrapper
    .def("addNeurons", Scene_addNeurons,
         (arg("neurons"), arg("attributes") = boost::ref(defaultAttributeMap)),
         DOXY_FN(bbp::rtneuron::Scene::addNeurons))
    .def("addAfferentSynapses", Scene_addAfferentSynapses,
         (arg("synapses"), arg("attributes") = boost::ref(defaultAttributeMap)),
         DOXY_FN(bbp::rtneuron::Scene::addAfferentSynapses))
    .def("addEfferentSynapses", Scene_addEfferentSynapses,
         (arg("synapses"), arg("attributes") = boost::ref(defaultAttributeMap)),
         DOXY_FN(bbp::rtneuron::Scene::addEfferentSynapses))
    .def("setSimulation", Scene_setSimulation1,
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::Scene::setSimulation(const CompartmentReportPtr&)))
    .def("setSimulation", Scene_setSimulation2,
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::Scene::setSimulation(const SpikeReportReaderPtr&)))
    .def("clearSimulation", Scene_clearSimulation,
         DOXY_FN(bbp::rtneuron::Scene::clearSimulation))
    .def("addModel", Scene_addModel1,
         (arg("model"), arg("transformation"),
          arg("attributes") = boost::ref(defaultAttributeMap)),
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::Scene::addModel(const char *, const Matrix4f&, const AttributeMap&)))
    .def("addModel", Scene_addModel2,
         (arg("model"), arg("transformation") = "",
          arg("attributes") = boost::ref(defaultAttributeMap)),
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::Scene::addModel(const char *, const char *, const AttributeMap&)))
    .def("addGeometry", Scene_addGeometry,
         (arg("vertices"), arg("primitive") = object(),
          arg("colors") = object(), arg("normals") = object(),
          arg("attributes") = boost::ref(defaultAttributeMap)),
         DOXY_FN(bbp::rtneuron::Scene::addGeometry))
    .add_property("circuit", &Scene::getCircuit, &Scene::setCircuit,
         (std::string("Get: ") + DOXY_FN(bbp::rtneuron::Scene::getCircuit) +
         "\nSet: " + DOXY_FN(bbp::rtneuron::Scene::setCircuit)).c_str())
    .add_property("objects", Scene_getObjects,
         DOXY_FN(bbp::rtneuron::Scene::getObjects))
    .add_property("circuitBoundingSphere", Scene_getCircuitSceneBoundingSphere,
         DOXY_FN(bbp::rtneuron::Scene::getCircuitSceneBoundingSphere))
    .add_property("somasBoundingSphere", Scene_getSomasBoundingSphere,
         DOXY_FN(bbp::rtneuron::Scene::getSomasBoundingSphere()))
    .add_property("synapsesBoundingSphere", Scene_getSynapsesBoundingSphere,
         DOXY_FN(bbp::rtneuron::Scene::getSynapsesBoundingSphere()))
    .def("remove", Scene_remove, DOXY_FN(bbp::rtneuron::Scene::remove))
    .def("clear", Scene_clear, DOXY_FN(bbp::rtneuron::Scene::clear))
    .def("update", Scene_update, DOXY_FN(bbp::rtneuron::Scene::update))
    .def("highlight", Scene_highlight,
         DOXY_FN(bbp::rtneuron::Scene::highlight(const GIDSet&, bool)))
    .add_property("highlightedNeurons", &Scene_getHighlightedNeurons,
                  DOXY_FN(bbp::rtneuron::Scene::getHighlightedNeurons))
    .add_property("neuronSelectionMask", Scene_getNeuronSelectionMask,
                  Scene_setNeuronSelectionMask,
                  DOXY_FN(bbp::rtneuron::Scene::getNeuronSelectionMask))
    .def("pick", Scene_pick, (arg("origin"), arg("direction")),
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::Scene::pick(const Vector3f&, const Vector3f&) const))
    .def("pick", (void (Scene::*)(const View&, float, float, float, float) const)
         &Scene::pick,
         (arg("view"), arg("left"), arg("right"), arg("bottom"), arg("top")),
         /* Do not wrap this line or change whitespace */
         DOXY_FN(bbp::rtneuron::Scene::pick(const View&, float, float, float, float) const))
    .def("setClipPlane", Scene_setClipPlane, (arg("index"), arg("plane")),
         DOXY_FN(bbp::rtneuron::Scene::setClipPlane))
    .def("getClipPlane", Scene_getClipPlane, (arg("index")),
         DOXY_FN(bbp::rtneuron::Scene::getClipPlane))
    .def("clearClipPlanes", &Scene::clearClipPlanes,
         DOXY_FN(bbp::rtneuron::Scene::clearClipPlanes))
    .add_property("attributes", make_function(Scene_getAttributes,
                                              return_internal_reference<>()),
        DOXY_FN(bbp::rtneuron::Scene::getAttributes))
    .def_readonly("cellSelected", &Scene::cellSelected,
                  DOXY_VAR(bbp::rtneuron::Scene::cellSelected))
    .def_readonly("cellSetSelected", &Scene::cellSetSelected,
                  DOXY_VAR(bbp::rtneuron::Scene::cellSetSelected))
    .def_readonly("synapseSelected", &Scene::synapseSelected,
                  DOXY_VAR(bbp::rtneuron::Scene::synapseSelected))
    .def_readonly("progress", &Scene::progress,
                  DOXY_VAR(bbp::rtneuron::Scene::progress))
;

}
// clang-format on
