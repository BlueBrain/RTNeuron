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

#include "NeuronDisplayControl.h"

#include "CircuitSceneAttributes.h"
#include "NeuronModel.h"

#include "data/loaders.h"

#include "config/Globals.h"
#include "config/constants.h"
#include "util/shapes.h"
#include "util/vec_to_vec.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

#include <lunchbox/scopedMutex.h>
#include <lunchbox/spinLock.h>

#include <osg/Geode>
#include <osg/PolygonMode>
#include <osg/PositionAttitudeTransform>
#include <osg/Version>

#include <boost/foreach.hpp>

#include <map>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Static definitions
*/

namespace
{
/* Global data mutex.
   Single mutex when contention is expected to be low and space overhead
   for having a mutex per neuron can be high */
lunchbox::SpinLock s_singleDataLock;

/* Cache of capsule based model of morphologies. */
typedef std::map<const brain::neuron::Morphology*, osg::ref_ptr<osg::Geode>>
    SimpleMorphologyGeometryMap;
SimpleMorphologyGeometryMap s_morphologyGeometryMap;

/*
  Helper functions
*/

inline osg::Vec3 _center(const brain::Vector4f& sample)
{
    return osg::Vec3(sample[0], sample[1], sample[2]);
}
inline float _radius(const brain::Vector4f& sample)
{
    return sample[3] * 0.5;
}

osg::StateSet* _createFlatShadingState()
{
    osg::StateSet* stateSet = new osg::StateSet;
    std::vector<std::string> shaders;
    shaders.push_back("shading/default_color.frag");
    shaders.push_back("shading/phong_mesh.frag");
    shaders.push_back("shading/phong.frag");
    shaders.push_back("geom/flat_mesh.vert");
    shaders.push_back("geom/flat_mesh.geom");
#ifdef OSG_GL3_AVAILABLE
    shaders.push_back("geom/compute_clip_distances.geom");
#endif
    shaders.push_back("main.vert");
    shaders.push_back("main.frag");
    osg::Program* program = Loaders::loadProgram(shaders);
    program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
    program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
    program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    stateSet->setAttributeAndModes(program);

    return stateSet;
}

void _addSkeletonSegments(const brain::Vector4fs& samples,
                          const brain::neuron::SectionType type,
                          osg::Geode& geode)
{
    using SectionType = brain::neuron::SectionType;
    for (size_t i = 0; i != samples.size() - 1; ++i)
    {
        const osg::Vec3 p = _center(samples[i]);
        const float r1 = _radius(samples[i]);
        const osg::Vec3 q = _center(samples[i + 1]);
        const float r2 = _radius(samples[i + 1]);

        const osg::Vec3 center = (p + q) * 0.5;
        const float radius =
            type == SectionType::soma ? 0.25 : std::max(r1, r2);
        const osg::Vec3 axis = p - q;

        osg::Vec4 color;
        if (type == SectionType::soma)
            color = osg::Vec4(0.2, 1.0, 0.2, 1.0);
        else if (type == SectionType::axon)
            color = osg::Vec4(0.2, 0.6, 1, 1);
        else
            color = osg::Vec4(1, 0.1, 0, 1);

        geode.addDrawable(capsuleDrawable(center, radius, axis, color, 0.2));
    }
}

osg::Geode* _getOrCreateSimpleRepresentation(
    const brain::neuron::Morphology& morphology)
{
    static OpenThreads::Mutex s_geometryDataMutex;
    /* Searching the morphology in the map*/
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(s_geometryDataMutex);

    auto& geode = s_morphologyGeometryMap[&morphology];

    if (geode)
        return geode;

    geode = new osg::Geode();

    using S = brain::neuron::SectionType;

    _addSkeletonSegments(morphology.getSoma().getProfilePoints(), S::soma,
                         *geode);

    const auto sections =
        morphology.getSections({S::dendrite, S::apicalDendrite, S::axon});
    for (const auto& section : sections)
    {
        const auto samples = section.getSamples();
        _addSkeletonSegments(samples, section.getType(), *geode);
    }

    if (::getenv("RTNEURON_WIREFRAME_MORPHOLOGIES"))
    {
#ifdef OSG_GL3_AVAILABLE
        std::cerr << "Wireframe skeleton representation mode not "
                  << "available" << std::endl;
#else
        static osg::ref_ptr<osg::StateSet> lineMode;
        if (!lineMode.valid())
        {
            lineMode = new osg::StateSet;
            osg::PolygonMode* polygonMode =
                new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,
                                     osg::PolygonMode::LINE);
            lineMode->setAttributeAndModes(polygonMode);
        }
        geode->setStateSet(lineMode.get());
#endif
    }
    else
    {
        static osg::ref_ptr<osg::StateSet> flatRenderingMode;
        if (!flatRenderingMode.valid())
            flatRenderingMode = _createFlatShadingState();
        geode->setStateSet(flatRenderingMode.get());
    }

    return geode;
}
}

/*
   Constructors/destructtor
*/
Neuron::DisplayControl::DisplayControl(const Neuron* neuron)
    : _neuron(neuron)
{
    osg::ref_ptr<osg::PositionAttitudeTransform> objects(
        new osg::PositionAttitudeTransform());
    objects->setPosition(_neuron->getPosition());
    objects->setAttitude(_neuron->getOrientation());
    _objects = objects;

    osg::Group* morphGeom = new osg::Group;
    _objects->addChild(morphGeom);
    morphGeom->setNodeMask(~CULL_AND_DRAW_MASK);
    _displayMode.setValue(types::NO_DISPLAY);
}

Neuron::DisplayControl::DisplayControl(const DisplayControl& other)
    : _neuron(neuron)
    , _displayMode(

Neuron::DisplayControl::~DisplayControl()
{
}

/*
  Member functions
*/

osg::Node* Neuron::DisplayControl::getNode()
{
    return _objects.get();
}

types::RepresentationMode Neuron::DisplayControl::getRepresentationMode() const
{
    lunchbox::ScopedFastRead mutex(s_singleDataLock);
    return _displayMode.getValue();
}

types::RepresentationMode Neuron::DisplayControl::getRepresentationMode(
    const unsigned long frameNumber) const
{
    lunchbox::ScopedFastRead mutex(s_singleDataLock);
    types::RepresentationMode mode;
    /* If there is no value for the given frameNumber then we return
       the last value stored. */
    if (!_displayMode.getValue(frameNumber, mode))
        mode = _displayMode.getValue();
    return mode;
}

void Neuron::DisplayControl::setRepresentationMode(
    types::RepresentationMode mode)
{
    lunchbox::ScopedFastWrite mutex(s_singleDataLock);

    _displayMode.setValue(mode);

    osg::Group* morphNode = _getMorphologicalSkeleton();
    assert(morphNode);
    if (mode == types::SEGMENT_SKELETON)
    {
        /* Switching to morphology representation.
           Checking if already added and add it if not */
        if (!morphNode->getNumChildren())
        {
            const auto& morphology = *_neuron->getMorphology();
            morphNode->addChild(_getOrCreateSimpleRepresentation(morphology));
        }
        morphNode->setNodeMask(0xFFFFFFFF);
        /* Enabling cull/draw at the node containing the extra objects.
           The group won't be set back to invisible for simplicity. */
        assert(!_objects->getParents().empty());
        _objects->getParent(0)->setNodeMask(0xFFFFFFFF);
    }
    else
    {
        morphNode->setNodeMask(~CULL_AND_DRAW_MASK);
    }
}

osg::Group* Neuron::DisplayControl::_getMorphologicalSkeleton()
{
    assert(_objects->getNumChildren() > 0);
    return _objects->getChild(0)->asGroup();
}
}
}
}
