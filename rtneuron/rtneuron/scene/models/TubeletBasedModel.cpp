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

#include "TubeletBasedModel.h"

#include "config/constants.h"
#include "render/SceneStyle.h"
#include "scene/CircuitSceneAttributes.h"
#include "util/vec_to_vec.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>
#include <brain/neuron/soma.h>

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

/* For more details, have look in the thesis, section 3.3.4, figure 3.13
   https://bbpteam.epfl.ch/project/spaces/display/RTNRN/RTNeuron+-+Portal */

enum EndSide
{
    RIGHT_END = 0,
    LEFT_END = 1
};
void _endTubelet(const osg::Vec3& p0, const osg::Vec3& p1,
                 const float endRadius, const EndSide end, osg::Vec4& cutPlane)
{
    osg::Vec3 p = p1 - p0;
    p.normalize();

    if (end == RIGHT_END)
        /* The cut plane is placed tangent to the spherical cap */
        cutPlane = osg::Vec4(p, -p * (p1 + p * (endRadius * 1.00001f)));
    else
        /* Left ends don't include the spherical cap, they are cut at the
           segment end. */
        cutPlane = osg::Vec4(p, -p * p0);
}

osg::Vec2 _circularSectorAntiAppex(const osg::Vec2& p, const osg::Vec2& q)
{
    /* Computing the mid point between p and q */
    osg::Vec2 m = (p + q) * 0.5;
    /* The triangle [O, p, m] has two known sides and is similar to the
       triangle [O, p, a] where a is the anti appex of the sector.
       We use this to compute the length of the segment Oa. */
    const float l = m.normalize();
    const float a = p * p / l;
    return m * a;
}

float _cathetus(const float hypot, const float cat)
{
    return std::sqrt(hypot * hypot - cat * cat);
}

osg::Vec2 _invertY(const osg::Vec2& p)
{
    return osg::Vec2(p.x(), -p.y());
}

osg::Vec2 _moveTo(const osg::Vec2& p, const osg::Vec2& v, const osg::Vec2& u)
{
    const float cos_a = u * v;
    const float sen_a = u.x() * v.y() - u.y() * v.x();
    /* Rotating p1_2 to the p0_1 reference system */
    return osg::Vec2(p.x() * cos_a - p.y() * sen_a,
                     p.x() * sen_a + p.y() * cos_a);
}

float _xAxisIntersection(const osg::Vec2& p, const osg::Vec2& q)
{
    return p[0] + (q[0] - p[0]) * p[1] / (p[1] - q[1]);
}

/**
   Returns the coordinates of the tangency point of a line passing by
   a point located at (d, 0) with d > 0 and tangent to a circle
   centered at origin and radius r.
 */
osg::Vec2 _tangencyPoint(const float radius, const float distance)
{
    assert(distance * 2 >= radius);
    const float a = (radius * radius) / distance;
    return osg::Vec2(a, _cathetus(radius, a));
}

/**
   Returns the tangency points of the line tangent to two circles whose centers
   are on the x-axis. The radius r1 refers to the circle on the left and r2
   to the circle on the right. The tangency points returned are relative to
   the circle on the left (r1) if distance > 0 and for the circle on the
   right (r2) otherwise.
 */
osg::Vec2 _tangencyPoints(const float distance, const float r1, const float r2)
{
    /* Sanity check */
    if (std::fabs(r1 - r2) > std::fabs(distance) * 2)
        return osg::Vec2(0, 0);

    if (distance > 0)
    {
        /* Returning tangency points for circle 1 */
        if (r1 > r2)
        {
            const float r = r1 - r2;
            const osg::Vec2 t = _tangencyPoint(r, distance);
            assert(t[0] >= 0);
            return osg::Vec2(t * (r1 / r));
        }
        if (r1 < r2)
        {
            const float r = r2 - r1;
            osg::Vec2 t = _tangencyPoint(r, distance);
            assert(t[0] >= 0);
            t[0] = -t[0];
            return osg::Vec2(t * (r2 / r - 1));
        }
        return osg::Vec2(0, r1);
    }
    else
    {
        /* Returning tangency points for circle 2 */
        if (r2 > r1)
        {
            const float r = r2 - r1;
            osg::Vec2 t = _tangencyPoint(r, -distance);
            assert(t[0] >= 0);
            t[0] = -t[0];
            return osg::Vec2(t * (r2 / r));
        }
        if (r2 < r1)
        {
            const float r = r1 - r2;
            osg::Vec2 t = _tangencyPoint(r, -distance);
            assert(t[0] >= 0);
            return osg::Vec2(t * (r1 / r - 1));
        }
        return osg::Vec2(0, r2);
    }
}

void _tubeletsJoint(const osg::Vec3& p0, const osg::Vec3& p1,
                    const osg::Vec3& p2, const float r0, const float r1,
                    const float r2, osg::Vec4& cutPlane)
{
    /* Normalized directions of p0 to p1 and p1 to p2 and distances */
    osg::Vec3 p0_1 = p1 - p0;
    const float d0_1 = p0_1.normalize();
    osg::Vec3 p1_2 = p2 - p1;
    const float d1_2 = p1_2.normalize();

    /* Normal of the plane containing the 3 points. */
    osg::Vec3 n = p0_1 ^ p1_2;
    n.normalize();

    const osg::Vec3 up = n ^ p0_1;
    const osg::Vec2 p1_2_2D(p0_1 * p1_2, up * p1_2);

    /* Finding the tangency points involving p0 and p1. */
    const osg::Vec2 tR = _tangencyPoints(-d0_1, r0, r1);
    /* And now for p1 and p2. */
    const osg::Vec2 tL = _tangencyPoints(d1_2, r1, r2);

    /* Calculating the intersection plane */
    const osg::Vec2 isec0 =
        _circularSectorAntiAppex(tR, _moveTo(tL, p1_2_2D, osg::Vec2(1, 0)));
    const osg::Vec2 isec1 =
        _circularSectorAntiAppex(_invertY(tR), _moveTo(_invertY(tL), p1_2_2D,
                                                       osg::Vec2(1, 0)));
    osg::Vec2 normal = (isec0 - isec1);
    normal.normalize();
    normal = osg::Vec2(normal[1], -normal[0]);
    const osg::Vec3 modelNormal = p0_1 * normal.x() + up * normal.y();
    const osg::Vec3 cutPoint = p1 + p0_1 * _xAxisIntersection(isec0, isec1);
    cutPlane = osg::Vec4(modelNormal, -modelNormal * cutPoint);
}
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
TubeletBasedModel::TubeletBasedModel(const NeuronParts parts,
                                     const ConstructionData& data)
{
    /** \bug If tubelets are used for neurolucida display,
        the decision about whether connecting the first order sections to
        the spherical soma or the detailed mesh model cannot be taken
        locally. */
    _create(parts, data);

    osg::ref_ptr<osg::VertexBufferObject> vbo = new osg::VertexBufferObject();
    _vertices->setVertexBufferObject(vbo);
    _pointRadii->setVertexBufferObject(vbo);
    _cutPlanes->setVertexBufferObject(vbo);

    _useCLOD = (*data.sceneAttr.neuronLODs)("clod", false);
}

void TubeletBasedModel::clip(const std::vector<osg::Vec4d>& planes)
{
    if (planes.empty())
        return;

    LBUNIMPLEMENTED;
}

Skeleton::PortionRanges TubeletBasedModel::postProcess(NeuronSkeleton& skeleton)
{
    return skeleton.postProcess((osg::Vec3*)_vertices->getDataPointer(),
                                _sections->data(), _positions->data(),
                                (uint32_t*)_indices.get(),
                                osg::PrimitiveSet::LINES, _primitiveLength);
}

osg::StateSet* TubeletBasedModel::getModelStateSet(
    const bool subModel, const SceneStyle& style) const
{
    if (!subModel)
        return 0;

    AttributeMap extra;
    if (_useCLOD)
        extra.set("clod", _useCLOD);
    return style.getStateSet(SceneStyle::NEURON_TUBELETS, extra);
}

DetailedNeuronModel::Drawable* TubeletBasedModel::instantiate(
    const SkeletonPtr& skeleton, const CircuitSceneAttributes& sceneAttr)
{
    Drawable* geometry = new Drawable();
    if (_primitiveLength == 0)
        return geometry;

    /* Setting up the geometry object */
    geometry->setVertexArray(_vertices);
    geometry->setVertexAttribArray(TUBELET_POINT_RADIUS_ATTRIB_NUM,
                                   _pointRadii);
    geometry->setVertexAttribBinding(TUBELET_POINT_RADIUS_ATTRIB_NUM,
                                     osg::Geometry::BIND_PER_VERTEX);
    geometry->setVertexAttribArray(TUBELET_CUT_PLANE_ATTRIB_NUM, _cutPlanes);
    geometry->setVertexAttribBinding(TUBELET_CUT_PLANE_ATTRIB_NUM,
                                     osg::Geometry::BIND_PER_VERTEX);

    /* Creating the primitive */
    DrawElementsPortions* primitive =
        new DrawElementsPortions(osg::PrimitiveSet::LINES, _indices,
                                 _primitiveLength);
    /* Checking if there is a DrawElementsPortions holding a EBO for this neuron
       and taking the EBO from it. Creating a new EBO and selecting this
       primitive as the owner otherwise */
    {
        /* This may be called concurrently from CircuitScene::addNeuronList */
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
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

void TubeletBasedModel::_create(const NeuronParts parts,
                                const ConstructionData& data)
{
    const auto& morphology = data.morphology;
    const bool connectToSoma = data.sceneAttr.connectFirstOrderBranches;
    if (parts == NEURON_PARTS_SOMA)
        throw std::runtime_error("Tubelet neuron models cannot be soma only");
    const bool noAxon = parts == NEURON_PARTS_SOMA_DENDRITES;

    /** Special points to connect first order sections if required */
    util::SectionStartPointsMap sectionPoints;
    if (connectToSoma && data.mesh)
        sectionPoints =
            util::extractCellCoreSectionStartPoints(morphology, *data.mesh);

    using S = brain::neuron::SectionType;
    std::vector<S> types{S::dendrite, S::apicalDendrite, S::axon};
    if (noAxon)
        types.pop_back();
    auto sections = morphology.getSections(types);
    util::sortSections(sections);

    const size_t numPoints = util::numberOfPoints(sections);
    _allocate(numPoints);
    osg::Vec3Array& vertices = static_cast<osg::Vec3Array&>(*_vertices);
    std::vector<GLuint> indices;
    indices.reserve((numPoints - sections.size()) * 2);

    GLuint primitiveIndex = 0;
    size_t pointCount = 0;
    for (const auto& section : sections)
    {
        _createSection(section, data, sectionPoints);

        for (size_t i = 0; i < vertices.size() - pointCount - 1;
             ++i, ++primitiveIndex)
        {
            indices.push_back(primitiveIndex);
            indices.push_back(primitiveIndex + 1);
        }
        pointCount = vertices.size();
        ++primitiveIndex;
    }

    _length = vertices.size();
    _primitiveLength = indices.size();

    assert(_pointRadii->size() == _length);
    assert(_cutPlanes->size() == _length);

    _indices.reset(new GLuint[_primitiveLength]);
    memcpy(&_indices[0], &indices[0], sizeof(GLuint) * _primitiveLength);
}

void TubeletBasedModel::_allocate(const size_t numPoints)
{
    _vertices = new osg::Vec3Array();
    _vertices->setDataVariance(osg::Object::STATIC);
    _pointRadii = new osg::FloatArray();
    _pointRadii->setDataVariance(osg::Object::STATIC);
    _cutPlanes = new osg::Vec4Array();
    _cutPlanes->setDataVariance(osg::Object::STATIC);
    _sections.reset(new uint16_ts);
    _positions.reset(new floats);

    static_cast<osg::Vec3Array&>(*_vertices).reserve(numPoints);
    _pointRadii->reserve(numPoints);
    _cutPlanes->reserve(numPoints);
    _sections->reserve(numPoints);
    _positions->reserve(numPoints);
}

void TubeletBasedModel::_createSection(
    const brain::neuron::Section& section, const ConstructionData& data,
    util::SectionStartPointsMap& sectionStarts)
{
    auto samples = section.getSamples();
    auto relativeDistances =
        util::computeRelativeDistances(samples, section.getLength());
    util::removeZeroLengthSegments(samples, relativeDistances);

    size_t index =
        _startSection(section, samples, relativeDistances, data, sectionStarts);

    osg::Vec3Array& vertices = static_cast<osg::Vec3Array&>(*_vertices);

    for (; index != samples.size(); ++index)
    {
        osg::Vec4 cutPlane;
        auto point = _center(samples[index]);
        auto radius = _radius(samples[index]);

        if (index == samples.size() - 1)
        {
            /* Last point in the section */
            const auto& child = util::mainBranchChildSection(section);
            if (child.getID() == section.getID())
            {
                /* Branch terminal segment */
                _endTubelet(vertices.back(), point, radius, RIGHT_END,
                            cutPlane);
            }
            else
            {
                assert(child.getNumSamples() > 1);
                const auto& next = child[1];
                const auto nextPoint = _center(next);
                const float nextRadius = _radius(next);
                _tubeletsJoint(vertices.back(), point, nextPoint,
                               _pointRadii->back(), radius, nextRadius,
                               cutPlane);
            }
        }
        else
        {
            const auto nextPoint = _center(samples[index + 1]);
            const auto nextRadius = _radius(samples[index + 1]);

            /* Checking the degenerate cases in which the tubelet is
               ill-defined. The handling of this cases is a bit hacky
               but it's satisfactory. */
            if (std::fabs(_pointRadii->back() - radius) >
                (vertices.back() - point).length())
            {
                cutPlane = osg::Vec4(0, 0, 0, 0);
            }
            else if (std::fabs(nextRadius - radius) >
                     (nextPoint - point).length())
            {
                cutPlane = osg::Vec4(0, 0, 0, 0);
            }
            else
            {
                _tubeletsJoint(vertices.back(), point, nextPoint,
                               _pointRadii->back(), radius, nextRadius,
                               cutPlane);
            }
        }

        vertices.push_back(point);
        _pointRadii->push_back(radius);
        _cutPlanes->push_back(cutPlane);
        _sections->push_back(section.getID());
        _positions->push_back(relativeDistances[index]);
    }
}

size_t TubeletBasedModel::_startSection(
    const brain::neuron::Section& section, const brain::Vector4fs& samples,
    const brain::floats& relativeDistances, const ConstructionData& data,
    util::SectionStartPointsMap& sectionStarts)
{
    const bool connectToSoma = data.sceneAttr.connectFirstOrderBranches;

    size_t index = 1;
    auto startPoint = _center(samples[0]);
    auto endPoint = _center(samples[1]);
    auto startRadius = _radius(samples[0]);
    auto endRadius = _radius(samples[1]);
    float position = 0;
    osg::Vec4 cutPlane;

    if (!section.hasParent())
    {
        if (connectToSoma && !sectionStarts.empty())
        {
            /* Creating a first tubelet that gives visual continuity
               between the mesh of the cell core and the tubelet
               skeleton. */
            const auto& sectionStart = sectionStarts[section.getID()];
            index = sectionStart.findSampleAfterSectionStart(
                samples, relativeDistances, section.getLength(), startRadius);
            startPoint = sectionStart.point;
            position = sectionStart.position;
            const auto& end = samples[index];
            endPoint = _center(end);
            endRadius = _radius(end);
        }
        else if (connectToSoma)
        {
            /* This case if for the connection to a spherically shaped
               soma. */
            const auto& soma = data.morphology.getSoma();
            const auto& somaCenter = vec_to_vec(soma.getCentroid());
            const float somaRadius = soma.getMeanRadius();

            startPoint -= somaCenter;
            startPoint.normalize();
            startPoint = somaCenter + startPoint * somaRadius;
        }
        _endTubelet(startPoint, endPoint, endRadius, LEFT_END, cutPlane);
    }
    else
    {
        /* Finding the attributes for the beginning of the section as a
           continuation of the parent section. */
        const auto& parent = section.getParent();
        const auto& parentLast = parent[-1];
        const auto& parentBeforeLast = parent[-2];

        /* Checking if this child section continues the main branch from the
           parent. */
        const auto bestChild = util::mainBranchChildSection(parent);
        if (bestChild.getID() == section.getID() &&
            /* Hotfix for 0 length segments */
            parentLast != parentBeforeLast)
        {
            startRadius = _radius(parentLast);
            _tubeletsJoint(_center(parentBeforeLast), startPoint, endPoint,
                           _radius(parentBeforeLast), startRadius, endRadius,
                           cutPlane);
        }
        else
        {
            startRadius *= 0.99;
            _endTubelet(startPoint, endPoint, endRadius, LEFT_END, cutPlane);
        }
    }
    /* Adding the data of the first real point of the section. */
    static_cast<osg::Vec3Array&>(*_vertices).push_back(startPoint);
    _pointRadii->push_back(startRadius);
    _cutPlanes->push_back(cutPlane);
    _sections->push_back(section.getID());
    _positions->push_back(position);

    return index;
}
}
}
}
}
