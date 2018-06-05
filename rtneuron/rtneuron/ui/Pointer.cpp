/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
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

#include "Pointer.h"

#include <co/dataIStream.h>
#include <co/dataOStream.h>

#include <osg/Geode>
#include <osg/ShapeDrawable>

#include <cmath>

namespace bbp
{
namespace rtneuron
{
/*
  Helper functions
*/
static co::DataOStream& operator<<(co::DataOStream& out, const osg::Vec3f& v)
{
    out << v[0] << v[1] << v[2];
    return out;
}

static co::DataIStream& operator>>(co::DataIStream& in, osg::Vec3f& v)
{
    in >> v[0] >> v[1] >> v[2];
    return in;
}

/*
  Constructors
*/
Pointer::Pointer()
    : _len(10000.0f)
    , _radius(1.f)
    , _distance(osg::Vec3(0.0f, -40.f, -100.f))
{
    osg::ref_ptr<osg::Cylinder> cylinder =
        new osg::Cylinder(osg::Vec3(0, 0, 0), _radius, _len);
    osg::ref_ptr<osg::Sphere> cap =
        new osg::Sphere(osg::Vec3(0, 0, _len / 2), _radius);

    osg::ref_ptr<osg::Geode> ray = new osg::Geode();
    ray->addDrawable(new osg::ShapeDrawable(cylinder));
    ray->addDrawable(new osg::ShapeDrawable(cap));

    _pointerGeometry = new osg::MatrixTransform();
    _pointerGeometry->addChild(ray);
}

/*
  Destructor
*/
Pointer::~Pointer()
{
}

/*
  Member functions
*/
osg::Node* Pointer::getGeometry() const
{
    return _pointerGeometry;
}

typedef co::Array<osg::Matrix::value_type> MatrixData;

void Pointer::serialize(co::DataOStream& os)
{
    os << _direction << _position.ptr() << MatrixData(_manip.ptr(), 16)
       << _radius << _distance;
}

void Pointer::deserialize(co::DataIStream& is)
{
    float radius;
    osg::Vec3 position, direction, distance;
    osg::Matrix manip;
    is >> direction >> position >> MatrixData(manip.ptr(), 16) >> radius >>
        distance;

    setDistance(distance);
    setRadius(radius);
    _updatePointerMatrix(position, direction, manip);
}

void Pointer::setManipMatrix(const osg::Matrix manip)
{
    _manip = manip;
}

void Pointer::adjust(float multiplier, bool relative)
{
    /* We only want to emit dirty once so we set the distance first,
       which is simpler, and then setRadius is used. */
    if (relative)
        _distance *= multiplier;
    else
        _distance = osg::Vec3(0.0f, -40.f, -100.f) * multiplier;
    if (relative)
        setRadius(getRadius() * multiplier);
    else
        setRadius(multiplier);
}

float Pointer::getRadius() const
{
    return _radius;
}

void Pointer::setRadius(const float radius)
{
    _radius = radius;
    osg::Geode* ray = _pointerGeometry->getChild(0)->asGeode();
    osg::Drawable* cylinder = ray->getDrawable(0);
    osg::Drawable* cap = ray->getDrawable(1);

    static_cast<osg::Cylinder*>(cylinder->getShape())->setRadius(_radius);
    static_cast<osg::Sphere*>(cap->getShape())->setRadius(_radius);

    cylinder->dirtyDisplayList();
    cap->dirtyDisplayList();
    dirty();
}

osg::Vec3 Pointer::getDistance() const
{
    return _distance;
}

void Pointer::setDistance(const osg::Vec3 distance)
{
    _distance = distance;
    dirty();
}

void Pointer::_updatePointerMatrix(const osg::Vec3& position,
                                   const osg::Vec3& direction,
                                   const osg::Matrix& manip)
{
    _position = position;
    _direction = direction;
    _manip = manip;

    osg::Matrix& m = _pointerMatrix;
    m.makeRotate(
        osg::Quat(-osg::DegreesToRadians(_direction.x()), osg::Vec3(1, 0, 0)));
    m.postMultRotate(
        osg::Quat(-osg::DegreesToRadians(_direction.z()), osg::Vec3(0, 1, 0)));
    m.postMultTranslate(osg::Vec3(_position.x(), -_position.y(), 0) * 0.05);

    /* Moving the pointer geometry down and away from the camera */
    m.postMultTranslate(_distance);

    /* Revert camera manipulator transformations */
    m.postMult(_manip);

    _pointerGeometry->setMatrix(osg::Matrixd::translate(0.0f, 0.0f, -_len / 2) *
                                m);
    dirty();
}
}
}
