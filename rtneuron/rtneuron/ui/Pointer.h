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

#ifndef RTNEURON_API_UI_POINTER_H
#define RTNEURON_API_UI_POINTER_H

#include "../types.h"

#include <osg/MatrixTransform>

#include <boost/noncopyable.hpp>
#include <boost/signals2/signal.hpp>

namespace co
{
class DataIStream;
class DataOStream;
}

namespace bbp
{
namespace rtneuron
{
/**
 * A base class for a pointer device to control the camera and to perform
 * interactions.
 */
class Pointer : public boost::noncopyable
{
public:
    /*--- Public declarations ---*/

    typedef void RayTransformSignature(osg::Vec3&);
    typedef boost::signals2::signal<RayTransformSignature> RayTransformSignal;

    typedef void DirtySignalSignature();
    typedef boost::signals2::signal<DirtySignalSignature> DirtySignal;

    typedef void StereoCorrectionSignalSignature(float);
    typedef boost::signals2::signal<StereoCorrectionSignalSignature>
        StereoCorrectionSignal;

    /*--- Public member functions C++ only ---*/

    /** @name C++ only public interface */
    ///@{

    /**
       \brief Derived classes use this function to update the transformation
       matrix of the pointer geometry.

       The signal dirty may be emmited.
    */
    virtual void update() = 0;

    /** \returns A reference to the internally stored node with the
        pointer geometry */
    osg::Node* getGeometry() const;

    void serialize(co::DataOStream& os);

    void deserialize(co::DataIStream& is);

    void setManipMatrix(const osg::Matrix manip);

    void adjust(const float multiplier, bool relative = false);

    void setRadius(const float radius);

    float getRadius() const;

    void setDistance(const osg::Vec3 distance);

    osg::Vec3 getDistance() const;

    /*--- Public signals ---*/

    RayPickSignal pick;
    RayTransformSignal transformCoord;

    DirtySignal dirty;

    StereoCorrectionSignal stereoCorrection;

    ///@}

protected:
    /*--- Protected constructors/destructors ---*/

    Pointer();

    virtual ~Pointer();

protected:
    /*--- Protected member attributes ---*/

    osg::Matrix _pointerMatrix;
    osg::Matrix _manip;

    const float _len;
    float _radius;
    osg::Vec3 _distance;

    osg::Vec3 _direction;
    /** \todo JH: position is a 2D coordinate associated to the Wiimote in
        particular. This should be changed to be the actual 3D position and
        direction of the 3D pointer. */
    osg::Vec3 _position;

    /*--- Protected member functions ---*/

    void _updatePointerMatrix(const osg::Vec3& position,
                              const osg::Vec3& direction,
                              const osg::Matrix& manip);

private:
    /*--- Private member attributes ---*/

    osg::ref_ptr<osg::MatrixTransform> _pointerGeometry;
};
}
}

#endif /* RTNEURON_API_POINTER_H */
