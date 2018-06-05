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

#ifndef RTNEURON_OSGEQ_SCENEDECORATOR_H
#define RTNEURON_OSGEQ_SCENEDECORATOR_H

#include "types.h"

#include <boost/serialization/assume_abstract.hpp>
#include <boost/signals2/signal.hpp>

namespace boost
{
namespace serialization
{
class access;
}
}

namespace osg
{
class Node;
class Group;
class Camera;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/**
   A SceneDecorator is an abstract serializable class that stores view specific
   rendering attributes and applies them to OSG cameras.

   This class is serializable and is meant to be only modified from the
   application node. The view in which this class is attached is
   reponsible of doing the distribution.
*/
class SceneDecorator
{
    friend class boost::serialization::access;

public:
    /*--- Public declarations ---*/

    typedef void DirtySignalSignature();
    typedef boost::signals2::signal<DirtySignalSignature> DirtySignal;

    /*--- Public constructors/destructor ---*/

    virtual ~SceneDecorator() {}
    /*--- Public member functions ---*/

    virtual osg::Group* decorateScenegraph(osg::Node* root) = 0;

    /**
       Called every frame from Channel to apply user specific data to the
       OSG camera object that affects the rendering appearance.
    */
    virtual void updateCamera(osg::Camera* camera) const = 0;

    /**
       Update the decorator node and state sets returned by decorateScenegraph

       Since Views are not available at Node::frameStart, this function
       gives the opportunity to Channels to update their decorator nodes
       at Channel::frameDraw.
     */
    virtual void update() = 0;

    /**
       Clone this decorator into a new one.

       The returned decorator must be in a state than can set to a view and
       is ready to be serialized.
     */
    virtual SceneDecoratorPtr clone() const = 0;

    /*--- Public signals ---*/

    DirtySignal dirty;

private:
    /*--- Private member functions */
    template <class Archive>
    void serialize(Archive&, const unsigned int)
    {
    }
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(SceneDecorator)
}
}
}
#endif
