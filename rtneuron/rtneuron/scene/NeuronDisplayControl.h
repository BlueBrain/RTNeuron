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

#ifndef RTNEURON_NEURONDISPLAYCONTROL_H
#define RTNEURON_NEURONDISPLAYCONTROL_H

#include "data/Neuron.h"
#include "scene/PerFrameAttribute.h"

/** \cond DOXYGEN_IGNORE_COND */
namespace osg
{
class Node;
class Group;
class Geode;
}
/** \endcond */

namespace bbp
{
namespace rtneuron
{
namespace core
{
class CircuitSceneAttributes;

/**
   Class that creates scene objects for additional data representations.

   It also handles representation mode changes.
*/
class Neuron::DisplayControl
{
    /*--- Public constructors/destructor ---*/
public:
    DisplayControl(const Neuron* neuron);
    ~DisplayControl();

    /*--- Public Member functions ---*/

    osg::Node* getNode();

    types::RepresentationMode getRepresentationMode(
        const unsigned long frameNumber) const;

    types::RepresentationMode getRepresentationMode() const;

    void setRepresentationMode(types::RepresentationMode mode);

private:
    /*--- Private member attributes ---*/

    const Neuron* _neuron;

    PerFrameAttribute<types::RepresentationMode> _displayMode;

    osg::ref_ptr<osg::Group> _objects;

    /*--- Private member functions ---*/

    osg::Group* _getMorphologicalSkeleton();
};
}
}
}
#endif
