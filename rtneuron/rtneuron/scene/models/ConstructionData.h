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

#ifndef RTNEURON_CONSTRUCTIONDATA_H
#define RTNEURON_CONSTRUCTIONDATA_H

#include "coreTypes.h"
#include "types.h"

#include "NeuronParts.h"

#include <boost/shared_ptr.hpp>

namespace brain
{
namespace neuron
{
class Morphology;
}
}

namespace bbp
{
namespace rtneuron
{
/* This namespace is closed to workaround an issue with ccpcheck 1.63 */
namespace core
{
class CircuitScene;
class CircuitSceneAttributes;
class Neuron;
}

namespace core
{
namespace model
{
class ConstructionData
{
public:
    /*--- Public member variables ---*/

    const Neuron& neuron;
    const brain::neuron::Morphology& morphology;
    /* This pointer may be null if the mesh is not available and it wasn't
       loaded by the constructor. */
    const NeuronMesh* const mesh;

    const CircuitScene& scene;
    const CircuitSceneAttributes& sceneAttr; /* Included just to reduce
                                                header dependencies. */
    /*--- Public constructors ---*/

    /**
       @param neuron The neuron object from which the data members will
              be extracted.
       @param scene
     */
    ConstructionData(const Neuron& neuron, const CircuitScene& scene);
};
}
}
}
}
#endif
