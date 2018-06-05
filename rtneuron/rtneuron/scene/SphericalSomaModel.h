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

#ifndef RTNEURON_SPHERICALSOMAMODEL_H
#define RTNEURON_SPHERICALSOMAMODEL_H

#include "scene/NeuronModel.h"
#include "scene/SphereSet.h"
#include "util/AutoExpandLFVector.h"

namespace osg
{
class Geometry;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! Class that handles the modelling of neuron somas as spheres
/*! All spherical somas are rendered using a single OSGGeometry object
  stored inside Scene as the "spheres" group. This geometry is rendered
  with a special shader that only requires a 4 coordinate vector to
  render an accurate sphere.
*/
class SphericalSomaModel : public NeuronModel
{
public:
    /*--- Public declarations ---*/

    friend class NeuronModel;

    /*--- Public member fuctions ---*/

    bool isSomaOnlyModel() const final { return true; }
    void setMaximumVisibleBranchOrder(unsigned int /* order */) final{};

    void setupSimulationOffsetsAndDelays(const SimulationDataMapper& mapper,
                                         bool reuseCached) final;

    /** This method is idempotent when run again on the same scene, i.e.
        doesn't change the scene or internal state.
        This is needed  to facilitate representation mode upgrades from soma
        only to something else. */
    void addToScene(CircuitScene& scene) final;

    void applyStyle(const SceneStyle& style) final;

    osg::BoundingBox getInitialBound() const final;

    void setColoring(const NeuronColoring& coloring) final;

    void highlight(bool on) final;

    osg::Drawable* getDrawable() final { return 0; }
    void softClip(const NeuronModelClipping& operation) final;

    /**
       Called from LODNeuronModelDrawable to make this sphere visible in
       the  scene-level sphere array and from to make it invisible.
     */
    void setVisibility(const uint32_t circuitSceneID, bool visible);

    void clearCircuitData(const CircuitScene& scene) final;

protected:
    /*--- Protected constructors/destructor ---*/

    SphericalSomaModel(const Neuron* neuron);

    /*--- Protected member functions ---*/

    void setupAsSubModel(CircuitScene& scene) final;

    void clip(const Planes&, const Clipping) final {}
private:
    /*--- Private declarations ---*/

    struct CircuitData
    {
        CircuitData();
        ~CircuitData();

        SphereSet::SubSetID _id;
        SphereSet* _spheres;
    };
    typedef AutoExpandLFVector<CircuitData> CircuitDataArray;
    typedef AutoExpandLFVector<CircuitData>::iterator CircuitDataIterator;

    /*--- Private member attributes ---*/

    CircuitDataArray _circuitData;
    float _radius;
    bool _clipped;
};
}
}
}
#endif
