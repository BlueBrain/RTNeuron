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

#ifndef RTNEURON_DETAILEDNEURONMODEL_H
#define RTNEURON_DETAILEDNEURONMODEL_H

#include "NeuronModel.h"
#include "models/ConstructionData.h"

#include <osg/Array>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace model
{
class SkeletonModel;
class NeuronSkeleton;
}

/** \brief Stores geometrical models of a neuron that can are culled by
    means of a NeuronSkeleton.

    Depending on the type of neuron model passed at construction time
    this object will hold a particular geometrical representation.
    Neuron skeleton share as much information as possible for all models
    that derive from the same neuron morphology.

    This class has some internal caches to speed up model creation.
    It's thread-safe to different models at the same time provided that their
    neuron gids are not the same.
*/

class DetailedNeuronModel : public NeuronModel
{
public:
    /*--- Public declarations ---*/

    friend class NeuronModel;
    friend class LODNeuronModel;

    class Drawable;

    /*--- Public Member fuctions ---*/

    bool isSomaOnlyModel() const final;

    void setMaximumVisibleBranchOrder(unsigned int order) final;

    void setupSimulationOffsetsAndDelays(const SimulationDataMapper& mapper,
                                         bool reuseCached) final;

    void addToScene(CircuitScene& scene) final;

    void applyStyle(const SceneStyle& style) final;

    void softClip(const NeuronModelClipping& operation) final;

    osg::BoundingBox getInitialBound() const final;

    void setColoring(const NeuronColoring& coloring) final;

    void highlight(bool on) final;

    void setRenderOrderHint(int order) final;

    osg::Drawable* getDrawable() final;

    /** Access the capsule skeleton of this model.
        The returned pointer can only be used as a reference. The pointer
        may be null. */
    Skeleton* getSkeleton();

protected:
    /*--- Protected constructor/destructors ---*/

    DetailedNeuronModel(NeuronLOD lod, model::NeuronParts parts,
                        const model::ConstructionData& data);

    ~DetailedNeuronModel();

    /*--- Protected member functions ---*/

    void setupAsSubModel(CircuitScene& scene) final;

    /* Hard clippping cannot be re-applied once this model has been added
       to any scene. */
    void clip(const Planes& planes, const Clipping type) final;

private:
    /*--- Private member attributes ---*/

    osg::ref_ptr<Drawable> _drawable;

    NeuronLOD _lod;

    std::shared_ptr<model::SkeletonModel> _model;
    std::shared_ptr<model::NeuronSkeleton> _skeleton;

    bool _simulationDataPending;
    bool _hasOffsetArray;
    /* We are only going to provide double buffering for the moment */
    osg::ref_ptr<osg::FloatArray> _simulationBuffers[2];
    unsigned char _backBuffer;

    /* Specifies if the model has been added to a LODNeuronModel or not.

       A model which is not a submodel is inserted under the same group
       as all other models of the same type, so the group is who contains
       the state set for the rendering style to use.
       When a model is a submodel, it needs is own StateSet, and it will
       be assigned when applyStyle is called. */
    bool _isSubModel;

    /*--- Private member functions ---*/

    /** Instantiates the osg::Drawable object for this model if needed
        distinguishing between circuits with unique morphologies or
        not. */
    void _instantiateDrawable(const CircuitScene& scene);

    bool _isDrawableEmpty() const;
};
}
}
}
#endif
