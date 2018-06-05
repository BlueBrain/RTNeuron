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

#ifndef RTNEURON_LODNEURONMODEL_H
#define RTNEURON_LODNEURONMODEL_H

#include "scene/NeuronModel.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! NeuronModel object that provides simple LOD capabilities.
/** An object of this class can manage several NeuronModel objects. Its inner
    Drawable class is in charge of selecting the proper object to render
    during the scenegraph traversal of the cull visitor
    \sa LODNeuronModel::Drawable. */
class LODNeuronModel : public NeuronModel
{
public:
    /*--- Public declarations ---*/
    class Drawable;

    /*--- Public constructor/destructor ---*/

    LODNeuronModel(const Neuron* neuron);

    /*--- Public member functions ---*/

    bool isSomaOnlyModel() const final { return false; }
    void setMaximumVisibleBranchOrder(unsigned int order) final;

    void setupSimulationOffsetsAndDelays(const SimulationDataMapper& mapper,
                                         bool reuseCached) final;

    void addToScene(CircuitScene& scene) final;

    void applyStyle(const SceneStyle& style) final;

    osg::BoundingBox getInitialBound() const final;

    void setColoring(const NeuronColoring& coloring) final;

    void highlight(bool on) final;

    void setRenderOrderHint(int order) final;

    osg::Drawable* getDrawable() final;

    void softClip(const NeuronModelClipping& operation) final;

    void addSubModel(NeuronModel* model, float min, float max);

    void accept(CollectSkeletonsVisitor& visitor) const;

protected:
    /*--- Protected construtors/destructor ---*/

    virtual ~LODNeuronModel();

    /*--- Protected member functions ---*/

    void setupAsSubModel(CircuitScene& scene) final;

    void clip(const Planes& planes, const Clipping type) final;

private:
    /*--- Private declarations ---*/

    struct SubModel
    {
        SubModel(const osg::ref_ptr<NeuronModel>& model_, float min_,
                 float max_)
            : model(model_)
            , min(min_)
            , max(max_)
        {
        }
        osg::ref_ptr<NeuronModel> model;
        float min;
        float max;
    };
    typedef std::vector<SubModel> SubModelList;

    /*--- Private member variables ---*/

    osg::ref_ptr<Drawable> _drawable;
    SubModelList _models;

    /*--- Private member functions ---*/
    template <typename T>
    void _doOncePerSkeleton(const T& functor);
};
}
}
}
#endif
