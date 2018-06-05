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

#ifndef RTNEURON_SKELETONMODEL_H
#define RTNEURON_SKELETONMODEL_H

#include "NeuronSkeleton.h"

#include "render/DetailedNeuronModelDrawable.h"
#include "render/NeuronColoring.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
class DrawElementsPortions;
typedef std::vector<osg::Vec4d> Planes;

namespace model
{
/** Base class for the different models internally managed by
    DetailedNeuronModel
*/
class SkeletonModel
{
public:
    /*--- Public declarations ---*/

    typedef DetailedNeuronModel::Drawable Drawable;

    /*--- Public Member variables ---*/

    NeuronLOD _lod;

    osg::ref_ptr<osg::Array> _vertices;
    uint16_tsPtr _sections;
    floatsPtr _positions;

    size_t _length;

    /* Used for static colors that can be shared between models. This
       cache is intended to be useful when the color scheme and base color is
       always the same for a given morphology.
       For unique morphologies it's useless. */
    NeuronColoring _coloring;
    osg::ref_ptr<osg::Array> _colors;

    /*--- Public constructors/destructor ---*/

    virtual ~SkeletonModel();

    /*--- Public member functions ---*/

    virtual Drawable* instantiate(const SkeletonPtr& skeleton,
                                  const CircuitSceneAttributes& sceneAttr) = 0;

    virtual void clip(const Planes& planes) = 0;

    virtual Skeleton::PortionRanges postProcess(NeuronSkeleton& skeleton) = 0;

    /** Returns the initial bounding box for this model.

        Scene clipping doesn't affect the bounding box on purpose because
        otherwise different rendering nodes will obtain different results
        for culling and LOD selection
    */
    virtual const osg::BoundingBox& getOrCreateBound(const Neuron& neuron);

    /* If this is not a LOD submodel, the StateSets with the rendering style
       are in the ancestor nodes so this must return 0. */
    virtual osg::StateSet* getModelStateSet(const bool subModel,
                                            const SceneStyle& style) const = 0;

    /** Updates or creates a color array for a given coloring scheme.

        Returns the cached array if the requested coloring scheme is valid
        for it. Otherwise, creates a new array. If the cached array can
        be reused to speed up the creation of the new array it will be used.

        This method must be used as the only source of color arrays to be
        set to drawables returned by instantiate(). This guarantees
        thread-safety when setColorArray is called concurrently on drawables
        that are implicity sharing the same VBO.
        The only safe alternative to this method is to assign a VBO to the
        color array before setting it to the drawable.
    */
    osg::ref_ptr<osg::Array> getOrCreateColorArray(
        const NeuronColoring& coloring,
        const brain::neuron::Morphology& morphology);

protected:
    /*--- Protected member variables ---*/

    OpenThreads::Mutex _mutex;
    osg::BoundingBox _bbox;
};
}
}
}
}
#endif
