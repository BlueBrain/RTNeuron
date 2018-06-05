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
#ifndef RTNEURON_SCENESTYLE_H
#define RTNEURON_SCENESTYLE_H

#include "coreTypes.h"
#include "types.h"

/* Forward references */
namespace osg
{
class StateSet;
class Node;
}

namespace bbp
{
namespace rtneuron
{
namespace alphablend
{
class BaseRenderBin;
}

namespace core
{
/**
   This class implements all the logic to setup the rendering style
   parameters of a scene.

   The style attributes are common for all views of this scene.

   Valid attributes are:
   - *accurate_headlight*:
     See Scene
   - clod (bool):
     Enable continuous level of detail for combined tublets/pseudocylinder
     models.
   - *em_shading*:
     See Scene
   - *faster_pseudocylinder_normals* (bool):
     Faster and lower quality rendering for pseudo-cylinders.
   - *inflatable_neurons*:
     See Scene
   - *smooth_tubelets* (bool):
     Use a better algorithm for normal computation in the tubelets model
   - *show_spikes* (bool):
     Enables shader code paths to display spikes in somas and axons.

   This class is thread safe.
*/
class SceneStyle
{
public:
    /*--- Public constructors/destructor ---*/

    /**
       @param attributes The attributes used to create the visual appearance
       of the scene. For the moment these attributes are static.
    */

    SceneStyle(const AttributeMap& attributes);

    ~SceneStyle();

    SceneStyle(const SceneStyle&) = delete;
    SceneStyle& operator=(const SceneStyle&) = delete;

    /*--- Public member functions ---*/

    const AttributeMap& getAttributes() const;

    /**
       Applies the static rendering style attributes to an arbitrary model.
    */
    void processModel(osg::Node* node);

    /**
       Returns the state set to use for a given type of objects.

       The state set returned must not be modified (it cannot be const
       because OSG won't accept it inside a scenegraph, but it should be
       treated as const).
    */
    enum StateType
    {
        /* General styles */
        MESH,
        FLAT_MESH,
        LINES,
        SPHERES,
        POINTS,
        /* Neuron styles */
        NEURON_MESH,
        NEURON_TUBELETS,
        NEURON_PSEUDO_CYLINDERS,
        NEURON_SPHERICAL_SOMA,

        STATE_TYPE_COUNT
    };
    osg::StateSet* getStateSet(const StateType type) const;

    /**
       Returns the state set to use for a given type of objects with extra
       parameters.

       The state set returned must not be modified (it cannot be const
       because OSG won't accept it inside a scenegraph, but it should be
       treated as const).

       @param type
       @param extra Extra attributes to override the internal attributes or
                    for options of specific styles.
    */
    osg::StateSet* getStateSet(const StateType type,
                               const AttributeMap& extra) const;

    /**
       Set the color map atlas to be used by rendering state sets generated
       by this class.
    */
    void setColorMapAtlas(const ColorMapAtlasPtr& colorMaps);

    RenderBinManager& getRenderBinManager();

private:
    /*--- Private member variables */

    class Impl;
    std::unique_ptr<Impl> _impl;
};
}
}
}
#endif
