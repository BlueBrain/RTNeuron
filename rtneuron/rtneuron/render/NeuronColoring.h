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

#ifndef RTNEURON_NEURONCOLORING_H
#define RTNEURON_NEURONCOLORING_H

#include "../coreTypes.h"
#include "../types.h"

#include <osg/Vec4>

namespace osg
{
class StateSet;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
/** See types::ColorScheme for details */
class NeuronColoring
{
public:
    static const float DEFAULT_ATTENUATION;

    /*--- Public constructors/destructor ---*/
    NeuronColoring(const ColorScheme scheme = ColorScheme::SOLID,
                   const osg::Vec4& primary = osg::Vec4(1, 1, 1, 1),
                   const osg::Vec4& secondary = osg::Vec4(1, 1, 1, 1))
        : _scheme(scheme)
        , _primaryColor(primary)
        , _secondaryColor(secondary)
        , _attenuation(DEFAULT_ATTENUATION)
    {
    }

    NeuronColoring(const AttributeMap& attributeMap);

    /*--- Public member/functions ---*/

    bool operator==(const NeuronColoring& other) const;

    /**
       @return true if for a given model model, the per vertex attribute arrays
       necessary for this coloring are equal to those required by the given
       coloring.
    */
    bool areVertexAttributesCompatible(const NeuronColoring& other) const;

    ColorScheme getScheme() const { return _scheme; }
    /**
       @return Primary color for this neuron coloring.

       For the different meanings of the primary color depending on the
       color scheme @see Scene::addNeurons()
    */
    const osg::Vec4& getPrimaryColor() const { return _primaryColor; }
    /**
       @return Primary color for this neuron coloring.

       For the different meanings of the secondary color depending on the
       color scheme @see Scene::addNeurons()
    */
    const osg::Vec4& getSecondaryColor() const { return _secondaryColor; }
    /**
       Extra attribute used only for BY_WIDTH. It's the constant
       a in the equation 1 - e^(-width * 1/a)
    */
    float getAttenuation() const { return _attenuation; }
    /**
      @return The color to be used for rendering the soma of the given neuron
              when no simulation data is applied.
              This color may also modulate how simulation data is displayed.
    */
    osg::Vec4 getSomaBaseColor(const Neuron& neuron) const;

    /**
       @return Atlas index to use to access the compartment color map
       in the GLSL shaders or max size_t if the atlas musn't be used.
    */
    size_t getCompartmentColorMapIndex() const;

    /**
       @return Atlas index to use to access the spike color map
       in the GLSL shaders or max size_t if the atlas musn't be used.
    */
    size_t getSpikeColorMapIndex() const;

    /**
       If scheme is RANDOM, assigns a random color to the primary color
       and switches the scheme to SOLID.
       No operation otherwise
    */
    void resolveRandomColor();

    /**
       Update the member attributes with the information provided from
       the given attribute map.

       @return true if any member attribute has changed.
    */
    bool update(const AttributeMap& attributes);

    /**
       Updates the color map state attributes in a state set.

       That implies clearing up unnecessary attributes that may be present
       in the state set (assuming they where added from a previous call.
    */
    void applyColorMapState(osg::StateSet* stateSet) const;

    /**
       @return true if this coloring requires a state set to stores uniform
       variables (e.g. to apply colormaps)
    */
    bool requiresStateSet() const;

    /**
       @return The list of colormaps stored that need to be added to the scene
               color map atlas. This list corresponds to simulation related
               colormaps set through the attribute API on the NeuronObject
               associated to this coloring.
    */
    std::vector<ColorMapPtr> getColorMapsForAtlas() const;

    /**
       Assigns default values to basic attributes if they don't have any yet.

       The attributes that may be assigned are color (and primary_color) and
       color_scheme.
    */
    static void assignDefaults(AttributeMap& attributes);

private:
    /*--- Private member variables ---*/

    ColorScheme _scheme;
    osg::Vec4 _primaryColor;   /* Dendrite and soma color in
                                  BY_BRANCH_TYPE */
    osg::Vec4 _secondaryColor; /* Axon color in BY_BRANCH_TYPE */

    /* Extra attribute used only for BY_WIDTH. It's the constant
       a in the equation 1 - e^(-width * 1/a) */
    float _attenuation;

    ColorMapPtr _compartmentColorMap;
    ColorMapPtr _spikesColorMap;
    ColorMapPtr _staticColorMap;

    /*--- Private member functions ---*/
    bool _updateColorMaps(const AttributeMap& colormaps,
                          bool updateStaticColorMap);
    ColorMapPtr _createDefaultStaticColorMap() const;
};
}
}
}
#endif
