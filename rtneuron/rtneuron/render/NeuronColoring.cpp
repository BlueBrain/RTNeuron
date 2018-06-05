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

#include "NeuronColoring.h"

#include "ColorMap.h"

#include "../ColorMap.h"
#include "config/Globals.h"
#include "config/constants.h"
#include "data/Neuron.h"
#include "util/attributeMapHelpers.h"

#include <brain/neuron/morphology.h>
#include <brain/neuron/soma.h>

#include <osg/StateSet>
#include <osg/Uniform>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const float DEFAULT_MAX_DISTANCE_TO_SOMA = 1000.0;
const float DEFAULT_MAX_WIDTH = 80.0;
const float NUM_BY_WIDTH_SAMPLES = 100.0;
const float MINIMUM_CHANNEL_VALUE = 0.2;

/** Find the soma radius without triggerring the creation of a local
    morphology for the neuron */
float _getSomaRadius(const Neuron& neuron)
{
    auto morphology = neuron.getMorphology();
    if (morphology)
        return morphology->getSoma().getMeanRadius();
    return Globals::getDefaultSomaRadius(neuron.getMorphologyLabel());
}

osg::Vec4 _createRandomColor(const float minimumValue)
{
    const float range = 1 - minimumValue;
    return osg::Vec4(rand() / (float)RAND_MAX * range + minimumValue,
                     rand() / (float)RAND_MAX * range + minimumValue,
                     rand() / (float)RAND_MAX * range + minimumValue, 1.0);
}

bool _needsStaticColormap(const ColorScheme scheme)
{
    return scheme == ColorScheme::BY_DISTANCE_TO_SOMA ||
           scheme == ColorScheme::BY_WIDTH;
}

bool _updateColorMap(const ColorMapPtr& from, ColorMapPtr& to,
                     const std::string& prefix, const unsigned int unit)
{
    if (!from)
    {
        if (!to)
            return false;
        to.reset();
        return true;
    }

    if (!to)
    {
        to.reset(new rtneuron::ColorMap());
        to->getImpl().createUniforms(prefix, unit);
    }
    if (*to == *from)
        return false;

    *to = *from;
    return true;
}
}

const float NeuronColoring::DEFAULT_ATTENUATION = 2.0;

NeuronColoring::NeuronColoring(const AttributeMap& attributes)
    : _attenuation(DEFAULT_ATTENUATION)
{
    using namespace AttributeMapHelpers;
    _scheme = getEnum(attributes, "color_scheme", ColorScheme::SOLID);

    if (!getColor(attributes, "color", _primaryColor) &&
        !getColor(attributes, "primary_color", _primaryColor))
    {
        _primaryColor = Globals::getDefaultNeuronColor();
    }

    if (!getColor(attributes, "secondary_color", _secondaryColor))
        _secondaryColor = osg::Vec4(1, 1, 1, 1);
    AttributeMapPtr extra = attributes("extra", AttributeMapPtr());
    if (extra)
        _attenuation =
            (double)(*extra)("attenuation", double(DEFAULT_ATTENUATION));

    AttributeMapPtr colormaps = attributes("colormaps", AttributeMapPtr());
    if (colormaps)
        _updateColorMaps(*colormaps, true);
    else
        _staticColorMap = _createDefaultStaticColorMap();
}

bool NeuronColoring::operator==(const NeuronColoring& other) const
{
    if (_scheme != other._scheme)
        return false;

    if (_scheme != ColorScheme::BY_WIDTH)
        return (_primaryColor == other._primaryColor &&
                _secondaryColor == other._secondaryColor);
    else
        return (_primaryColor == other._primaryColor &&
                _secondaryColor == other._secondaryColor &&
                _attenuation == other._attenuation);
}

bool NeuronColoring::areVertexAttributesCompatible(
    const NeuronColoring& other) const
{
    if (_scheme != other._scheme)
        return false;

    switch (_scheme)
    {
    case ColorScheme::SOLID:
    case ColorScheme::BY_BRANCH_TYPE:
        return (_primaryColor == other._primaryColor &&
                _secondaryColor == other._secondaryColor);
    case ColorScheme::BY_DISTANCE_TO_SOMA:
        return true;
    case ColorScheme::BY_WIDTH:
        return true;
    default:;
        return false;
    }
}

void NeuronColoring::resolveRandomColor()
{
    if (_scheme != ColorScheme::RANDOM)
        return;

    _primaryColor = _createRandomColor(MINIMUM_CHANNEL_VALUE);
    _scheme = ColorScheme::SOLID;
}

osg::Vec4 NeuronColoring::getSomaBaseColor(const Neuron& neuron) const
{
    switch (_scheme)
    {
    case ColorScheme::SOLID:
    case ColorScheme::BY_BRANCH_TYPE:
        return _primaryColor;
    case ColorScheme::BY_DISTANCE_TO_SOMA:
        return _staticColorMap->getColor(0);
    case ColorScheme::BY_WIDTH:
        return _staticColorMap->getColor(_getSomaRadius(neuron) * 2.0);
    default:
        /* RANDOM must be treated by the Neuron object, otherwise the
           colors are different for each submodel */
        std::cerr << "Unreachable code" << std::endl;
        abort();
    }
}

size_t NeuronColoring::getCompartmentColorMapIndex() const
{
    if (_compartmentColorMap)
        return _compartmentColorMap->getImpl().getAtlasIndex();
    return std::numeric_limits<size_t>::max();
}

size_t NeuronColoring::getSpikeColorMapIndex() const
{
    if (_spikesColorMap)
        return _spikesColorMap->getImpl().getAtlasIndex();
    return std::numeric_limits<size_t>::max();
}

bool NeuronColoring::update(const AttributeMap& attributes)
{
    using namespace AttributeMapHelpers;

    bool updated = false;

    /* Attributes must be already validated */
    for (AttributeMap::const_iterator i = attributes.begin();
         i != attributes.end(); ++i)
    {
        if (i->first == "color_scheme")
        {
            ColorScheme scheme = getEnum<ColorScheme>(i->second);
            if (scheme != _scheme)
            {
                _scheme = scheme;
                updated = true;
            }
        }
        else if (i->first == "color" || i->first == "primary_color")
        {
            osg::Vec4 color;
            getColor(i->second, color);
            if (color != _primaryColor)
            {
                _primaryColor = color;
                updated = true;
            }
        }
        else if (i->first == "secondary_color")
        {
            osg::Vec4 color;
            getColor(i->second, color);
            if (color != _secondaryColor)
            {
                _secondaryColor = color;
                updated = true;
            }
        }
        else if (i->first == "extra")
        {
            AttributeMapPtr extra = attributes("extra");
            const double attenuation = (*extra)("attenuation", _attenuation);
            if ((float)attenuation != _attenuation)
            {
                _attenuation = attenuation;
                updated = true;
            }
        }
    }

    /* Colormaps can be consider once we know if the static colormap needs
       to be updated */
    AttributeMapPtr colorMaps = attributes("colormaps", AttributeMapPtr());
    if (colorMaps)
        updated |= _updateColorMaps(*colorMaps, updated);
    else if (updated || (_needsStaticColormap(_scheme) && !_staticColorMap))
        _updateColorMap(_createDefaultStaticColorMap(), _staticColorMap,
                        STATIC_COLOR_MAP_UNIFORM_PREFIX,
                        STATIC_COLOR_MAP_TEXTURE_NUMBER);

    return updated;
}

void NeuronColoring::applyColorMapState(osg::StateSet* stateSet) const
{
    static osg::ref_ptr<osg::Uniform> s_useColorMap =
        new osg::Uniform("useColorMap", true);

    ColorMap::removeAllAttributes(stateSet);

    if (_staticColorMap)
    {
        stateSet->addUniform(s_useColorMap);
        _staticColorMap->getImpl().addStateAttributes(stateSet);
    }
    else
        stateSet->removeUniform(s_useColorMap);

    if (_compartmentColorMap)
        _compartmentColorMap->getImpl().addStateAttributes(stateSet);
    if (_spikesColorMap)
        _spikesColorMap->getImpl().addStateAttributes(stateSet);
}

bool NeuronColoring::requiresStateSet() const
{
    return (bool)_spikesColorMap || (bool)_compartmentColorMap ||
           (bool)_staticColorMap;
}

std::vector<ColorMapPtr> NeuronColoring::getColorMapsForAtlas() const
{
    std::vector<ColorMapPtr> colorMaps;
    if (_compartmentColorMap)
        colorMaps.push_back(_compartmentColorMap);
    if (_spikesColorMap)
        colorMaps.push_back(_spikesColorMap);
    return colorMaps;
}

void NeuronColoring::assignDefaults(AttributeMap& attributes)
{
    using namespace AttributeMapHelpers;
    osg::Vec4 dummy;
    if (!getColor(attributes, "color", dummy) &&
        !getColor(attributes, "primary_color", dummy))
    {
        const osg::Vec4& color = Globals::getDefaultNeuronColor();
        attributes.set("color", color[0], color[1], color[2], color[3]);
        attributes.set("primary_color", color[0], color[1], color[2], color[3]);
    }
    ColorScheme dummy2;
    if (!getEnum(attributes, "color_scheme", dummy2))
    {
        /* The attribute is stored as integer for simplicity. Otherwise the
           enum type needs to be registered for AttributeMap::hash to work. */
        attributes.set("color_scheme", int(ColorScheme::SOLID));
    }
}

bool NeuronColoring::_updateColorMaps(const AttributeMap& colorMaps,
                                      const bool updateStaticColorMap)
{
    bool updated = false;

    ColorMapPtr compartmentsColorMap = colorMaps("compartments", ColorMapPtr());
    updated |= _updateColorMap(compartmentsColorMap, _compartmentColorMap,
                               COMPARTMENT_COLOR_MAP_UNIFORM_PREFIX,
                               COMPARTMENT_COLOR_MAP_TEXTURE_NUMBER);
    ColorMapPtr spikesColorMap = colorMaps("spikes", ColorMapPtr());
    updated |= _updateColorMap(spikesColorMap, _spikesColorMap,
                               SPIKE_COLOR_MAP_UNIFORM_PREFIX,
                               SPIKE_COLOR_MAP_TEXTURE_NUMBER);

    ColorMapPtr staticColorMap;
    switch (_scheme)
    {
    case ColorScheme::BY_DISTANCE_TO_SOMA:
        colorMaps.get("by_distance_to_soma", staticColorMap);
        break;
    case ColorScheme::BY_WIDTH:
        colorMaps.get("by_width", staticColorMap);
        break;
    default:;
    }

    if (!staticColorMap &&
        (updateStaticColorMap || _needsStaticColormap(_scheme)))
    {
        staticColorMap = _createDefaultStaticColorMap();
    }
    updated |= _updateColorMap(staticColorMap, _staticColorMap,
                               STATIC_COLOR_MAP_UNIFORM_PREFIX,
                               STATIC_COLOR_MAP_TEXTURE_NUMBER);

    return updated;
}

ColorMapPtr NeuronColoring::_createDefaultStaticColorMap() const
{
    ColorMapPtr colorMap;
    ColorMap::ColorPoints points;
    switch (_scheme)
    {
    case ColorScheme::BY_WIDTH:
    {
        ColorMap transferFunction;
        ColorMap::ColorPoints colors;
        colors[0] = _primaryColor;
        colors[0.8] = (_primaryColor + _secondaryColor) * 0.5;
        colors[1] = _secondaryColor;
        transferFunction.setPoints(colors);
        const float attenuationInverse = 1 / _attenuation;
        for (size_t i = 0; i != NUM_BY_WIDTH_SAMPLES; ++i)
        {
            const float width =
                DEFAULT_MAX_WIDTH * i / float(NUM_BY_WIDTH_SAMPLES - 1);
            points[width] = transferFunction.getColor(
                1 - std::exp(-width * attenuationInverse));
        }
        break;
    }
    case ColorScheme::BY_DISTANCE_TO_SOMA:
        points[0] = _primaryColor;
        points[DEFAULT_MAX_DISTANCE_TO_SOMA] = _secondaryColor;
        break;
    default:;
    }

    if (!points.empty())
    {
        colorMap.reset(new rtneuron::ColorMap());
        colorMap->setTextureSize(256);
        colorMap->setPoints(points);
        colorMap->getImpl().createUniforms(STATIC_COLOR_MAP_UNIFORM_PREFIX,
                                           STATIC_COLOR_MAP_TEXTURE_NUMBER);
    }
    return colorMap;
}
}
}
}
