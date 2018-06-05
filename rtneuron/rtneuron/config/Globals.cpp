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

#include "Globals.h"
#include "constants.h"
#include "paths.h"

#include "../AttributeMap.h"
#include "util/attributeMapHelpers.h"

#include <osgText/Font>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const char* _getFontPath()
{
    char* path = ::getenv("RTNEURON_FONT_PATH");
    if (path)
        return path;
    return "";
}

const std::string s_fontPath(_getFontPath());

osg::Vec4 defaultAfferentSynapseColor(1, 0.6, 0.2, 1);
osg::Vec4 defaultEfferentSynapseColor(0.5, 0.2, 1.0, 1);
double defaultSynapseRadius = 1; // microns

osg::Vec4 defaultNeuronColor(0.8, 0.85, 0.9, 1.0);

/* Lenght units are in microns */
typedef std::map<std::string, float> SomaRadiusMap;
SomaRadiusMap defaultSomaRadii;
float defaultSomaRadius = DEFAULT_SOMA_RADIUS;

bool profilingEnabled = false;
bool dataLoadingProfilingEnabled = false;
bool channelProfilingEnabled = false;
std::string profilingLogfile = "statistics.txt";
bool compositingEnabled = true;
bool readbackEnabled = true;
bool glCallsEnabled = true;
}

const osg::Vec4& Globals::getDefaultEfferentSynapseColor()
{
    return defaultEfferentSynapseColor;
}

void Globals::setDefaultEfferentSynapseColor(const osg::Vec4& color)
{
    defaultEfferentSynapseColor = color;
}

const osg::Vec4& Globals::getDefaultAfferentSynapseColor()
{
    return defaultAfferentSynapseColor;
}

void Globals::setDefaultAfferentSynapseColor(const osg::Vec4& color)
{
    defaultAfferentSynapseColor = color;
}

float Globals::getDefaultSynapseRadius()
{
    return defaultSynapseRadius;
}

void Globals::setDefaultSynapseRadius(const float radius)
{
    defaultSynapseRadius = radius;
}

void Globals::setDefaultNeuronColor(const AttributeMap& attributes)
{
    try
    {
        std::string value = attributes("neuron_color");
        if (value == "random")
            defaultNeuronColor = osg::Vec4(std::rand() / double(RAND_MAX),
                                           std::rand() / double(RAND_MAX),
                                           std::rand() / double(RAND_MAX), 1);
        else if (value == "all-random")
            defaultNeuronColor = osg::Vec4(0, 0, 0, -1);
    }
    catch (...)
    {
        osg::Vec4 color;
        if (AttributeMapHelpers::getColor(attributes, "neuron_color", color))
            defaultNeuronColor = color;
    }
}

void Globals::setDefaultNeuronColor(const osg::Vec4& color)
{
    defaultNeuronColor = color;
}

osg::Vec4 Globals::getDefaultNeuronColor()
{
    if (defaultNeuronColor[3] == -1)
        return osg::Vec4(std::rand() / double(RAND_MAX),
                         std::rand() / double(RAND_MAX),
                         std::rand() / double(RAND_MAX), 1);
    else
        return defaultNeuronColor;
}

bool Globals::isDefaultNeuronColorRandom()
{
    return defaultNeuronColor[3] == -1;
}

void Globals::setDefaultSomaRadii(const SomaRadiusMap& nameRadiusTable)
{
    defaultSomaRadii = nameRadiusTable;
}

void Globals::setDefaultSomaRadii(const AttributeMap& nameRadiusTable)
{
    defaultSomaRadii.clear();
    for (const auto& nameRadius : nameRadiusTable)
        defaultSomaRadii[nameRadius.first] = (double)nameRadius.second;
}

float Globals::getDefaultSomaRadius(const std::string& morphology)
{
    if (morphology.empty())
        return defaultSomaRadius;

    SomaRadiusMap::const_iterator i = defaultSomaRadii.find(morphology);
    if (i != defaultSomaRadii.end())
        return i->second;

    return defaultSomaRadius;
}

void Globals::setDefaultSomaRadius(const float radius)
{
    defaultSomaRadius = radius;
}

bool Globals::doContinuousRendering()
{
    return profilingEnabled;
}

void Globals::setProfilingOptions(const AttributeMap& attributes)
{
    profilingEnabled = attributes("enable");

    /** Variables not modified if attributes not present. */
    attributes.get("logfile", profilingLogfile);
    attributes.get("compositing", compositingEnabled);
    attributes.get("readback", readbackEnabled);
    attributes.get("channels", channelProfilingEnabled);
    attributes.get("gldraw", glCallsEnabled);
    attributes.get("dataloading", dataLoadingProfilingEnabled);

    if (!readbackEnabled)
        compositingEnabled = false;
}

bool Globals::isProfilingEnabled()
{
    return profilingEnabled;
}

bool Globals::profileChannels()
{
    return channelProfilingEnabled;
}

bool Globals::profileDataLoading()
{
    return dataLoadingProfilingEnabled;
}

bool Globals::isCompositingEnabled()
{
    return compositingEnabled;
}

bool Globals::isReadbackEnabled()
{
    return readbackEnabled;
}

const std::string& Globals::getProfileLogfile()
{
    return profilingLogfile;
}

bool Globals::areGLDrawCallsEnabled()
{
    return glCallsEnabled;
}
}
}
}
