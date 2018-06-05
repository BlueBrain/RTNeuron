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
#ifndef RTNEURON_CONFIG_GLOBALS_H
#define RTNEURON_CONFIG_GLOBALS_H

#include "types.h"

#include <osg/Vec4> // Can't be a forward reference

#include <map>
#include <string>

namespace osgText
{
class Font;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   Global properties of the application.

   \todo Values are supposed to be set only and read by the main thread.
   Review if that's the case or some locking is needed.
*/
class Globals
{
public:
    /*--- Synapse options ---*/

    /** @return (0.5, 0.2, 1.0, 1.0) until changed. */
    static const osg::Vec4& getDefaultEfferentSynapseColor();
    static void setDefaultEfferentSynapseColor(const osg::Vec4& color);

    /** @return (1.0, 0.6, 0.2, 1.0) until changed. */
    static const osg::Vec4& getDefaultAfferentSynapseColor();
    static void setDefaultAfferentSynapseColor(const osg::Vec4& color);

    /** Default value is 1 micron */
    static float getDefaultSynapseRadius();
    static void setDefaultSynapseRadius(const float radius);

    /*--- Neurons options ---*/

    /** @return (0, 0.6, 0.6, 1.0) until changed. */
    static osg::Vec4 getDefaultNeuronColor();

    /** Grabs the default neuron color from an attribute map.
        Search for attribute 'neuron_color' and:
        - If 'random', generates a random color
        - If 'allrandom', generates a random color everytime
        getDefaultNeuronColor is called.
        - If a 3xfloat or 4xfloat tuple can be parsed, it assigns that color
        (alpha channel equals 1 for the 3xfloat tuple).
     */
    static void setDefaultNeuronColor(const AttributeMap& attributes);

    static void setDefaultNeuronColor(const osg::Vec4& color);

    static void setDefaultSomaRadii(
        const std::map<std::string, float>& nameRadiusTable);

    static void setDefaultSomaRadii(const AttributeMap& nameRadiusTable);

    static float getDefaultSomaRadius(const std::string& morphology = "");

    static void setDefaultSomaRadius(const float radius);

    static bool isDefaultNeuronColorRandom();

    static osgText::Font* getDefaultFont();

    /*--- Rendering control options ---*/

    static bool doContinuousRendering();

    static void setProfilingOptions(const AttributeMap& attributes);

    static bool isProfilingEnabled();

    static bool isCompositingEnabled();

    static bool isReadbackEnabled();

    static bool profileChannels();

    static bool profileDataLoading();

    static bool areGLDrawCallsEnabled();

    static const std::string& getProfileLogfile();
};
}
}
}
#endif
