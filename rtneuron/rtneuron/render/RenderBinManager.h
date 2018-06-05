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
#ifndef RTNEURON_RENDERBINMANAGER_H
#define RTNEURON_RENDERBINMANAGER_H

#include "types.h"

#include <eq/fabric/pixelViewport.h>

#include <memory>

namespace bbp
{
namespace osgTransparency
{
class BaseRenderBin;
}

namespace rtneuron
{
namespace core
{
class RenderBinManager
{
public:
    /** Chooses and configures the render bin to use for alpha blending.

        @param attributes the attribute map passed to the rtneuron::Scene
        constructor
    */
    RenderBinManager(const AttributeMap& attributes);

    ~RenderBinManager();

    /** Checks the coherency and completeness of the attribute map to
        configure an alpha blending algorithm and throws in case of error. */
    static void validateAttributes(const AttributeMap& attributes);

    /** @return the render bin to use for different alpha blending styles */
    osgTransparency::BaseRenderBin* getAlphaBlendedRenderBin() const;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
}
}
}
#endif
