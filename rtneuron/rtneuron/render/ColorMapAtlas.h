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
#ifndef RTNEURON_COLORMAPATLAS_H
#define RTNEURON_COLORMAPATLAS_H

#include "types.h"

#include <osg/StateSet>
#include <osg/Texture2DArray>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class ColorMap;

class ColorMapAtlas
{
public:
    /*--- Public constructors/destructor ---*/

    ColorMapAtlas();

    /*--- Public member functions ---*/

    /**
       Adds a color map to the altas.

       The color map will be assigned an atlas index. If the color map is
       destroyed, the atlas will free its slot (this operation is deferred
       until a new atlas is added).
       The atlas will track changes in the colormap and update the internal
       texture and range variables accordingly.
       If the color map is already part of the atlas this operation does
       nothing.
    */
    void addColorMap(const ColorMapPtr& colorMap);

    /**
       Adds the colormap texture and the GLSL uniforms required for shading
       to the given StateSet
    */
    void addStateAttributes(osg::StateSet* stateSet);

    /**
       Removes the state attributes associated to this color map from a
       state set.
    */
    void removeStateAttributes(osg::StateSet* stateSet);

private:
    /*--- Private member variables ---*/

    typedef std::weak_ptr<rtneuron::ColorMap> ColorMapWeakPtr;
    std::vector<ColorMapWeakPtr> _colorMapArray;

    osg::ref_ptr<osg::Texture2DArray> _texture;
    osg::ref_ptr<osg::Uniform> _textureUniform;
    osg::ref_ptr<osg::Uniform> _rangeUniform;

    /*--- Private member functions ---*/

    void _update();

    void _onColorMapDirty(size_t index);
    void _updateAtlasLayer(size_t layer, const core::ColorMap& colorMap);
};
}
}
}
#endif
