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

#include "ColorMapAtlas.h"

#include "../ColorMap.h"
#include "config/constants.h"
#include "render/ColorMap.h"

#include <lunchbox/debug.h>

#include <boost/foreach.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
/* This limit comes from two facts:
   - The maximum guaranteed depth of GL_TEXTURE_2D_ARRAY is defined by
     GL_MAX_ARRAY_TEXTURE_LAYERS which in GL 3.3 is guaranteed to be at least
     256.
   - The length of the uniform array with the ranges must be preset at
     construction and can't be changed. Instead of a uniform, a texture buffer
     could be used, but the uniform is simpler and more convenient.

  The maximum number of colormaps is reduced by 1 so we can use the index 0
  to flag the default colormaps from the view.
*/
const size_t MAX_COLOR_MAPS_PER_ATLAS = 255;

template <typename T>
bool _isNull(const std::weak_ptr<T>& p)
{
    return (!p.owner_before(std::weak_ptr<T>{}) &&
            !std::weak_ptr<T>{}.owner_before(p));
}
}

ColorMapAtlas::ColorMapAtlas()
    : _texture(new osg::Texture2DArray())
    , _textureUniform(
          new osg::Uniform("colorMapAtlas", COLOR_MAP_ATLAS_TEXTURE_NUMBER))
    , _rangeUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC2,
                                     "colorMapAtlasRanges",
                                     MAX_COLOR_MAPS_PER_ATLAS))
{
}

void ColorMapAtlas::addColorMap(const ColorMapPtr& colorMap)
{
    bool dirty = false;
    bool present = false;
    /* Cleaning exprired pointer and detecting if this color map is already
       in the atlas. */
    for (auto& current : _colorMapArray)
    {
        /* expired is true also for default constructed weak ptrs */
        if (!_isNull(current) && current.expired())
        {
            current.reset();
            dirty = true;
        }
        else if (current.lock() == colorMap)
            present = true;
    }

    if (present)
    {
        if (dirty)
            _update();
        return;
    }

    /* Finding the first free slot and assigning it */
    size_t index = 0;
    for (; index != _colorMapArray.size(); ++index)
    {
        if (_colorMapArray[index].expired())
            break;
    }

    if (index >= MAX_COLOR_MAPS_PER_ATLAS)
        LBTHROW(std::runtime_error(
            "Maximum number of color maps per color map atlas exceeded"));

    if (index == _colorMapArray.size())
        _colorMapArray.resize(index + 1);

    _colorMapArray[index] = colorMap;
    colorMap->getImpl().setAtlasIndex(index);
    colorMap->dirty.connect(
        boost::bind(&ColorMapAtlas::_onColorMapDirty, this, index));

    _update();
}

void ColorMapAtlas::addStateAttributes(osg::StateSet* stateSet)
{
    stateSet->setTextureAttribute(COLOR_MAP_ATLAS_TEXTURE_NUMBER, _texture);
    stateSet->addUniform(_textureUniform);
    stateSet->addUniform(_rangeUniform);
}

void ColorMapAtlas::removeStateAttributes(osg::StateSet* stateSet)
{
    stateSet->removeTextureAttribute(COLOR_MAP_ATLAS_TEXTURE_NUMBER, _texture);
    stateSet->removeUniform(_textureUniform);
    stateSet->removeUniform(_rangeUniform);
}

void ColorMapAtlas::_update()
{
    size_t width = 0;
    for (auto& i : _colorMapArray)
    {
        ColorMapPtr colorMap = i.lock();
        if (colorMap)
            width = std::max(width, colorMap->getTextureSize());
    }

    /* We have to allocate space for as many layers as the highest color map
       index because it's not possible to properly reassign the atlas indices of
       colormaps after they are added to the atlas (e.g. dependent geometry
       objects can't be notified to update their uniforms/attributes) */
    _texture->setTextureSize(width, 1, _colorMapArray.size());

    osg::FloatArray& ranges = *_rangeUniform->getFloatArray();
    for (size_t i = 0; i != _colorMapArray.size(); ++i)
    {
        osg::Image* image = _texture->getImage(i);
        if (!image)
        {
            image = new osg::Image;
            _texture->setImage(i, image);
        }
        if (image->s() != int(width))
        {
            unsigned char* data = new unsigned char[width * 4];
            image->setImage(width, 1, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                            data, osg::Image::USE_NEW_DELETE);
        }

        const ColorMapPtr& colorMap = _colorMapArray[i].lock();
        if (colorMap)
        {
            _updateAtlasLayer(i, colorMap->getImpl());
            colorMap->getImpl().getAdjustedRange(ranges[i * 2],
                                                 ranges[i * 2 + 1]);
        }
    }
    _texture->dirtyTextureObject();
    _rangeUniform->dirty();
}

void ColorMapAtlas::_onColorMapDirty(const size_t index)
{
    assert(_texture);

    const ColorMapPtr& colorMap = _colorMapArray[index].lock();
    assert(colorMap);

    /* If the texture size of the colormap is greater than the current altas
       width we have to update all the layers, otherwise, just the layer
       for this colormap. */
    if (int(colorMap->getTextureSize()) > _texture->getTextureWidth())
    {
        _update();
        return;
    }

    /* Updating the layer and range uniform values */
    _updateAtlasLayer(index, colorMap->getImpl());

    osg::FloatArray& ranges = *_rangeUniform->getFloatArray();
    colorMap->getImpl().getAdjustedRange(ranges[index * 2],
                                         ranges[index * 2 + 1]);
    _rangeUniform->dirty();
}

void ColorMapAtlas::_updateAtlasLayer(const size_t layer,
                                      const ColorMap& colorMap)
{
    osg::Image* image = _texture->getImage(layer);
    colorMap.fillTextureData(*image);
    image->dirty();
}
}
}
}
