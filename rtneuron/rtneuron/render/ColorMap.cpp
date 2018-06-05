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

#include "ColorMap.h"

#include <cmath>
#include <limits>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const size_t DEFAULT_COLORMAP_TEXTURE_SIZE = 512;

osg::ref_ptr<osg::Referenced> s_tag(new osg::Referenced());

// Helper functions -----------------------------------------------------------

void _storeColor(unsigned char* target, const osg::Vec4& color1,
                 const osg::Vec4& color2, const float alpha)
{
    target[0] =
        (unsigned char)(255 * (color1[0] * (1 - alpha) + color2[0] * alpha));
    target[1] =
        (unsigned char)(255 * (color1[1] * (1 - alpha) + color2[1] * alpha));
    target[2] =
        (unsigned char)(255 * (color1[2] * (1 - alpha) + color2[2] * alpha));
    target[3] =
        (unsigned char)(255 * (color1[3] * (1 - alpha) + color2[3] * alpha));
}

void _storeColor(unsigned char* target, const osg::Vec4& color)
{
    target[0] = (unsigned char)(255 * color[0]);
    target[1] = (unsigned char)(255 * color[1]);
    target[2] = (unsigned char)(255 * color[2]);
    target[3] = (unsigned char)(255 * color[3]);
}

void _removeUniforms(osg::StateSet& stateSet)
{
    osg::StateSet::UniformList& uniforms = stateSet.getUniformList();
    std::vector<std::string> toRemove;
    for (auto& uniform : uniforms)
    {
        if (uniform.second.first->getUserData() == s_tag)
            toRemove.push_back(uniform.first);
    }
    /* We dont remove directly from the UniformList because the removal
       code from osg::StateSet does additional things */
    for (const std::string& name : toRemove)
        stateSet.removeUniform(name);
}

void _removeTextureAttributes(osg::StateSet& stateSet)
{
    typedef std::pair<unsigned int, osg::StateAttribute*> UnitAttribute;
    std::vector<UnitAttribute> toRemove;

    osg::StateSet::TextureAttributeList& attributes =
        stateSet.getTextureAttributeList();
    for (unsigned int i = 0; i != attributes.size(); ++i)
    {
        for (auto& attribute : attributes[i])
        {
            if (attribute.second.first->getUserData() == s_tag)
                toRemove.push_back(UnitAttribute(i, attribute.second.first));
            else
                ++i;
        }
    }
    /* We dont remove directly from the UniformList because the removal
       code from osg::StateSet does additional things */
    for (const auto& unitAttr : toRemove)
        stateSet.removeTextureAttribute(unitAttr.first, unitAttr.second);
}

/**
   Computes the range in which the minimum of a point map maps to the middle of
   the first texel and the real maximum to middle of the last one when the
   values are converted into texture coordinates using the formula
   (a - min) / (max - min).

   The formulas are derived like this:
   1) m - m'           2) M  - m
      ------ = 1/(2t)     ------- = 1 - 1/t    where t = texture size
      M'- m'              M' - m'
   replacing M'-m' from 2) in 1) -> 2(t - 1)(m - m') = M - m.
   Then m' = m - (M - m)/(2(t - 1))

   M' is derived from a variation of 1): (M' - M)/(M' - m') = 1/(2t)

   The cases in which the number of points is <= 1 is treated specially because
   the range is degenerate.
*/
void _adjustedRangeFromPoints(const ColorMap::ColorPoints& points,
                              const size_t textureSize, float& min, float& max)
{
    if (points.empty())
    {
        min = max = 0;
    }
    else if (points.size() == 1)
    {
        min = max = points.begin()->first;
    }
    else
    {
        const float m = points.begin()->first;
        const float M = points.rbegin()->first;
        const float divisor = 2.f * (float(textureSize) - 1.f);
        min = m - (M - m) / divisor;
        max = M + (M - m) / divisor;
    }
}
}

ColorMap::ColorMap()
    : _textureNeedsUpdate(false)
    , _textureSize(DEFAULT_COLORMAP_TEXTURE_SIZE)
    , _atlasIndex(std::numeric_limits<size_t>::max())
{
    /* Default texture with just a black texel. The texture object is
       created so we can add something to a state set if requested. */
    _colorMap = new osg::Image();
    unsigned char* data = new unsigned char[4];
    memset(data, 0, sizeof(unsigned char) * 4);
    _colorMap->setImage(1, 1, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, data,
                        osg::Image::USE_NEW_DELETE);

    _texture = new osg::Texture1D();
    _texture->setImage(_colorMap);
    _texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    _texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    _texture->setUserData(s_tag);
}

ColorMap& ColorMap::operator=(const ColorMap& other)
{
    if (&other == this)
        return *this;

    set(other.getTextureSize(), other._originalPoints, other._realRange);
    return *this;
}

bool ColorMap::operator==(const ColorMap& other) const
{
    return (_textureSize == other._textureSize &&
            _realRange == other._realRange &&
            _originalPoints == other._originalPoints);
}

osg::Vec4 ColorMap::getColor(float value) const
{
    /* Finding two texels to inteporlate linearly between them */
    value = std::min(std::max(_realRange[0], value), _realRange[1]);

    const unsigned char* data = _colorMap->data();
    if (_realRange[0] == _realRange[1])
        return osg::Vec4(data[0], data[1], data[2], data[3]) / 255.0;

    float m, M;
    getAdjustedRange(m, M);

    /* Float texel position in texture coordinates, min maps to 0.5 and
       max maps to _textureSize - 0.5. */
    float position = ((value - m) / (M - m)) * _textureSize;
    /* Clamping position to [0.5, - tex - 0.5] to avoid rounding problems */
    position = std::min(_textureSize - 0.5f, std::max(0.5f, position));
    size_t t1 = size_t(std::floor(position));
    float alpha = position - std::floor(position);
    alpha = alpha >= 0.5 ? alpha - 0.5 : alpha + 0.5;
    if (alpha >= 0.5)
        --t1;
    const size_t channels = 4;
    const size_t t2 = std::min(t1 + 1, _textureSize - 1) * channels;
    t1 *= channels;
    const osg::Vec4 a =
        osg::Vec4(data[t1], data[t1 + 1], data[t1 + 2], data[t1 + 3]) / 255.0;
    const osg::Vec4 b =
        osg::Vec4(data[t2], data[t2 + 1], data[t2 + 2], data[t2 + 3]) / 255.0;
    return a * (1 - alpha) + b * alpha;
}

void ColorMap::setPoints(const ColorPoints& colorPoints)
{
    if (_points == colorPoints)
        return;

    _originalPoints = colorPoints;
    _points = colorPoints;
    if (colorPoints.size() != 0)
        _realRange =
            osg::Vec2(colorPoints.begin()->first, colorPoints.rbegin()->first);
    else
        _realRange = osg::Vec2(0, 0);
    _textureNeedsUpdate = true;
    _update();
}

void ColorMap::setRange(const float min, const float max)
{
    const osg::Vec2 range(min, max);
    if (_realRange == range)
        return;

    _realRange = range;
    _transformPoints(range);
    _update();
}

void ColorMap::getAdjustedRange(float& min, float& max) const
{
    if (_range)
    {
        osg::Vec2 range;
        _range->get(range);
        min = range[0];
        max = range[1];
    }
    else
    {
        _adjustedRangeFromPoints(_points, _textureSize, min, max);
    }
}

void ColorMap::setTextureSize(const size_t texels)
{
    _textureSize = std::max(2ul, texels);
    _update();
}

void ColorMap::set(unsigned int textureSize, const ColorPoints& points,
                   const osg::Vec2& range)
{
    bool updatePoints = false;
    if (_textureSize != textureSize)
    {
        _textureSize = textureSize;
        _textureNeedsUpdate = true;
    }
    if (_originalPoints != points)
    {
        _originalPoints = points;
        _textureNeedsUpdate = true;
        updatePoints = true;
    }
    if (_realRange != range)
    {
        _realRange = range;
        updatePoints = true;
    }
    if (updatePoints)
        _transformPoints(_realRange);

    /* Something has changed, we have to update the range uniform, the
       texture or both. */
    _update();
}

void ColorMap::fillTextureData(osg::Image& image) const
{
    unsigned char* data = image.data();
    if (_points.empty())
    {
        memset(data, 0, sizeof(unsigned char) * 4 * image.s());
        return;
    }
    if (_points.size() == 1)
    {
        for (int i = 0; i != image.s(); ++i)
            _storeColor(data + i * 4, _points.begin()->second);
        return;
    }

    const size_t textureSize = image.s();

    const float min = _points.begin()->first;
    const float max = _points.rbegin()->first;
    ColorPoints::const_iterator next = _points.begin();
    ColorPoints::const_iterator current = next++;
    for (size_t i = 0; i < textureSize; ++i)
    {
        float value = min + (max - min) * i / float(textureSize - 1);
        /* Clamping the value for safety (and simpler code). */
        value = std::min(max, std::max(min, value));
        if (value > next->first)
            current = next++;
        const float alpha =
            (value - current->first) / (next->first - current->first);
        _storeColor(data + i * 4, current->second, next->second, alpha);
    }
}

void ColorMap::createUniforms(const std::string& prefix, int unit)
{
    _samplerUniform =
        new osg::Uniform(prefix.empty() ? "colorMap"
                                        : (prefix + "ColorMap").c_str(),
                         unit);
    _samplerUniform->setUserData(s_tag);
    _range =
        new osg::Uniform(prefix.empty() ? "colorMapRange"
                                        : (prefix + "ColorMapRange").c_str(),
                         osg::Vec2f(0.0f, 0.0f));
    _range->setUserData(s_tag);
    _updateRange();
}

void ColorMap::addStateAttributes(osg::StateSet* stateSet)
{
    if (!_samplerUniform)
        /* Creating uniform variables with a default name and texture unit. */
        createUniforms("", 0);
    int unit;
    _samplerUniform->get(unit);
    stateSet->setTextureAttribute(unit, _texture);
    stateSet->addUniform(_samplerUniform);
    stateSet->addUniform(_range);
}

void ColorMap::removeStateAttributes(osg::StateSet* stateSet)
{
    int unit;
    _samplerUniform->get(unit);
    stateSet->removeTextureAttribute(unit, _texture);
    stateSet->removeUniform(_samplerUniform);
    stateSet->removeUniform(_range);
}

void ColorMap::removeAllAttributes(osg::StateSet* stateSet)
{
    _removeUniforms(*stateSet);
    _removeTextureAttributes(*stateSet);
}

void ColorMap::_update()
{
    _updateRange();

    if (_points.size() == 0)
    {
        unsigned char* data = new unsigned char[4];
        memset(data, 0, sizeof(unsigned char) * 4);
        _colorMap->setImage(1, 1, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, data,
                            osg::Image::USE_NEW_DELETE);
        _texture->dirtyTextureObject();
        return;
    }
    if (_points.size() == 1)
    {
        unsigned char* data = new unsigned char[4];
        _storeColor(data, _points.begin()->second);
        _colorMap->setImage(1, 1, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, data,
                            osg::Image::USE_NEW_DELETE);
        _texture->dirtyTextureObject();
        return;
    }

    unsigned char* data = _colorMap->data();
    if (_colorMap->s() != (int)_textureSize)
    {
        data = new unsigned char[_textureSize * 4];
        _colorMap->setImage(_textureSize, 1, 1, GL_RGBA, GL_RGBA,
                            GL_UNSIGNED_BYTE, data, osg::Image::USE_NEW_DELETE);
        _textureNeedsUpdate = true;
    }

    if (!_textureNeedsUpdate)
        /* If the points (or the texture size) haven't changed, there's no
           need to recompute the texture texels */
        return;

    fillTextureData(*_colorMap);

    _texture->dirtyTextureObject();
    _textureNeedsUpdate = false;
}

void ColorMap::_updateRange()
{
    if (!_range)
        return;

    osg::Vec2 range;
    _adjustedRangeFromPoints(_points, _textureSize, range[0], range[1]);
    _range->set(range);
}

void ColorMap::_transformPoints(const osg::Vec2& range)
{
    const float min = range[0];
    const float max = range[1];
    if (_originalPoints.size() == 0)
        return;
    if (_originalPoints.size() == 1)
    {
        ColorPoints newPoints;
        newPoints[(min + max) * 0.5] = _originalPoints.begin()->second;
        _points.swap(newPoints);
        return;
    }

    float oldMin = _originalPoints.begin()->first;
    float oldLength = _originalPoints.rbegin()->first - oldMin;
    float length = max - min;

    ColorPoints newPoints;
    for (ColorPoints::iterator p = _originalPoints.begin();
         p != _originalPoints.end(); ++p)
    {
        float value = ((p->first - oldMin) / oldLength) * length + min;
        newPoints.insert(newPoints.end(), std::make_pair(value, p->second));
    }
    /* _textureNeedsUpdate is not set to true because the relative positions
       between them haven't changed. */
    _points.swap(newPoints);
}
}
}
}
