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

#ifndef RTNEURON_RENDER_COLORMAP_H
#define RTNEURON_RENDER_COLORMAP_H

#include <osg/StateSet>
#include <osg/Texture1D>

namespace bbp
{
namespace rtneuron
{
namespace core
{
class ColorMap
{
public:
    typedef std::map<float, osg::Vec4f> ColorPoints;

    /*--- Public constructors/destructor ---*/

    ColorMap();

    /*--- Public member functions ---*/

    ColorMap& operator=(const ColorMap& other);

    bool operator==(const ColorMap& other) const;

    osg::Vec4 getColor(float value) const;

    void setPoints(const ColorPoints& points);

    const ColorPoints& getPoints() const { return _points; }
    /** Return the non range-adjusted point map that was set by setPoints()
        or set() */
    const ColorPoints& getOriginalPoints() const { return _originalPoints; }
    void setRange(float min, float max);

    void getRange(float& min, float& max) const
    {
        min = _realRange[0];
        max = _realRange[1];
    }

    /** Returns the range [m', M'] to be used in texture lookups to make sure
        that map(m) maps to the middle of the first texel (1/2t) and
        map(M) maps to the middle of the last one (1 - 1/2t).
        Where:
        * and [m, M] is the range returned by getRange()
        * t = texture size
        * map(m) = m-m' / (M'-m')
    */
    void getAdjustedRange(float& min, float& max) const;

    void setTextureSize(size_t texels);

    size_t getTextureSize() const { return _textureSize; }
    /**
       Sets the colormap parameters.
       @param textureSize
       @param points The set of points. This point set is subject to
                     adjustment by the range parameters.
       @param range The actual range to be used for the points.
    */
    void set(unsigned int textureSize, const ColorPoints& points,
             const osg::Vec2& range);

    /**
       @return The atlas index of this color map or
       std::numeric_limits<size_t>::max() if not defined
    */
    size_t getAtlasIndex() const { return _atlasIndex; }
    /**
       Creates the uniform variables for the shaders

       New objects are created because uniform variable names cannot be
       changed. Make sure old uniform variables are removed from the
       state sets using them to avoid keeping unnecessary uniform variables
       in the scenegraph.
    */
    void createUniforms(const std::string& prefix, int unit);

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

    /**
       Removes all colormap related attributes from a stateset.
    */
    static void removeAllAttributes(osg::StateSet* stateSet);

private:
    /*--- Private member functions accessed by ColorMapAtlas ---*/

    friend class ColorMapAtlas;
    /**
       Called from ColorMapAtlas::addColorMap to assign the texture index
       of the atlas where this color map is copied.
     */
    void setAtlasIndex(const size_t index) { _atlasIndex = index; }
    /**
       Fill an image buffer with the color map texture data.
    */
    void fillTextureData(osg::Image& image) const;

    /*--- Private member attributes ---*/

    ColorPoints _points;
    ColorPoints _originalPoints;
    bool _textureNeedsUpdate;

    osg::Image* _colorMap;
    osg::ref_ptr<osg::Texture1D> _texture;
    osg::ref_ptr<osg::Uniform> _samplerUniform;
    size_t _textureSize;

    osg::Vec2 _realRange;
    osg::ref_ptr<osg::Uniform> _range;

    std::string _uniformPrefix;

    size_t _atlasIndex;

    /*--- Private member functions ---*/

    void _update();
    void _updateRange();
    void _transformPoints(const osg::Vec2& range);
};
}
}
}
#endif
