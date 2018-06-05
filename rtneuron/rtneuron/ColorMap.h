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

#ifndef RTNEURON_API_COLORMAP_H
#define RTNEURON_API_COLORMAP_H

#include <osg/Vec4f>

#include <boost/signals2.hpp>

namespace boost
{
namespace serialization
{
class access;
}
}

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
class ColorMap;
}

/**
   Class to create color maps and store them as 1D texture of arbitrary size.

   Objects of this class can be used as attributes in AttributeMap both in
   Python and C++.

   This class handles the OSG objects used to apply the color map to the
   scenegraph.
*/
class ColorMap : private boost::noncopyable
{
public:
    /** @name C++ public interface wrapped in Python */
    ///@{

    typedef std::map<float, osg::Vec4f> ColorPoints;

    /*--- Public constructors/destructor ---*/

    /**
       Default constructor

       Creates a color map with the no control points.
       If used at rendering, this color map always resolves to (0, 0, 0, 0).
    */
    ColorMap();

    ~ColorMap();

    /*--- Public member functions ---*/

    /**
       Creates the internal look up table using the map of (value, color)
       points given.

       \if pybind
       \param colorPoints The control points dictionary. The keys must be floats
                          and the items are 4-float tuples (RGBA).
       \else
       \param colorPoints The control points.
       \endif
                          If any channel is outside the range [0, 1] the
                          underlying color map will be undefined.

       \ifnot pybind
       The dirty signal is emitted.
       \endif
    */
    void setPoints(const ColorPoints& colorPoints);

    /**
       The control points of this color map.
     */
    const ColorPoints& getPoints() const;

    /**
       Changes the colormap range adjusting the point values.

       The value of the points are ajusted to the new range and the dirty
       signal is emitted.
     */
    void setRange(float min, float max);

/**
   Return the color map range.
 */
#ifdef DOXGEN_TO_BREATHE
    tuple getRange() const;
#else
    void getRange(float& min, float& max) const;
#endif

    /**
       Returns the color for the given value.

       @param value Clamped to current range before sampling the internal
       texture.
       @return The color at the given value using linear interpolation of the
       control points.
     */
    osg::Vec4f getColor(float value) const;

    /**
       Load a color map from the file with the given name.

       Throws if an error occurs.
    */
    void load(const std::string& fileName);

    /**
       Save a color map to a file with the given name.

       Throws if an error occurs.
    */
    void save(const std::string& fileName);

    /**
       Set the resolution of the internal texture used for the
       colormap (measured in texels).

       The minimum texture size is bounded to 2 texels.
    */
    void setTextureSize(size_t texels);

    size_t getTextureSize() const;

    ///@}

    /*--- Public C++ only member functions ---*/

    /** @name C++ only public interface */
    ///@{

    /*--- Public declarations ---*/
    typedef void DirtySignalSignature();
    typedef boost::signals2::signal<DirtySignalSignature> DirtySignal;

    /*--- Public signals ---*/

    /** @internal */
    DirtySignal dirty;

    /**
       @internal
       Copy the range, point set and texture size from other color map.
     */
    ColorMap& operator=(const ColorMap& other);

    bool operator==(const ColorMap& other) const;

    /** @internal */
    core::ColorMap& getImpl() { return *_impl; }
    /** @internal */
    const core::ColorMap& getImpl() const { return *_impl; }
    ///@}

private:
    /*--- Private member attributes ---*/
    core::ColorMap* _impl;

    /*--- Private member functions ---*/
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version);

#ifndef DOXYGEN_TO_BREATHE // Guarded to avoid confusing breathe
    /* Doesn't emit the dirty signal */
    template <class Archive>
    void load(Archive& ar, const unsigned int version);

    template <class Archive>
    void save(Archive& ar, const unsigned int version) const;
#endif

    void _loadOldFileFormat(const std::string& file);
};
}
}
#endif
