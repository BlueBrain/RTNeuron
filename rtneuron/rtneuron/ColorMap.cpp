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

#include "config/constants.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "render/ColorMap.h"

#include <lunchbox/any.h>
#include <lunchbox/debug.h>

#include <boost/serialization/export.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>

BOOST_CLASS_EXPORT(bbp::rtneuron::ColorMap)

namespace bbp
{
namespace rtneuron
{
/*
  Constructor
*/
ColorMap::ColorMap()
    : _impl(new core::ColorMap())
{
}

ColorMap::~ColorMap()
{
    delete _impl;
}

ColorMap& ColorMap::operator=(const ColorMap& other)
{
    if (&other == this)
        return *this;

    *_impl = *other._impl;

    dirty();

    return *this;
}

bool ColorMap::operator==(const ColorMap& other) const
{
    if (&other == this)
        return true;
    return *_impl == *other._impl;
}

void ColorMap::setPoints(const ColorPoints& colorPoints)
{
    _impl->setPoints(colorPoints);
    dirty();
}

const ColorMap::ColorPoints& ColorMap::getPoints() const
{
    return _impl->getPoints();
}

void ColorMap::setRange(const float min, const float max)
{
    if (max < min)
        throw std::runtime_error("Invalid color map range");

    _impl->setRange(min, max);

    dirty();
}

osg::Vec4 ColorMap::getColor(const float value) const
{
    return _impl->getColor(value);
}

void ColorMap::getRange(float& min, float& max) const
{
    _impl->getRange(min, max);
}

void ColorMap::load(const std::string& fileName)
{
    std::ifstream file(fileName.c_str());
    if (!file)
        throw std::runtime_error("Error opening file: " + fileName);
    /* The dirty signal is emitted implicitly by the internal calls
       to setPoints. */
    try
    {
        /* Try loading the old file format first */
        _loadOldFileFormat(fileName);
    }
    catch (std::runtime_error&)
    {
        boost::archive::text_iarchive archive(file);
        archive&* this;
    }
}

void ColorMap::save(const std::string& fileName)
{
    std::ofstream file(fileName.c_str());
    if (!file)
        throw std::runtime_error("Error opening file: " + fileName);
    boost::archive::text_oarchive archive(file);
    archive&* this;
}

void ColorMap::setTextureSize(size_t texels)
{
    if (texels != _impl->getTextureSize())
    {
        _impl->setTextureSize(texels);
        dirty();
    }
}

size_t ColorMap::getTextureSize() const
{
    return _impl->getTextureSize();
}

template <class Archive>
void ColorMap::serialize(Archive& ar, const unsigned int version)
{
    /* Registering lunchbox::Any<ColorMapPtr> in the archive
       programmatically (the export macro doesn't seem to work with Collage's
       archives).
       Intuitively, this is not the right place to do it, because
       trying to serialize a lunchbox::Any<ColorMapPtr> before this function
       is invoked will fail. I'd assume than when a lunchbox::Any<ColorMapPtr>
       is going to be serialized, the type registry is searched first to look
       for the pointer serialization and only then this function is invoked.
       However this works, probably because the type registry is  global, and
       this function is invoked before any lunchbox::Any<ColorMapPtr> is
       serialized. */
    ar.template register_type<
        lunchbox::Any::holder<std::shared_ptr<ColorMap>>>();
    boost::serialization::split_member(ar, *this, version);
}

template <class Archive>
void ColorMap::load(Archive& ar, const unsigned int /* version */)
{
    unsigned int textureSize = 0;
    ar& textureSize;
    ColorPoints points;
    size_t size = 0;
    ar& size;
    for (size_t i = 0; i != size; ++i)
    {
        float value = 0, r = 0, g = 0, b = 0, a = 0;
        ar& value& r& g& b& a;
        points[value] = osg::Vec4(r, g, b, a);
    }
    osg::Vec2 range;
    ar& range[0] & range[1];
    _impl->set(textureSize, points, range);
    dirty();
}

void ColorMap::_loadOldFileFormat(const std::string& fileName)
{
    std::ifstream in(fileName.c_str());
    if (!in)
        throw std::runtime_error("Error opening file: " + fileName);
    float min, max;
    in >> min >> max >> std::ws;
    if (max < min)
        throw std::runtime_error("Invalid color map range");

    std::vector<osg::Vec4> colors;
    while (!in.eof())
    {
        std::string line;
        std::getline(in, line);
        if (in.fail())
            break;

        std::stringstream buffer(line);
        unsigned int r, g, b, a = 255;
        buffer >> r >> g >> b;
        buffer >> std::ws;
        if (!buffer.eof())
            buffer >> a;
        if (buffer.fail())
            break;
        colors.push_back(osg::Vec4(r / 255.0, g / 255.0, b / 255.0, a / 255.0));

        in >> std::ws;
    }
    if (in.fail() || colors.size() == 0)
    {
        throw std::runtime_error(
            "Parsing error reading colormap file in old format: " + fileName);
    }

    ColorPoints points;
    for (size_t i = 0; i != colors.size(); ++i)
    {
        const float value = min + (max - min) * i / float(colors.size() - 1);
        points[value] = colors[i];
    }
    _impl->setTextureSize(colors.size());
    _impl->setPoints(points);
}

template <class Archive>
void ColorMap::save(Archive& ar, const unsigned int /* version */) const
{
    size_t size = _impl->getTextureSize();
    ar& size;
    const ColorPoints& points = _impl->getOriginalPoints();
    size = points.size();
    ar& size;
    for (ColorPoints::const_iterator i = points.begin(); i != points.end(); ++i)
    {
        float r, g, b, a;
        r = i->second[0];
        g = i->second[1];
        b = i->second[2];
        a = i->second[3];
        ar & i->first& r& g& b& a;
        ;
    }
    float min, max;
    _impl->getRange(min, max);
    ar& min& max;
}

template void ColorMap::serialize<net::DataOStreamArchive>(
    net::DataOStreamArchive& ar, const unsigned int file_version);

template void ColorMap::serialize<net::DataIStreamArchive>(
    net::DataIStreamArchive& ar, const unsigned int file_version);

template void ColorMap::serialize<boost::archive::binary_oarchive>(
    boost::archive::binary_oarchive& ar, const unsigned int file_version);

template void ColorMap::load<net::DataIStreamArchive>(
    net::DataIStreamArchive& ar, const unsigned int file_version);

template void ColorMap::save<net::DataOStreamArchive>(
    net::DataOStreamArchive& ar, const unsigned int file_version) const;

template void ColorMap::load<boost::archive::binary_oarchive>(
    boost::archive::binary_oarchive& ar, const unsigned int file_version);
}
}
