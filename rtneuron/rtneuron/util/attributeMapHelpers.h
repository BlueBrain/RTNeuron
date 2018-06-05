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

#ifndef RTNEURON_ATTRIBUTEMAPHELPERS_H
#define RTNEURON_ATTRIBUTEMAPHELPERS_H

#include "../AttributeMap.h"

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Vec4d>

#include <vmmlib/vmmlib.hpp>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace AttributeMapHelpers
{
/**
   Extracts an RGB[A] colors from an attribute map using an attribute name.

   Returns true iff the color was successfully extracted.
*/
bool getColor(const AttributeMap& attributes, const std::string& name,
              osg::Vec4& v);

/**
   Extracts an RGB[A] colors from an attribute proxy.

   Returns true iff the color was successfully extracted.
*/
bool getColor(const AttributeMap::AttributeProxy& proxy, osg::Vec4& v);

/**
   The same as the function above but throws an exception instead of
   returning false.
*/
void getRequiredColor(const AttributeMap& attributes, const std::string& name,
                      osg::Vec4& v);

/**
   The same as the function above but throws an exception instead of
   returning false.
*/
void getRequiredColor(const AttributeMap::AttributeProxy& proxy, osg::Vec4& v);

/**
   Extracts a matrix from an attribue map using an attribute name.
   If an error occurs the output matrix is not modified.

   Returns true iff the matrix was successfully extracted.
*/
bool getMatrix(const AttributeMap& attributes, const std::string& name,
               vmml::Matrix3f& matrix);

/**
   Extracts a matrix from an attribute proxy
   If an error occurs the output matrix is not modified.

   Returns true iff the matrix was successfully extracted.
*/
bool getMatrix(const AttributeMap::AttributeProxy& proxy,
               vmml::Matrix3f& matrix);

inline void getVector(const AttributeMap& attributes, const std::string& name,
                      osg::Vec4& v)
{
    attributes.get(name, v[0], v[1], v[2], v[3]);
}

inline void getVector(const AttributeMap& attributes, const std::string& name,
                      osg::Vec3& v)
{
    attributes.get(name, v[0], v[1], v[2]);
}

/**
   Tries to extract an enum from an attribute map either using directly the
   enum type or casting from an integer (or char if that's the enum size).

   @return True if the attribute exists and the conversion was successful.
*/
template <typename T>
inline bool getEnum(const AttributeMap& attributes, const std::string& name,
                    T& value)
{
    /* Implemented according to C99 standard rules for enums, 6.7.2.2p4 */
    if (attributes.get(name, value) == 1)
        return true;

    if (sizeof(T) == sizeof(int))
    {
        unsigned int ui = 0;
        if (attributes.get(name, ui) == 1)
        {
            value = (T)ui;
            return true;
        }
        int i = 0;
        if (attributes.get(name, i) == 1)
        {
            value = (T)i;
            return true;
        }
    }
    else if (sizeof(T) == sizeof(char))
    {
        char c = 0;
        if (attributes.get(name, c) == 1)
        {
            value = (T)c;
            return true;
        }
    }
    return false;
}

/**
   The same as above but return by value.
   @throw Runtime error is the attribute name is not found or the conversion
   is not possible.
*/
template <typename T>
inline T getEnum(const AttributeMap& attributes, const std::string& name)
{
    /* Implemented according to C99 standard rules for enums, 6.7.2.2p4 */
    T t;
    if (!getEnum(attributes, name, t))
        LBTHROW(std::runtime_error("Error converting parameter " + name +
                                   " into enum " + lunchbox::className(t)));
    return t;
}

template <typename T>
inline T getEnum(const AttributeMap::AttributeProxy& proxy)
{
#define TRY_CONVERSION(type) \
    try                      \
    {                        \
        type t = proxy;      \
        return (T)t;         \
    }                        \
    catch (...)              \
    {                        \
    }

    TRY_CONVERSION(T)
    if (sizeof(T) == sizeof(int))
    {
        TRY_CONVERSION(unsigned int)
        TRY_CONVERSION(int)
    }
    else if (sizeof(T) == sizeof(char))
    {
        TRY_CONVERSION(char)
    }
#undef TRY_CONVERSION
    LBTHROW(std::runtime_error("Error converting parameter into enum " +
                               lunchbox::className(T())));
}

template <typename T>
inline T getEnum(const AttributeMap& attributes, const std::string& name,
                 const T& default_)
{
    T value;
    if (getEnum<T>(attributes, name, value))
        return value;
    return default_;
}
}
}
}
}
#endif
