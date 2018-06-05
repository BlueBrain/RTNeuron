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

#include "attributeMapHelpers.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace AttributeMapHelpers
{
namespace
{
bool _getColor3(const AttributeMap& attributes, const std::string& name,
                osg::Vec4& color)
{
    osg::Vec4d vd(0, 0, 0, 1);
    if (attributes.get(name, vd[0], vd[1], vd[2]) == 3)
    {
        color = vd;
        return true;
    }

    osg::Vec4 v(0, 0, 0, 1);
    if (attributes.get(name, v[0], v[1], v[2]) == 3)
    {
        color = v;
        return true;
    }
    return false;
}

bool _getColor4(const AttributeMap& attributes, const std::string& name,
                osg::Vec4& color)
{
    osg::Vec4d vd;
    if (attributes.get(name, vd[0], vd[1], vd[2], vd[3]) == 4)
    {
        color = vd;
        return true;
    }
    osg::Vec4 v;
    if (attributes.get(name, v[0], v[1], v[2], v[3]) == 4)
    {
        color = v;
        return true;
    }
    return false;
}
}

bool getColor(const AttributeMap& attributes, const std::string& name,
              osg::Vec4& color)

{
    try
    {
        switch (attributes.getParameters(name).size())
        {
        case 3:
            return _getColor3(attributes, name, color);
            break;
        case 4:
            return _getColor4(attributes, name, color);
            break;
        default:
            return false;
        }
    }
    catch (...)
    {
        return false;
    }
}

void getRequiredColor(const AttributeMap& attributes, const std::string& name,
                      osg::Vec4& v)
{
    if (!getColor(attributes, name, v))
        LBTHROW(std::runtime_error("Error converting attribute " + name +
                                   " to RGBA color"));
}

bool _getColorf(const AttributeMap::AttributeProxy& proxy, osg::Vec4& v)
{
    try
    {
        switch (proxy.getSize())
        {
        case 3:
            v.set(proxy(0), proxy(1), proxy(2), 1);
            return true;
        case 4:
            v.set(proxy(0), proxy(1), proxy(2), proxy(3));
            return true;
        }
    }
    catch (...)
    {
    }
    return false;
}

bool _getColord(const AttributeMap::AttributeProxy& proxy, osg::Vec4& v)
{
    try
    {
        switch (proxy.getSize())
        {
        case 3:
            v.set((double)proxy(0), (double)proxy(1), (double)proxy(2), 1);
            return true;
        case 4:
            v.set((double)proxy(0), (double)proxy(1), (double)proxy(2),
                  (double)proxy(3));
            return true;
        }
    }
    catch (...)
    {
    }
    return false;
}

bool getColor(const AttributeMap::AttributeProxy& proxy, osg::Vec4& v)
{
    if (_getColord(proxy, v))
        return true;
    return _getColorf(proxy, v);
}

void getRequiredColor(const AttributeMap::AttributeProxy& proxy, osg::Vec4& v)
{
    if (!getColor(proxy, v))
        LBTHROW(std::runtime_error("Error converting attribute to RGBA color"));
}

bool getMatrix(const AttributeMap& attributes, const std::string& name,
               vmml::Matrix3f& matrix)
{
    return getMatrix(attributes(name), matrix);
}

bool getMatrix(const AttributeMap::AttributeProxy& proxy,
               vmml::Matrix3f& matrix)
{
    try
    {
        /* Extracing a double matrix because Python always uses doubles. */
        for (size_t i = 0; i != 9; ++i)
        {
            const double tmp = proxy(i);
            matrix(i / 3, i % 3) = float(tmp);
        }
        return true;
    }
    catch (...)
    {
    }
    return false;
}
}
}
}
}
