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

#ifndef RTNEURON_UTIL_EXTENSIONS_H
#define RTNEURON_UTIL_EXTENSIONS_H

#include <osg/Version>

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
#include <osg/GLExtensions>
#else
#include <osg/BufferObject>
#include <osg/Drawable>
#include <osg/FrameBufferObject>
#include <osg/GL2Extensions>
#endif

namespace bbp
{
namespace rtneuron
{
namespace core
{
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)

typedef osg::GLExtensions DrawExtensions;
inline osg::GLExtensions* getDrawExtensions(const unsigned int contextID)
{
    return osg::GLExtensions::Get(contextID, true);
}

typedef osg::GLExtensions FBOExtensions;
inline osg::GLExtensions* getFBOExtensions(const unsigned int contextID)
{
    return osg::GLExtensions::Get(contextID, true);
}

typedef osg::GLExtensions BufferExtensions;
inline osg::GLExtensions* getBufferExtensions(const unsigned int contextID)
{
    return osg::GLExtensions::Get(contextID, true);
}

typedef osg::GLExtensions GL2Extensions;
inline osg::GLExtensions* getGL2Extensions(const unsigned int contextID)
{
    return osg::GLExtensions::Get(contextID, true);
}

#else

typedef osg::Drawable::Extensions DrawExtensions;

inline osg::Drawable::Extensions* getDrawExtensions(
    const unsigned int contextID)
{
    return osg::Drawable::getExtensions(contextID, true);
}

typedef osg::FBOExtensions FBOExtensions;
inline osg::FBOExtensions* getFBOExtensions(const unsigned int contextID)
{
    return osg::FBOExtensions::instance(contextID, true);
}

typedef osg::GLBufferObject::Extensions BufferExtensions;
inline osg::GLBufferObject::Extensions* getBufferExtensions(
    const unsigned int contextID)
{
    return osg::GLBufferObject::getExtensions(contextID, true);
}

typedef osg::GL2Extensions GL2Extensions;
inline osg::GL2Extensions* getGL2Extensions(const unsigned int contextID)
{
    return osg::GL2Extensions::Get(contextID, true);
}

#endif
}
}
}
#endif
