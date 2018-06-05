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

#include <eq/eq.h>

#include "FrameData.h"

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*
  Static definitions
*/
static FrameData::ObjectAttribute::ConstructorMap& _constructors()
{
    static FrameData::ObjectAttribute::ConstructorMap map;
    return map;
}

/*
  Helper functions
*/
co::DataOStream& operator<<(co::DataOStream& out,
                            FrameData::FrameAttributeSetter& attribute)
{
    out << (unsigned char)attribute._type;
    switch (attribute._type)
    {
    case FrameData::BOOL:
        out << attribute._value._bool;
        break;
    case FrameData::FLOAT:
        out << attribute._value._float;
        break;
    case FrameData::DOUBLE:
        out << attribute._value._double;
        break;
    case FrameData::INT:
        out << attribute._value._int;
        break;
    default:;
    }
    return out;
}

co::DataIStream& operator>>(co::DataIStream& in,
                            FrameData::FrameAttributeSetter& attribute)
{
    char type;
    in >> type;
    attribute._type = (FrameData::AttributeType)type;
    switch (attribute._type)
    {
    case FrameData::BOOL:
        in >> attribute._value._bool;
        break;
    case FrameData::FLOAT:
        in >> attribute._value._float;
        break;
    case FrameData::DOUBLE:
        in >> attribute._value._double;
        break;
    case FrameData::INT:
        in >> attribute._value._int;
        break;
    default:;
    }
    return in;
}

/*
  Member functions of member classes
*/
FrameData::FrameAttributeGetter& FrameData::FrameAttributeGetter::operator=(
    const FrameData::FrameAttributeSetter& rhs)
{
    if (_type != rhs._type)
    {
        _type = INVALID;
    }
    else
    {
        switch (_type)
        {
        case BOOL:
            *_value._bool = rhs._value._bool;
            break;
        case FLOAT:
            *_value._float = rhs._value._float;
            break;
        case DOUBLE:
            *_value._double = rhs._value._double;
            break;
        case INT:
            *_value._int = rhs._value._int;
            break;
        default:
            std::cerr << "This code shouldn't be reached" << std::endl;
            abort();
        }
    }
    return *this;
}

FrameData::ObjectAttributePtr FrameData::ObjectAttribute::deserialize(
    co::DataIStream& in)
{
    std::string type;
    in >> type;

    ConstructorMap::const_iterator constructor = _constructors().find(type);
    if (constructor == _constructors().end())
    {
        std::cerr << "Error: Unknown frame object type to be deserialized: "
                  << type << std::endl
                  << "       This may leave the stream inconsistent"
                  << std::endl;
        return ObjectAttributePtr();
    }

    ObjectAttributePtr object = constructor->second();
    object->deserializeImplementation(in);

    return object;
}

/*
  Member functions
*/
FrameData::FrameData()
    : statistics(false)
    , isIdle(false)
#ifndef NDEBUG
    , drawViewportBorder(false)
#endif
{
}

void FrameData::getInstanceData(co::DataOStream& out)
{
    out << statistics << isIdle;
#ifndef NDEBUG
    out << drawViewportBorder;
#endif

    out << commits.size();
    for (ObjectUUIDCommitMap::const_iterator i = commits.begin();
         i != commits.end(); ++i)
    {
        out << i->first << i->second;
    }

    /* Serializing the IDs of the active scenes */
    out << activeScenes.size();
    for (SceneIDs::iterator i = activeScenes.begin(); i != activeScenes.end();
         ++i)
    {
        out << *i;
    }

    /* Serializing attributes for next frame. */
    std::unique_lock<std::mutex> lock(_mutex);
    /* Basic types attributes */
    out << _nextFrameAttributes.size();
    for (AttributeMap::iterator i = _nextFrameAttributes.begin();
         i != _nextFrameAttributes.end(); ++i)
    {
        out << i->first << i->second;
    }

    /* Object attributes */
    out << _nextFrameObjectAttributes.size();
    for (ObjectAttributeMap::iterator i = _nextFrameObjectAttributes.begin();
         i != _nextFrameObjectAttributes.end(); ++i)
    {
        out << i->first;
        i->second->serialize(out);
    }

    /* Clearing pending attributes */
    _currentFrameAttributes = _nextFrameAttributes;
    _nextFrameAttributes.clear();

    _currentFrameObjectAttributes = _nextFrameObjectAttributes;
    _nextFrameObjectAttributes.clear();
}

void FrameData::applyInstanceData(co::DataIStream& in)
{
    in >> statistics >> isIdle;
#ifndef NDEBUG
    in >> drawViewportBorder;
#endif

    commits.clear();
    size_t commitCount;
    in >> commitCount;
    for (size_t i = 0; i != commitCount; ++i)
    {
        co::uint128_t uuid, commitId;
        in >> uuid >> commitId;
        commits[uuid] = commitId;
    }

    /* Deserializing active scene IDs. */
    activeScenes.clear();
    size_t sceneCount;
    in >> sceneCount;
    for (size_t i = 0; i < sceneCount; ++i)
    {
        unsigned int id;
        in >> id;
        activeScenes.push_back(id);
    }

    /* Deserializing attributes for next frame. */
    size_t size;
    std::unique_lock<std::mutex> lock(_mutex);
    /* Basic types attributes */
    in >> size;
    _currentFrameAttributes.clear();
    for (size_t i = 0; i < size; ++i)
    {
        std::string name;
        FrameAttributeSetter attribute;
        in >> name >> attribute;
        _currentFrameAttributes[name] = attribute;
    }

    /* Object attributes */
    in >> size;
    _currentFrameObjectAttributes.clear();
    for (size_t i = 0; i < size; ++i)
    {
        std::string name;
        in >> name;
        _currentFrameObjectAttributes[name] = ObjectAttribute::deserialize(in);
    }
}

void FrameData::setFrameAttribute(const char* name,
                                  const FrameAttributeSetter& attribute)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _nextFrameAttributes[name] = attribute;
}

bool FrameData::getFrameAttribute(const char* name,
                                  FrameAttributeGetter attribute) const
{
    AttributeMap::const_iterator attr = _currentFrameAttributes.find(name);
    if (attr != _currentFrameAttributes.end())
    {
        // cppcheck-suppress uselessAssignmentArg
        // cppcheck-suppress uselessAssignmentPtrArg
        attribute = attr->second;
        return true;
    }
    return false;
}

void FrameData::setFrameAttribute(const char* name, ObjectAttribute* object)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _nextFrameObjectAttributes[name] = object;
}

FrameData::ObjectAttribute* FrameData::getFrameAttribute(const char* name)
{
    ObjectAttributeMap::iterator attr =
        _currentFrameObjectAttributes.find(name);
    if (attr != _currentFrameObjectAttributes.end())
        return attr->second.get();
    return 0;
}

void FrameData::ObjectAttribute::registerType_(
    const std::string& name, const boost::function0<ObjectAttribute*>& ctor)
{
    _constructors().insert(std::make_pair(name, ctor));
}
}
}
}
