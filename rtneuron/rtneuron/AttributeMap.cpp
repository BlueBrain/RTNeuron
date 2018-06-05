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

#include "AttributeMap.h"

#include "detail/attributeMapTypeRegistration.h"

#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "util/MurmurHash3.h"

#include <lunchbox/anySerialization.h>

#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

/*
  Registration of types used in rtneuron::core::any for serialization of their
  any placeholders.
  Declared according to types known in AttributeMap.inc
*/
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<bbp::rtneuron::AttributeMapPtr>,
                        "a:AttributeMapPtr");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<bool>, "a:b");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<int>, "a:i");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<unsigned int>, "a:ui");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<float>, "a:f");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<double>, "a:d");
BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<std::string>, "a:s");

namespace bbp
{
namespace rtneuron
{
typedef const std::vector<lunchbox::Any> Anys;

/*
  AttributeMap::_Impl
*/

class AttributeMap::_Impl
{
public:
    typedef std::map<std::string, TypeRegistrationPtr> TypeRegistry;
    static TypeRegistry s_typeRegistry;
    static TypeRegistry s_typeNameRegistry;

    typedef std::map<std::string, Parameters> Attributes;

    void triggerAttributeChange(const AttributeMap& map,
                                const std::string& name)
    {
        map.attributeChanged(name);
    }

    _Impl(AttributeMap& parent)
        : _parent(parent)
        , _hashDirty(true)
        , _hashValue(0)
    {
        _parent.attributeMapChanged.connect(
            boost::bind(&_Impl::triggerAttributeChange, this, _1, _2));
    }

    ~_Impl()
    {
        /* Disconnecting from all the attribute maps */
        for (Attributes::iterator i = _attributes.begin();
             i != _attributes.end(); ++i)
        {
            checkAndDisconnectFromAttributeMap(i->first, i->second);
            disconnectFromUserDefinedObjects(i->first, i->second);
        }
        _parent.attributeMapChanged.disconnect(
            boost::bind(&_Impl::triggerAttributeChange, this, _1, _2));
    }

    void merge(const _Impl& other)
    {
        for (Attributes::const_iterator i = other._attributes.begin();
             i != other._attributes.end(); ++i)
        {
            Parameters params;
            if (i->second.size() == 1 &&
                i->second[0].type() == typeid(AttributeMapPtr))
            {
                AttributeMapPtr nested(new AttributeMap());
                nested->merge(
                    *lunchbox::any_cast<const AttributeMapPtr&>(i->second[0]));
                params.push_back(nested);
            }
            else
            {
                params = i->second;
            }
            replaceParams(i->first, params);
        }
        if (!other._attributes.empty())
            _hashDirty = true;
    }

    static Attributes::const_iterator findAndCheckSize(
        const Attributes& attributes, const std::string& name,
        unsigned int size)
    {
        Attributes::const_iterator attr = attributes.find(name);
        if (attr == attributes.end() || attr->second.size() != size)
            return attributes.end();
        return attr;
    }

    void rewireSignals(Parameters& current, Parameters& incoming,
                       const std::string& name)
    {
        /* Check if the old parameters was an attribute map and
           disconnecting from it */
        checkAndDisconnectFromAttributeMap(name, current);
        disconnectFromUserDefinedObjects(name, current);
        if (incoming.size() == 1)
            checkAndRegisterAttributeMapParameter(name, incoming[0]);
        connectToUserDefinedObjects(name, incoming);
    }

    void checkAndRegisterAttributeMapParameter(const std::string& name,
                                               const lunchbox::Any& p)
    {
        if (p.type() == typeid(AttributeMapPtr))
        {
            AttributeMapPtr map = lunchbox::any_cast<const AttributeMapPtr&>(p);
            map->attributeMapChanged.connect(
                boost::bind(&_Impl::onNestedAttributeMapChanged, this, name, _1,
                            _2));
            map->attributeMapChanging.connect(
                boost::bind(&_Impl::onNestedAttributeMapChanging, this, name,
                            _1, _2, _3));
        }
    }

    void checkAndDisconnectFromAttributeMap(const std::string& name,
                                            const Parameters& params)
    {
        if (params.size() == 1 && params[0].type() == typeid(AttributeMapPtr))
        {
            AttributeMapPtr map =
                lunchbox::any_cast<const AttributeMapPtr&>(params[0]);
            map->attributeMapChanged.disconnect(
                boost::bind(&_Impl::onNestedAttributeMapChanged, this, name, _1,
                            _2));
            map->attributeMapChanging.disconnect(
                boost::bind(&_Impl::onNestedAttributeMapChanging, this, name,
                            _1, _2, _3));
        }
    }

    void connectToUserDefinedObjects(const std::string& name,
                                     Parameters& params)
    {
        for (Parameters::iterator p = params.begin(); p != params.end(); ++p)
        {
            TypeRegistry::const_iterator entry =
                s_typeRegistry.find(p->type().name());
            if (entry == s_typeRegistry.end())
                continue;

            entry->second->connectDirtySignal(
                boost::bind(&_Impl::onUserDefinedAttributeChanged, this, name),
                *p);
        }
    }

    void disconnectFromUserDefinedObjects(const std::string& name,
                                          Parameters& params)
    {
        for (Parameters::iterator p = params.begin(); p != params.end(); ++p)
        {
            TypeRegistry::const_iterator entry =
                s_typeRegistry.find(p->type().name());
            if (entry == s_typeRegistry.end())
                continue;

            entry->second->disconnectDirtySignal(
                boost::bind(&_Impl::onUserDefinedAttributeChanged, this, name),
                *p);
        }
    }

    void ensureParameterNotAttributeMap(const lunchbox::Any& p)
    {
        if (p.type() == typeid(AttributeMapPtr))
        {
            LBTHROW(
                std::runtime_error("AttributeMap is not allowed inside a"
                                   " multi-parameter attribute"));
        }
    }

    void ensureNoCycles(_Impl* impl, const AttributeMapPtr& map)
    {
        if (map->_impl == impl)
            LBTHROW(std::runtime_error("AttributeMap cycles are not allowed"));

        for (Attributes::const_iterator i = map->_impl->_attributes.begin();
             i != map->_impl->_attributes.end(); ++i)
        {
            const Parameters& params = i->second;
            if (params.size() == 1 &&
                params[0].type() == typeid(AttributeMapPtr))
            {
                ensureNoCycles(impl,
                               lunchbox::any_cast<AttributeMapPtr>(params[0]));
            }
        }
    }

    /* The first parameter is bounded on connection so we know the name
       of the parameter and it's easier to composite the final name. */
    void onNestedAttributeMapChanged(const std::string& mapName,
                                     const AttributeMap&,
                                     const std::string& name)
    {
        _parent.attributeMapChanged(_parent, mapName + '.' + name);
        _hashDirty = true;
    }

    void onUserDefinedAttributeChanged(const std::string& name)
    {
        _parent.attributeMapChanged(_parent, name);
        _hashDirty = true;
    }

    /* The first parameter is bounded on connection so we know the name
       of the parameter and it's easier to composite the final name. */
    void onNestedAttributeMapChanging(const std::string& mapName,
                                      const AttributeMap&,
                                      const std::string& name,
                                      const AttributeProxy& parameters)
    {
        _parent.attributeMapChanging(_parent, mapName + '.' + name, parameters);
    }

    const AttributeMap::Parameters& getParameters(const std::string& name)
    {
        _Impl::Attributes::const_iterator attr = _attributes.find(name);
        if (attr == _attributes.end())
        {
            std::stringstream msg;
            msg << "Attribute " << name << " not found";
            throw std::runtime_error(msg.str());
        }

        return attr->second;
    }

    void replaceParams(const std::string& name, Parameters& params)
    {
        if (name.empty())
            LBTHROW(
                std::runtime_error("Attribute names cannot be empty strings"));

        if (params.size() > 1)
        {
            for (Parameters::const_iterator i = params.begin();
                 i != params.end(); ++i)
                ensureParameterNotAttributeMap(*i);
        }
        else
        {
            if (params.begin()->type() == typeid(AttributeMapPtr))
                ensureNoCycles(this,
                               lunchbox::any_cast<AttributeMapPtr>(params[0]));
        }

        _parent.attributeMapChanging(_parent, name, AttributeProxy(params));

        Parameters& old = _attributes[name];
        rewireSignals(old, params, name);

        if (params == old)
            return;
        old.swap(params);
        _hashDirty = true;
        _parent.attributeMapChanged(_parent, name);
    }

    void print(std::ostream& out, const std::string& indent = "") const
    {
        for (Attributes::const_iterator i = _attributes.begin();
             i != _attributes.end(); ++i)
        {
            out << indent << i->first << ':';
            const Parameters& params = i->second;
            if (params.size() == 1 &&
                params[0].type() == typeid(AttributeMapPtr))
            {
                AttributeMapPtr nested =
                    lunchbox::any_cast<AttributeMapPtr>(params[0]);
                out << std::endl;
                nested->_impl->print(out, indent + "  ");
            }
            else
            {
                for (Parameters::const_iterator p = params.begin();
                     p != params.end(); ++p)
                {
                    out << ' ';
                    if (p->type() == typeid(bool))
                        out << lunchbox::any_cast<bool>(*p);
                    else if (p->type() == typeid(int))
                        out << lunchbox::any_cast<int>(*p);
                    else if (p->type() == typeid(unsigned int))
                        out << lunchbox::any_cast<unsigned int>(*p);
                    else if (p->type() == typeid(float))
                        out << lunchbox::any_cast<float>(*p);
                    else if (p->type() == typeid(double))
                        out << lunchbox::any_cast<double>(*p);
                    else if (p->type() == typeid(std::string))
                        out << lunchbox::any_cast<const std::string&>(*p);
                    else if (p->type() == typeid(AttributeMapPtr))
                        out << *lunchbox::any_cast<const AttributeMapPtr&>(*p);
                    else
                    {
                        TypeRegistry::const_iterator entry =
                            s_typeRegistry.find(p->type().name());
                        if (entry == s_typeRegistry.end())
                            out << "???";
                        else
                            entry->second->print(out, *p);
                    }
                }
                out << std::endl;
            }
        }
    }

    void computeHash() const
    {
        if (!_hashDirty)
            return;

        std::vector<uint32_t> buffer;

        for (Attributes::const_iterator i = _attributes.begin();
             i != _attributes.end(); ++i)
        {
            const Parameters& params = i->second;
            if (params.size() == 1 &&
                params[0].type() == typeid(AttributeMapPtr))
            {
                AttributeMapPtr nested =
                    lunchbox::any_cast<AttributeMapPtr>(params[0]);
                buffer.push_back(nested->hash());
            }
            else
            {
                for (Parameters::const_iterator p = params.begin();
                     p != params.end(); ++p)
                {
                    if (p->type() == typeid(bool))
                    {
                        buffer.push_back(lunchbox::any_cast<bool>(*p));
                    }
                    else if (p->type() == typeid(int))
                    {
                        buffer.push_back(lunchbox::any_cast<int>(*p));
                    }
                    else if (p->type() == typeid(unsigned int))
                    {
                        buffer.push_back(lunchbox::any_cast<unsigned int>(*p));
                    }
                    else if (p->type() == typeid(float))
                    {
                        const float x = lunchbox::any_cast<float>(*p);
                        buffer.push_back(
                            *reinterpret_cast<const uint32_t*>(&x));
                    }
                    else if (p->type() == typeid(double))
                    {
                        const double x = lunchbox::any_cast<double>(*p);
                        const uint32_t* y =
                            reinterpret_cast<const uint32_t*>(&x);
                        buffer.push_back(y[0]);
                        buffer.push_back(y[1]);
                    }
                    else if (p->type() == typeid(std::string))
                    {
                        buffer.push_back(_computeBufferHash(
                            lunchbox::any_cast<const std::string&>(*p)));
                    }
                    else
                    {
                        TypeRegistry::const_iterator entry =
                            s_typeRegistry.find(p->type().name());
                        if (entry == s_typeRegistry.end())
                        {
                            std::stringstream msg;
                            msg << "Cannot compute hash, unregistered type "
                                << p->type().name();
                            LBTHROW(std::runtime_error(msg.str()));
                        }
                        buffer.push_back(entry->second->hash(*p));
                    }
                }
            }
        }
        MurmurHash3_x86_32(buffer.data(), buffer.size() * 4, 0, &_hashValue);
        _hashDirty = false;
    }

    template <class Archive>
    void serialize(Archive& archive, const unsigned int /* version */)
    {
        /* Not the ideal solution, but at least works */
        lunchbox::registerTypelist<lunchbox::podTypes>(archive);
        archive& _attributes;
    }

    bool compare(const Anys& a, const Anys& b)
    {
        bool equal = true;
        Anys::const_iterator i = a.begin(), j = b.begin();
        for (; i != a.end() && j != b.end() && equal; ++i, ++j)
        {
            try
            {
                if (i->type() == typeid(bool))
                    equal &= lunchbox::any_cast<bool>(*i) ==
                             lunchbox::any_cast<bool>(*j);
                else if (i->type() == typeid(int))
                    equal &= lunchbox::any_cast<int>(*i) ==
                             lunchbox::any_cast<int>(*j);
                else if (i->type() == typeid(unsigned int))
                    equal &= lunchbox::any_cast<unsigned int>(*i) ==
                             lunchbox::any_cast<unsigned int>(*j);
                else if (i->type() == typeid(float))
                    equal &= lunchbox::any_cast<float>(*i) ==
                             lunchbox::any_cast<float>(*j);
                else if (i->type() == typeid(double))
                    equal &= lunchbox::any_cast<double>(*i) ==
                             lunchbox::any_cast<double>(*j);
                else if (i->type() == typeid(std::string))
                    equal &= lunchbox::any_cast<const std::string&>(*i) ==
                             lunchbox::any_cast<const std::string&>(*j);
                else if (i->type() == typeid(bbp::rtneuron::AttributeMapPtr))
                    equal &= *lunchbox::any_cast<const AttributeMapPtr&>(*i) ==
                             *lunchbox::any_cast<const AttributeMapPtr&>(*j);
                else
                {
                    /* Checking registered types */
                    TypeRegistry::const_iterator entry =
                        s_typeRegistry.find(i->type().name());
                    if (entry == s_typeRegistry.end())
                    {
                        std::stringstream msg;
                        msg << "Comparison between non registered types";
                        LBTHROW(std::runtime_error(msg.str()));
                    }
                    else
                        equal = entry->second->compare(*i, *j);
                }
            }
            catch (...)
            {
                equal = false;
            }
        }
        return equal && i == a.end() && j == b.end();
    }

    Attributes _attributes;

    AttributeMap& _parent;
    mutable bool _hashDirty;
    mutable boost::uint32_t _hashValue;
    std::string _docstring;
};

AttributeMap::_Impl::TypeRegistry AttributeMap::_Impl::s_typeRegistry;
AttributeMap::_Impl::TypeRegistry AttributeMap::_Impl::s_typeNameRegistry;

/*
  AttributeMap
*/
AttributeMap::AttributeMap()
    : _impl(new _Impl(*this))
{
}

AttributeMap::AttributeMap(const AttributeMap& other)
    : _impl(new _Impl(*this))
{
    _impl->merge(*other._impl);
}

AttributeMap::AttributeMap(AttributeMap&& other)
    : _impl(other._impl)
{
    other._impl = 0;
}

AttributeMap::~AttributeMap()
{
    delete _impl;
}

AttributeMap& AttributeMap::operator=(AttributeMap&& other)
{
    if (&other == this)
        return *this; /* In case of the improbable x = std::move(x) */

    delete _impl;
    _impl = other._impl;
    other._impl = 0;

    return *this;
}

void AttributeMap::merge(const AttributeMap& other)
{
    _impl->merge(*other._impl);
}

void AttributeMap::set(const std::string& name, const Parameter& p1)
{
    Parameters params;
    params.push_back(p1);
    _impl->replaceParams(name, params);
}

void AttributeMap::set(const std::string& name, const Parameter& p1,
                       const Parameter& p2)
{
    Parameters params;
    params.push_back(p1);
    params.push_back(p2);
    _impl->replaceParams(name, params);
}

void AttributeMap::set(const std::string& name, const Parameter& p1,
                       const Parameter& p2, const Parameter& p3)
{
    Parameters params;
    params.push_back(p1);
    params.push_back(p2);
    params.push_back(p3);
    _impl->replaceParams(name, params);
}

void AttributeMap::set(const std::string& name, const Parameter& p1,
                       const Parameter& p2, const Parameter& p3,
                       const Parameter& p4)
{
    Parameters params;
    params.push_back(p1);
    params.push_back(p2);
    params.push_back(p3);
    params.push_back(p4);
    _impl->replaceParams(name, params);
}

void AttributeMap::set(const std::string& name, const Parameter& p1,
                       const Parameter& p2, const Parameter& p3,
                       const Parameter& p4, const Parameter& p5)
{
    Parameters params;
    params.push_back(p1);
    params.push_back(p2);
    params.push_back(p3);
    params.push_back(p4);
    params.push_back(p5);
    _impl->replaceParams(name, params);
}

void AttributeMap::set(const std::string& name, const Parameters& params)
{
    Parameters newParams = params;
    _impl->replaceParams(name, newParams);
}

void AttributeMap::set(const std::string& name, const AttributeProxy& params)
{
    Parameters newParams = *params._parameters;
    _impl->replaceParams(name, newParams);
}

int AttributeMap::get(const std::string& name, OutWrapper p1) const
{
    _Impl::Attributes::const_iterator attr =
        _Impl::findAndCheckSize(_impl->_attributes, name, 1);
    if (attr == _impl->_attributes.end())
        return -1; /* If strict exception already thrown */
    const Parameters& params = attr->second;
    int matches = 0;
    try
    {
        // cppcheck-suppress uselessAssignmentArg
        p1 = params[0];
        ++matches;
    }
    catch (const std::runtime_error& e)
    {
    }
    return matches;
}

int AttributeMap::get(const std::string& name, OutWrapper p1,
                      OutWrapper p2) const
{
    _Impl::Attributes::const_iterator attr =
        _Impl::findAndCheckSize(_impl->_attributes, name, 2);
    if (attr == _impl->_attributes.end())
        return -1; /* If strict exception already thrown */
    const Parameters& params = attr->second;
    int matches = 0;
    try
    {
        // cppcheck-suppress uselessAssignmentArg
        p1 = params[0];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p2 = params[1];
        ++matches;
    }
    catch (const std::runtime_error& e)
    {
        return matches;
    }
    return matches;
}

int AttributeMap::get(const std::string& name, OutWrapper p1, OutWrapper p2,
                      OutWrapper p3) const
{
    _Impl::Attributes::const_iterator attr =
        _Impl::findAndCheckSize(_impl->_attributes, name, 3);
    if (attr == _impl->_attributes.end())
        return -1; /* If strict exception already thrown */
    const Parameters& params = attr->second;
    int matches = 0;
    try
    {
        // cppcheck-suppress uselessAssignmentArg
        p1 = params[0];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p2 = params[1];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p3 = params[2];
        ++matches;
    }
    catch (const std::runtime_error& e)
    {
    }
    return matches;
}

int AttributeMap::get(const std::string& name, OutWrapper p1, OutWrapper p2,
                      OutWrapper p3, OutWrapper p4) const
{
    _Impl::Attributes::const_iterator attr =
        _Impl::findAndCheckSize(_impl->_attributes, name, 4);
    if (attr == _impl->_attributes.end())
        return -1; /* If strict exception already thrown */
    const Parameters& params = attr->second;
    int matches = 0;
    try
    {
        // cppcheck-suppress uselessAssignmentArg
        p1 = params[0];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p2 = params[1];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p3 = params[2];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p4 = params[3];
        ++matches;
    }
    catch (const std::runtime_error& e)
    {
    }
    return matches;
}

int AttributeMap::get(const std::string& name, OutWrapper p1, OutWrapper p2,
                      OutWrapper p3, OutWrapper p4, OutWrapper p5) const
{
    _Impl::Attributes::const_iterator attr =
        _Impl::findAndCheckSize(_impl->_attributes, name, 5);
    if (attr == _impl->_attributes.end())
        return -1; /* If strict exception already thrown */
    const Parameters& params = attr->second;
    int matches = 0;
    try
    {
        // cppcheck-suppress uselessAssignmentArg
        p1 = params[0];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p2 = params[1];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p3 = params[2];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p4 = params[3];
        ++matches;
        // cppcheck-suppress uselessAssignmentArg
        p5 = params[4];
        ++matches;
    }
    catch (const std::runtime_error& e)
    {
    }
    return matches;
}

int AttributeMap::get(const std::string& name,
                      std::vector<OutWrapper>& params) const
{
    _Impl::Attributes::const_iterator attr = _impl->_attributes.find(name);
    if (attr == _impl->_attributes.end())
        return -1;
    const Parameters& in = attr->second;
    if (in.size() != params.size())
        return -1;
    int matches = 0;
    std::vector<OutWrapper>::iterator i = params.begin();
    Parameters::const_iterator j = in.begin();
    try
    {
        for (; i != params.end(); ++i, ++j)
        {
            *i = *j;
            ++matches;
        }
    }
    catch (const std::runtime_error& e)
    {
    }
    return matches;
}

const AttributeMap::Parameters& AttributeMap::getParameters(
    const std::string& name) const
{
    assert(name != "");
    typedef std::vector<std::string> Strings;
    Strings tokens;
    boost::split(tokens, name, boost::is_any_of("."));
    assert(tokens.size() != 0);

    const AttributeMap* map = this;
    for (Strings::const_iterator i = tokens.begin(); i != tokens.end() - 1; ++i)
    {
        AttributeMapPtr nested;
        const Parameters& param = map->_impl->getParameters(*i);
        try
        {
            nested = lunchbox::any_cast<AttributeMapPtr>(param[0]);
        }
        catch (...)
        {
            LBTHROW(std::runtime_error("Attribute " + name +
                                       " is not an attribute map"));
        }
        map = nested.get();
    }

    return map->_impl->getParameters(*tokens.rbegin());
}

void AttributeMap::copy(const AttributeMap& map, const std::string& name)
{
    typedef std::vector<std::string> Strings;
    Strings tokens;
    boost::split(tokens, name, boost::is_any_of("."));
    assert(tokens.size() != 0);

    const AttributeMap* source = &map;
    AttributeMap* target = this;
    for (Strings::const_iterator i = tokens.begin(); i != tokens.end() - 1; ++i)
    {
        AttributeMapPtr nested;
        const Parameters& param = source->_impl->getParameters(*i);
        try
        {
            nested = lunchbox::any_cast<AttributeMapPtr>(param[0]);
        }
        catch (...)
        {
            LBTHROW(std::runtime_error("Attribute " + name +
                                       " is not an attribute map"));
        }
        source = nested.get();

        AttributeMapPtr nested2 = (*target)(*i, AttributeMapPtr());
        if (!nested2)
        {
            nested2.reset(new AttributeMap());
            target->set(*i, nested2);
        }
        target = nested2.get();
    }
    const std::string& suffix = *tokens.rbegin();
    target->set(suffix, source->_impl->getParameters(suffix));
}

void AttributeMap::clear()
{
    _impl->_attributes.clear();
    _impl->_hashDirty = true;
}

void AttributeMap::unset(const std::string& name)
{
    _Impl::Attributes::iterator attr = _impl->_attributes.find(name);
    if (attr == _impl->_attributes.end())
        return;

    _impl->disconnectFromUserDefinedObjects(name, attr->second);
    _impl->_attributes.erase(attr);
    _impl->_hashDirty = true;
}

bool AttributeMap::empty() const
{
    return _impl->_attributes.empty();
}

AttributeMap::const_iterator AttributeMap::begin() const
{
    return const_iterator(_impl->_attributes.begin());
}

AttributeMap::const_iterator AttributeMap::end() const
{
    return const_iterator(_impl->_attributes.end());
}

boost::uint32_t AttributeMap::hash() const
{
    _impl->computeHash();
    return _impl->_hashValue;
}

AttributeMap::AttributeProxy AttributeMap::operator()(
    const std::string& name) const
{
    return AttributeProxy(getParameters(name));
}

bool AttributeMap::operator==(const AttributeMap& other) const
{
    /* This should suffice, but a difference in the input string to the
       hashing function (localization, float poing precision, ??) can lead
       to different hashes. */
    if (hash() == other.hash())
        return true;

    _Impl::Attributes::const_iterator attr1 = _impl->_attributes.begin();
    _Impl::Attributes::const_iterator attr2 = other._impl->_attributes.begin();

    for (; (attr1 != _impl->_attributes.end() &&
            attr2 != other._impl->_attributes.end() &&
            attr1->first == attr2->first &&
            _impl->compare(attr1->second, attr2->second));
         ++attr1, ++attr2)
        ;
    return (attr1 == _impl->_attributes.end() &&
            attr2 == other._impl->_attributes.end());
}

void AttributeMap::setExtraDocstring(const std::string& docstring)
{
    _impl->_docstring = docstring;
}

const std::string& AttributeMap::getExtraDocstring() const
{
    return _impl->_docstring;
}

void AttributeMap::disconnectAllSlots()
{
    AttributeChangedSignal().swap(attributeChanged);
    AttributeMapChangingSignal().swap(attributeMapChanging);
    AttributeMapChangedSignal().swap(attributeMapChanged);
}

AttributeMap::TypeRegistrationPtr AttributeMap::getTypeRegistration(
    const std::string& name)
{
    _Impl::TypeRegistry::const_iterator entry =
        _Impl::s_typeNameRegistry.find(name);
    if (entry == _Impl::s_typeNameRegistry.end())
        return TypeRegistrationPtr();
    return entry->second;
}

AttributeMap::TypeRegistrationPtr AttributeMap::getTypeRegistration(
    const std::type_info& type)
{
    _Impl::TypeRegistry::const_iterator entry =
        _Impl::s_typeRegistry.find(type.name());
    if (entry == _Impl::s_typeRegistry.end())
        return TypeRegistrationPtr();
    return entry->second;
}

template <class Archive>
void AttributeMap::serialize(Archive& archive, const unsigned int version)
{
    boost::serialization::split_member(archive, *this, version);
}

template <class Archive>
void AttributeMap::load(Archive& archive, const unsigned int /* version */)
{
    delete _impl; /* This is the easiest way to start clean */
    _impl = new _Impl(*this);
    archive >> *_impl;
}

template <class Archive>
void AttributeMap::save(Archive& archive,
                        const unsigned int /* version */) const
{
    archive << *_impl;
}

void AttributeMap::_registerType(const std::type_info& type,
                                 TypeRegistrationPtr registration)
{
    _Impl::s_typeRegistry.insert(std::make_pair(type.name(), registration));
}

void AttributeMap::_registerType(const std::string& typeName,
                                 TypeRegistrationPtr registration)
{
    _Impl::s_typeNameRegistry.insert(std::make_pair(typeName, registration));
}

uint32_t AttributeMap::_computeBufferHash(const std::string& buffer)
{
    uint32_t hash;
    MurmurHash3_x86_32(buffer.c_str(), buffer.size(), 0, &hash);
    return hash;
}

/* Instantiating the serialization templates for known archive types */
template void AttributeMap::serialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive& archive, const unsigned int version);

template void AttributeMap::serialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive& archive, const unsigned int version);

template void AttributeMap::load<boost::archive::text_iarchive>(
    boost::archive::text_iarchive& archive, const unsigned int version);

template void AttributeMap::save<boost::archive::text_oarchive>(
    boost::archive::text_oarchive& archive, const unsigned int version) const;

template void AttributeMap::serialize<net::DataOStreamArchive>(
    net::DataOStreamArchive& archive, const unsigned int version);

template void AttributeMap::serialize<net::DataIStreamArchive>(
    net::DataIStreamArchive& archive, const unsigned int version);

template void AttributeMap::load<net::DataIStreamArchive>(
    net::DataIStreamArchive& archive, const unsigned int version);

template void AttributeMap::save<net::DataOStreamArchive>(
    net::DataOStreamArchive& archive, const unsigned int version) const;

std::ostream& operator<<(std::ostream& out, const AttributeMap& map)
{
    map._impl->print(out);
    return out;
}
}
}
