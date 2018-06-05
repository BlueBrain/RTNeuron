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
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <rtneuron/AttributeMap.h>
#include <rtneuron/detail/attributeMapTypeRegistration.h>

#define BOOST_TEST_MODULE
#include <boost/test/unit_test.hpp>

#include <sstream>

using namespace bbp::rtneuron;

class UserDefined
{
public:
    UserDefined() {}
    UserDefined(const std::string& value)
        : _value(value)
    {
    }
    bool operator==(const UserDefined& other) const
    {
        return other._value == _value;
    }

    std::string _value;

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar& _value;
    }
};
typedef std::shared_ptr<UserDefined> UserDefinedPtr;

std::ostream& operator<<(std::ostream& out, const UserDefined& x)
{
    out << x._value;
    return out;
}

class UserDefined2
{
public:
    UserDefined2() {}
    UserDefined2(const std::string& value)
        : _value(value)
    {
    }
    bool operator==(const UserDefined2& other) const
    {
        return other._value == _value;
    }

    void setValue(const std::string& value)
    {
        _value = value;
        dirty();
    }

    std::string _value;

    boost::signals2::signal<void()> dirty;

private:
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar& _value;
    }
};
typedef std::shared_ptr<UserDefined2> UserDefined2Ptr;

namespace bbp
{
namespace rtneuron
{
ATTRIBUTE_MAP_IS_PRINTABLE(UserDefined)
}
}

BOOST_AUTO_TEST_CASE(set_value)
{
    AttributeMap map;
    map.set("custom", UserDefined("foo"));
    map.set("custom", UserDefinedPtr(new UserDefined("foo")));
}

BOOST_AUTO_TEST_CASE(simple_get_value)
{
    AttributeMap map;
    map.set("custom", UserDefined("foo"));

    UserDefined x;
    map.get("custom", x);
    BOOST_CHECK_EQUAL(x._value, "foo");

    map.set("custom", UserDefinedPtr(new UserDefined("foo")));
    UserDefinedPtr p;
    map.get("custom", p);
    BOOST_CHECK_EQUAL(p->_value, "foo");
}

BOOST_AUTO_TEST_CASE(get_type_by_proxy)
{
    AttributeMap map;
    map.set("custom", UserDefined("foo"));

    UserDefined x = map("custom");
    BOOST_CHECK_EQUAL(x._value, "foo");

    UserDefined def = map("unexistent", UserDefined("default"));
    BOOST_CHECK_EQUAL(def._value, "default");

    map.set("custom", UserDefinedPtr(new UserDefined("foo")));
    UserDefinedPtr p = map("custom");
    BOOST_CHECK_EQUAL(p->_value, "foo");

    UserDefinedPtr p2 = map("missing", UserDefinedPtr());
    BOOST_CHECK(!p2);
}

BOOST_AUTO_TEST_CASE(hashing)
{
    AttributeMap map;
    AttributeMap::registerType<UserDefined>();
    map.set("custom", UserDefined("foo"));
    uint32_t hash1 = map.hash();
    map.set("custom", UserDefined("bar"));
    uint32_t hash2 = map.hash();
    BOOST_CHECK(hash1 != hash2);
    map.set("custom", UserDefined("foo"));
    BOOST_CHECK_EQUAL(hash1, map.hash());
}

BOOST_AUTO_TEST_CASE(comparisons)
{
    AttributeMap::registerType<UserDefined>();

    AttributeMap map1;
    map1.set("custom", UserDefined("foo"));
    map1.set("custom2", UserDefined("foo"), 10);

    AttributeMap map2;
    map2.set("custom", UserDefined("bar"));
    map2.set("custom2", UserDefined("foo"), 10);

    BOOST_CHECK(map1 != map2);

    map2.set("custom", UserDefined("foo"));
    BOOST_CHECK(map1 == map2);

    AttributeMap::registerType<UserDefinedPtr>();

    map1.set("pointer", UserDefinedPtr(new UserDefined("extra")));
    map2.set("pointer", UserDefinedPtr(new UserDefined("different")));
    BOOST_CHECK(map1 != map2);

    map2.set("pointer", UserDefinedPtr(new UserDefined("extra")));
    BOOST_CHECK(map1 == map2);
}

AttributeMap* nullMap = 0;

class Notifications
{
public:
    void onAttributeMapChanged(const AttributeMap& map, const std::string& name)
    {
        _attributes = &map;
        _name = name;
    }

    void onAttributeChanged(const std::string& name)
    {
        _attributes = nullMap;
        _name = name;
    }

    const AttributeMap* _attributes;
    std::string _name;
};

namespace bbp
{
namespace rtneuron
{
ATTRIBUTE_MAP_HAS_DIRTY_SIGNAL(UserDefined2)
}
}

BOOST_AUTO_TEST_CASE(test_dirty)
{
    AttributeMap::registerType<UserDefined>();
    AttributeMap::registerType<UserDefined2Ptr>();

    AttributeMap map;
    Notifications n1, n2;
    map.attributeMapChanged.connect(
        boost::bind(&Notifications::onAttributeMapChanged, &n1, _1, _2));
    map.attributeChanged.connect(
        boost::bind(&Notifications::onAttributeChanged, &n2, _1));

    UserDefined2Ptr user2(new UserDefined2("foo"));
    map.set("user2", user2);
    BOOST_CHECK_EQUAL(n1._attributes, &map);
    BOOST_CHECK_EQUAL(n1._name, "user2");
    BOOST_CHECK_EQUAL(n2._name, "user2");

    map.set("user1", UserDefined("who cares"));
    BOOST_CHECK_EQUAL(n1._attributes, &map);
    BOOST_CHECK_EQUAL(n1._name, "user1");
    BOOST_CHECK_EQUAL(n2._name, "user1");

    user2->setValue("bar");
    BOOST_CHECK_EQUAL(n1._attributes, &map);
    BOOST_CHECK_EQUAL(n1._name, "user2");
    BOOST_CHECK_EQUAL(n2._name, "user2");

    uint32_t hash = map.hash();
    user2->setValue("foo");
    BOOST_CHECK(hash != map.hash());

    AttributeMapPtr nested(new AttributeMap);
    nested->set("user", user2);
    AttributeMap parent;
    parent.set("nested", nested);
    parent.attributeMapChanged.connect(
        boost::bind(&Notifications::onAttributeMapChanged, &n1, _1, _2));
    parent.attributeChanged.connect(
        boost::bind(&Notifications::onAttributeChanged, &n2, _1));
    hash = parent.hash();
    user2->setValue("bar");
    BOOST_CHECK_EQUAL(n1._attributes, &parent);
    BOOST_CHECK_EQUAL(n1._name, "nested.user");
    BOOST_CHECK_EQUAL(n2._name, "nested.user");
    BOOST_CHECK(hash != parent.hash());
}
