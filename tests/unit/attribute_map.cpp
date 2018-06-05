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

#include "util/MurmurHash3.h"
#include "util/attributeMapHelpers.h"

#include <rtneuron/AttributeMap.h>

#define BOOST_TEST_MODULE
#include <boost/test/unit_test.hpp>

#include <sstream>

using namespace bbp::rtneuron;
using namespace bbp::rtneuron::core::AttributeMapHelpers;

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

    void onAttributeMapChanging(const AttributeMap& map,
                                const std::string& name,
                                const AttributeMap::AttributeProxy&)
    {
        _attributes = &map;
        _name = name;
    }

    const AttributeMap* _attributes;
    std::string _name;
};

BOOST_AUTO_TEST_CASE(simple_get_set_and_conversions)
{
    bool b = false;
    int x = 0;
    float y = 0;
    double z = 0;
    std::string s;

    AttributeMap map;
    map.set("foo", true);
    BOOST_CHECK_EQUAL(map.get("foo", b), 1);
    BOOST_CHECK_EQUAL(b, true);
    BOOST_CHECK_EQUAL((bool)map("foo"), true);
    BOOST_CHECK_EQUAL(map.get("foo", x), 1);
    BOOST_CHECK_EQUAL(x, 1);
    BOOST_CHECK_EQUAL((int)map("foo"), 1);
    BOOST_CHECK_EQUAL(map.get("foo", y), 1);
    BOOST_CHECK_EQUAL(y, 1.f);
    BOOST_CHECK_EQUAL((float)map("foo"), 1.0f);
    BOOST_CHECK_EQUAL(map.get("foo", z), 1);
    BOOST_CHECK_EQUAL(z, 1.0);
    BOOST_CHECK_EQUAL((double)map("foo"), 1.0);

    map.set("foo", 10);
    BOOST_CHECK_EQUAL(map.get("foo", b), 0);
    BOOST_CHECK_THROW((void)(bool)map("foo"), std::runtime_error);
    BOOST_CHECK_EQUAL(map.get("foo", x), 1);
    BOOST_CHECK_EQUAL(x, 10);
    BOOST_CHECK_EQUAL((int)map("foo"), 10);
    BOOST_CHECK_EQUAL(map.get("foo", y), 1);
    BOOST_CHECK_EQUAL(y, 10.0f);
    BOOST_CHECK_EQUAL((float)map("foo"), 10.0f);
    BOOST_CHECK_EQUAL(map.get("foo", z), 1);
    BOOST_CHECK_EQUAL(z, 10.0);
    BOOST_CHECK_EQUAL((double)map("foo"), 10.0);

    map.set("foo", 10.f);
    BOOST_CHECK_EQUAL(map.get("foo", b), 0);
    BOOST_CHECK_THROW((void)(bool)map("foo"), std::runtime_error);
    BOOST_CHECK_EQUAL(map.get("foo", x), 0);
    BOOST_CHECK_THROW((void)(int)map("foo"), std::runtime_error);
    BOOST_CHECK_EQUAL(map.get("foo", y), 1);
    BOOST_CHECK_EQUAL(y, 10.0f);
    BOOST_CHECK_EQUAL(map.get("foo", z), 1);
    BOOST_CHECK_EQUAL(z, 10.0);

    map.set("foo", 10.0);
    BOOST_CHECK_EQUAL(map.get("foo", b), 0);
    BOOST_CHECK_THROW((void)(bool)map("foo"), std::runtime_error);
    BOOST_CHECK_EQUAL(map.get("foo", x), 0);
    BOOST_CHECK_THROW((void)(int)map("foo"), std::runtime_error);
    BOOST_CHECK_EQUAL(map.get("foo", y), 0);
    BOOST_CHECK_THROW((void)(float)map("foo"), std::runtime_error);
    BOOST_CHECK_EQUAL(map.get("foo", z), 1);
    BOOST_CHECK_EQUAL(z, 10.0);

    map.set("foo", std::string("hola"));
    BOOST_CHECK_EQUAL(map.get("foo", s), 1);
    BOOST_CHECK_EQUAL(s, "hola");
    /* Assignment, type conversion, or copy construction to std::string
       don't compile, only the statement below does. */
    std::string s2 = map("foo");
    BOOST_CHECK_EQUAL(s2, "hola");

    map.set("foo", "hola");
    BOOST_CHECK_EQUAL(map.get("foo", s), 1);
    BOOST_CHECK_EQUAL(s, "hola");
}

BOOST_AUTO_TEST_CASE(multi_get_set)
{
    int x = 0, y = 0;
    bool a = false;
    std::string s, t;

    AttributeMap map;
    map.set("bar", 25, true);
    BOOST_CHECK_EQUAL(map.get("bar", x, a), 2);
    BOOST_CHECK_EQUAL(x, 25);
    BOOST_CHECK_EQUAL(a, true);

    map.set("foo", 13, "hi", "good");
    BOOST_CHECK_EQUAL(map.get("bar", a), -1);
    BOOST_CHECK_EQUAL(map.get("foo", a), -1);
    BOOST_CHECK_EQUAL(map.get("foo", a, x, y), 0);
    BOOST_CHECK_EQUAL(x, 25);
    BOOST_CHECK_EQUAL(s, "");
    BOOST_CHECK_EQUAL(y, 0);
    BOOST_CHECK_EQUAL(map.get("foo", x, a, y), 1);
    BOOST_CHECK_EQUAL(x, 13);
    BOOST_CHECK_EQUAL(s, "");
    BOOST_CHECK_EQUAL(y, 0);
    BOOST_CHECK_EQUAL(map.get("foo", x, s, y), 2);
    BOOST_CHECK_EQUAL(x, 13);
    BOOST_CHECK_EQUAL(s, "hi");
    BOOST_CHECK_EQUAL(y, 0);
    BOOST_CHECK_EQUAL(map.get("foo", x, s, t), 3);
    BOOST_CHECK_EQUAL(x, 13);
    BOOST_CHECK_EQUAL(s, "hi");
    BOOST_CHECK_EQUAL(t, "good");

    map.set("bar", -13, "good", "hi", false);
    BOOST_CHECK_EQUAL(map.get("bar", x, s, t, a), 4);
    BOOST_CHECK_EQUAL(x, -13);
    BOOST_CHECK_EQUAL(s, "good");
    BOOST_CHECK_EQUAL(t, "hi");
    BOOST_CHECK_EQUAL(a, false);

    BOOST_CHECK(map("bar")(0) == -13.0);
    BOOST_CHECK(map("bar")(1) == "good");
    BOOST_CHECK(map("bar")(2) == std::string("hi"));
    BOOST_CHECK_THROW(map("bar")(4), std::runtime_error);
    BOOST_CHECK_EQUAL((double)map("unnexisting", 123), 123.0);
    BOOST_CHECK_EQUAL((double)map("unnexisting", 123.0f), 123.0);
    BOOST_CHECK_EQUAL((double)map("unnexisting", 123.0), 123.0);
}

BOOST_AUTO_TEST_CASE(unset)
{
    AttributeMap map;
    map.set("bar", 25, true);
    map.set("foo", "hello");
    map.unset("bar");
    BOOST_CHECK_THROW(map("bar"), std::runtime_error);
    map.unset("foo");
    BOOST_CHECK(map.empty());
    map.unset("unexistent"); /* This is allowed */
}

BOOST_AUTO_TEST_CASE(iterators)
{
    AttributeMap map;
    map.set("foo", "fah");
    map.set("bar", -13.0, "good");
    AttributeMap::const_iterator begin = map.begin(), end = map.end();
    BOOST_CHECK(begin != end);
    /* If the internal storage becomes unsorted this check will fail */
    BOOST_CHECK_EQUAL(begin->first, "bar");
    BOOST_CHECK_EQUAL((double)begin->second, -13.0);
    BOOST_CHECK_EQUAL((double)begin->second(0), -13.0);
    std::string s = begin->second(1);
    BOOST_CHECK_EQUAL(s, std::string("good"));
    BOOST_CHECK(++begin != end);
    BOOST_CHECK_EQUAL(begin->first, "foo");
    BOOST_CHECK(++begin == end);
}

BOOST_AUTO_TEST_CASE(get_set_nested)
{
    int x = 0;
    AttributeMap map;
    AttributeMapPtr nested(new AttributeMap());
    map.set("nested", nested);
    map("nested").set("x", 10);
    BOOST_CHECK_EQUAL(map("nested").get("x", x), 1);
    BOOST_CHECK_EQUAL(x, 10);
    BOOST_CHECK_EQUAL((int)map("nested.x"), 10);
    BOOST_CHECK_EQUAL((int)map("nested")("x"), 10);
    BOOST_CHECK_EQUAL((int)map("nested")("x")(0), 10);
    BOOST_CHECK_EQUAL((int)map("nested")("y", -123)(0), -123);
    BOOST_CHECK(map.getParameters("nested.x") == nested->getParameters("x"));

    /* This has to throw an exception because it's now allowed to assign
       an AttributeMap to a >1-dimensional attribute. */
    BOOST_CHECK_THROW(map.set("bad", 0, nested), std::runtime_error);
}

enum Enum
{
    A = 123,
    B = 283,
    C = 2829,
    D = 0
};

BOOST_AUTO_TEST_CASE(get_set_enum)
{
    Enum a, b, c;
    AttributeMap map;

    map.set("a", A);
    map.set("b", B);
    map.set("abc", A, B, C);
    BOOST_CHECK_EQUAL(map.get("a", a), 1);
    BOOST_CHECK_EQUAL(a, A);
    BOOST_CHECK_EQUAL(map.get("b", b), 1);
    BOOST_CHECK_EQUAL(b, B);
    a = b = c = D;
    BOOST_CHECK_EQUAL(map.get("abc", a, b, c), 3);
    BOOST_CHECK_EQUAL(a, A);
    BOOST_CHECK_EQUAL(b, B);
    BOOST_CHECK_EQUAL(c, C);
}

BOOST_AUTO_TEST_CASE(signals)
{
    AttributeMap map;
    Notifications n1, n2, n3;
    map.attributeMapChanging.connect(
        boost::bind(&Notifications::onAttributeMapChanging, &n1, _1, _2, _3));
    map.attributeMapChanged.connect(
        boost::bind(&Notifications::onAttributeMapChanged, &n2, _1, _2));
    map.attributeChanged.connect(
        boost::bind(&Notifications::onAttributeChanged, &n3, _1));

    map.set("foo", 10, 20);
    BOOST_CHECK_EQUAL(n1._attributes, &map);
    BOOST_CHECK_EQUAL(n2._attributes, &map);
    BOOST_CHECK_EQUAL(n3._attributes, nullMap);
    BOOST_CHECK_EQUAL(n1._name, "foo");
    BOOST_CHECK_EQUAL(n2._name, "foo");
    BOOST_CHECK_EQUAL(n3._name, "foo");

    AttributeMapPtr submap(new AttributeMap());
    map.set("foo", submap);
    submap->set("bar", 10);
    BOOST_CHECK_EQUAL(n1._attributes, &map);
    BOOST_CHECK_EQUAL(n2._attributes, &map);
    BOOST_CHECK_EQUAL(n3._attributes, nullMap);
    BOOST_CHECK_EQUAL(n1._name, "foo.bar");
    BOOST_CHECK_EQUAL(n2._name, "foo.bar");
    BOOST_CHECK_EQUAL(n3._name, "foo.bar");

    map.set("foo", 4.5);
    submap->set("x", 1);
    BOOST_CHECK_EQUAL(n1._name, "foo");
    BOOST_CHECK_EQUAL(n2._name, "foo");
    BOOST_CHECK_EQUAL(n3._name, "foo");
}

BOOST_AUTO_TEST_CASE(print)
{
    AttributeMap map;
    map.set("foo", 10, 20);
    AttributeMapPtr submap(new AttributeMap());
    map.set("bar", submap);
    submap->set("bar", "hola", 10, true);
    submap->set("x", 4.5, -1);

    std::stringstream stream;
    stream << map;
    BOOST_CHECK_EQUAL(stream.str(),
                      "bar:\n"
                      "  bar: hola 10 1\n"
                      "  x: 4.5 -1\n"
                      "foo: 10 20\n");
    std::stringstream stream2;
    map.clear();
    stream2 << map;
    BOOST_CHECK_EQUAL(stream2.str(), "");
}

BOOST_AUTO_TEST_CASE(merge_and_compare)
{
    AttributeMap map1, map2;
    BOOST_CHECK_EQUAL(map1, map2);
    BOOST_CHECK_EQUAL(map1.hash(), map2.hash());

    map1.set("a", 10, "hola");
    map1.set("b", 20, 4.5, false);
    map1.set("n", AttributeMapPtr(new AttributeMap()));
    map1("n").set("a", 100001);
    map1("n").set("b", 100002);

    map2.merge(map1);
    BOOST_CHECK_EQUAL(map1, map2);
    BOOST_CHECK_EQUAL(map1.hash(), map2.hash());

    map1("n").set("b", 100001);
    BOOST_CHECK_NE(map1, map2);
    BOOST_CHECK_NE(map1.hash(), map2.hash());

    map1("n").set("b", 100003);
    BOOST_CHECK_NE(map1.hash(), map2.hash());

    map1("n").set("b", 100002);
    BOOST_CHECK_EQUAL(map1, map2);
    BOOST_CHECK_EQUAL(map1.hash(), map2.hash());

    AttributeMap empty;
    map1.clear();
    BOOST_CHECK_EQUAL(map1, empty);
    BOOST_CHECK_EQUAL(map1.hash(), empty.hash());
    std::stringstream s;
    empty.set("a", false);
    s << empty.hash();
}

BOOST_AUTO_TEST_CASE(copy_attribute)
{
    AttributeMap map1, map2;
    map1.set("a", 10);
    map2.copy(map1, "a");
    BOOST_CHECK_EQUAL(map1, map2);
    AttributeMapPtr nested(new AttributeMap());
    map1.set("nested", nested);
    nested->set("b", 37);
    map2.copy(map1, "nested.b");
    BOOST_CHECK_EQUAL(map1, map2);
}

BOOST_AUTO_TEST_CASE(cycles)
{
    AttributeMapPtr map1(new AttributeMap());
    AttributeMapPtr map2(new AttributeMap());
    AttributeMapPtr map3(new AttributeMap());
    BOOST_CHECK_THROW(map1->set("a", map1), std::runtime_error);
    map2->set("map1", map1);
    BOOST_CHECK_THROW(map1->set("map2", map2), std::runtime_error);
    map1->set("map3", map3);
    BOOST_CHECK_THROW(map3->set("map2", map2), std::runtime_error);
}
