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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>

#define BOOST_TEST_MODULE
#include <boost/test/unit_test.hpp>

#include <sstream>

using namespace bbp::rtneuron;

BOOST_AUTO_TEST_CASE(simple_serialization)
{
    std::stringstream sstream;
    boost::archive::text_oarchive os(sstream);

    AttributeMap original;
    original.set("foo", 10);
    original.set("bar", "hola");
    original.set("brick", false);
    AttributeMapPtr nested(new AttributeMap());
    original.set("nested", nested);
    nested->set("a", 15.2f);
    nested->set("b", 15.2);

    os << original;

    boost::archive::text_iarchive is(sstream);

    AttributeMap recovered;
    is >> recovered;

    BOOST_CHECK_EQUAL(original, recovered);
    std::cout << recovered;
}

enum Test
{
    A = 12122,
    B = 28383
};

BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<Test>, "a:t");

BOOST_AUTO_TEST_CASE(enum_serialization)
{
    std::stringstream sstream;
    boost::archive::text_oarchive os(sstream);
    AttributeMap original;
    original.set("a", A);
    original.set("b", B);
    os << original;

    boost::archive::text_iarchive is(sstream);
    AttributeMap recovered;
    is >> recovered;

    BOOST_CHECK_EQUAL((Test)original("a"), (Test)recovered("a"));
    BOOST_CHECK_EQUAL((Test)original("b"), (Test)recovered("b"));
}

class UserDefined
{
public:
    UserDefined() {}
    explicit UserDefined(const std::string& value)
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

BOOST_CLASS_EXPORT_GUID(lunchbox::Any::holder<UserDefined>, "a:user");

BOOST_AUTO_TEST_CASE(user_defined_type_serialization)
{
    AttributeMap::registerType<UserDefined>();

    std::stringstream sstream;
    boost::archive::text_oarchive os(sstream);
    AttributeMap original;
    original.set("a", UserDefined("a"));
    original.set("b", UserDefined("b"));
    os << original;

    boost::archive::text_iarchive is(sstream);
    AttributeMap recovered;
    is >> recovered;

    BOOST_CHECK(original == recovered);
    UserDefined a = recovered("a");
    UserDefined b = recovered("b");
    BOOST_CHECK_EQUAL(a._value, "a");
    BOOST_CHECK_EQUAL(b._value, "b");
}
