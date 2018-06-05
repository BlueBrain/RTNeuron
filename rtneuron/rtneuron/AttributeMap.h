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

#ifndef RTNEURON_API_ATTRIBUTEMAP_H
#define RTNEURON_API_ATTRIBUTEMAP_H

#include <lunchbox/any.h>

#include <boost/cstdint.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/signals2/signal.hpp>

#include <iostream>
#include <memory>
#include <string>

namespace boost
{
namespace serialization
{
class access;
}
}

namespace bbp
{
namespace rtneuron
{
class AttributeMap;
typedef std::shared_ptr<AttributeMap> AttributeMapPtr;

/**
   \if pybind
   An key-value storage with additional capabilities in the native code side.

   An attribute map is a table container that stores key-value pairs where
   the keys are strings and the values are scalars of type bool, int, float,
   string, wrapped enums and AttributeMap or Python lists of scalars of type
   bool, int, float, string and wrapped enums.

   Other wrapped types can be used as values only if their documentation
   states so.

   @note An AttributeMap cannot be nested as part of a list of values.

   The attribute keys are presented as regular attributes in Python. This
   class defines special '\__setattr__' and '\__getattr__' methods to handle
   attribute read/writes and translation of types to/from the native code.
   Writting to a non existing attribute creates it. Accessing a non existing
   attribute raises a ValueError exception. Trying to set an attribute with
   an unsupported type raises a KeyError exception.

   Examples:
   \verbatim
   a = AttributeMap()               # Create a new attribute map.
   a.x = 10.0                       # Set a new attribute.
   print a.x + 3.3                  # Retrieve the attribute value.
   a.x = [1, "hi", False]           # Resetting the previous attribute.
   a.nested = AttributeMap()        # Nesting an attribute map.
   a.nested.x = [1, AttributeMap()] # raises, AttributeMap cannot be in a list.
   a.nested.x = dict                # raises, invalid type in assignment.
   a.nested.colors = ColorMap()     # OK if ColorMap has been made available
                                    # to AttributeMap in the wrapping.
   a.colors = [ColorMap(), 1, "a"]  # If the above works, this will also do.
   \endverbatim

   Native code objects that hold attribute maps can provide extra handles
   for attribute modification. This implies that trying to set attribute
   names/values unsupported by a holder can also raise exceptions.

   Tab completion inside an IPython shell works by the internal redefinition
   of the '\__dir__' method. The string conversion operator is also defined to
   print the attributes and their values.
   \else
   An attribute map that stores arbitrary values using strings as keys.

   Despite nothing prevents it, attribute names may not contain dots because
   they are used to report changes in nested attribute maps.

   The Python wrapping of this class relies on special definitions of
   '\__setattr__' and '\__getattr__' so there is no direct translation
   from the C++ interface to Python.

   \endif
*/
class AttributeMap
{
public:
    /*--- Public declarations ---*/

    friend std::ostream& operator<<(std::ostream& out, const AttributeMap& map);
    friend class boost::serialization::access;

    typedef std::vector<lunchbox::Any> Parameters;

    /** @internal */
    class OutWrapper;
    /** @internal */
    class AttributeProxy;
    /** @internal */
    class ParameterProxy;
    /** @internal */
    class BaseTypeRegistration;
    /** @internal */
    typedef std::shared_ptr<BaseTypeRegistration> TypeRegistrationPtr;

    /**
       lunchbox::Any wrapper.

       This class is used to force automatic conversion from const char *
       to std::string at AttributeMap::set argument passing.
    */
    struct Parameter
    {
        template <typename T>
        Parameter(const T& v)
            : value(v)
        {
        }

        Parameter(const char* v)
            : value(std::string(v))
        {
        }

        Parameter(AttributeMap* map)
            : value(AttributeMapPtr(map))
        {
        }

        operator const lunchbox::Any&() const { return value; }
        lunchbox::Any value;
    };

    /** @internal */
    typedef void AttributeMapChangedSignature(const AttributeMap&,
                                              const std::string&);
    typedef boost::signals2::signal<AttributeMapChangedSignature>
        AttributeMapChangedSignal;

    /** @internal */
    typedef void AttributeChangedSignature(const std::string&);
    typedef boost::signals2::signal<AttributeChangedSignature>
        AttributeChangedSignal;

    /** @internal */
    typedef void AttributeMapChangingSignature(const AttributeMap&,
                                               const std::string&,
                                               const AttributeProxy&);
    typedef boost::signals2::signal<AttributeMapChangingSignature>
        AttributeMapChangingSignal;

    /*--- Public constructors/destructor ---*/

    AttributeMap();
    AttributeMap(const AttributeMap& other); //!< Performs deep copy
    AttributeMap(AttributeMap&& other);
    ~AttributeMap();

    /*--- Public wrapped member functions ---*/

    AttributeMap& operator=(AttributeMap&& other);
    AttributeMap& operator=(const AttributeMap&) = delete;

    /**
       Deep copy the attributes from another attribute map.

       Attributes with the same name will be replaced.

       Attribute changing and changed signals are only emitted for attributes
       at this level (i.e. the copy of nested attributes won't emit any
       signal from this object)
    */
    void merge(const AttributeMap& origin);

    /**
       \ifnot pybind
       @name Attribute setters

       Nested attribute map parameters can only be set in the
       1-parameter case, in all the other cases an exception will be throw.
       (This restriction is motivated by the attributeChaging and
       attributeChanged signals. If an attribute map can be part of an
       arbitrary arity attribute, the naming scheme to report which
       attribute has changed needs to get more complex)
       \endif
     */
    ///@{
    void set(const std::string& name, const Parameter& p1);
    void set(const std::string& name, const Parameter& p1, const Parameter& p2);
    void set(const std::string& name, const Parameter& p1, const Parameter& p2,
             const Parameter& p3);
    void set(const std::string& name, const Parameter& p1, const Parameter& p2,
             const Parameter& p3, const Parameter& p4);
    void set(const std::string& name, const Parameter& p1, const Parameter& p2,
             const Parameter& p3, const Parameter& p4, const Parameter& p5);
    void set(const std::string& name, const Parameters& params);
    ///@}

    /** @internal To easily transfer attributes between attribute maps */
    void set(const std::string& name,
             const AttributeMap::AttributeProxy& params);

    /**
       \ifnot pybind
       @name Attribute getters

       These functions return the number of parameters that could be matched
       and extracted before the first conversion error occurs, or -1 if the
       attribute does not exists or the number of function parameters does
       not match the number of attribute parameters.
       \endif
    */
    ///@{
    int get(const std::string& name, OutWrapper p1) const;
    int get(const std::string& name, OutWrapper p1, OutWrapper p2) const;
    int get(const std::string& name, OutWrapper p1, OutWrapper p2,
            OutWrapper p3) const;
    int get(const std::string& name, OutWrapper p1, OutWrapper p2,
            OutWrapper p3, OutWrapper p4) const;
    int get(const std::string& name, OutWrapper p1, OutWrapper p2,
            OutWrapper p3, OutWrapper p4, OutWrapper p5) const;
    int get(const std::string& name, std::vector<OutWrapper>& params) const;
    ///@}

    /*--- Public C++ only member fuctions ---*/

    /**
       Returns the list of parameters assigned to an attribute.

       This function resolves attribute map nesting as
       AttributeMap::operator(). For that purpose the attribute name is
       tokenized using '.' as a separator and all the tokens but the last are
       considered to be nested AttributeMaps.

       Throws a runtime exception if the attribute does not exist.
    */
    const Parameters& getParameters(const std::string& name) const;

    /**
       Copies an attribute from one map into this.

       This function resolves attribute map nesting as
       AttributeMap::operator().

       Throws a runtime exception if the attribute does not exist.
    */
    void copy(const AttributeMap& map, const std::string& attribute);

    //! Removes all stored attributes.
    void clear();

    /** @internal
        Removes an attribute from the map.
    */
    void unset(const std::string& name);

    /** Returns true if no attributes are stored.
     */
    bool empty() const;

    /**
       Input iterator that iterates over the attributes.

       Being i a const_iterator, i->first is the attribute and the result
       of i->second supports the same operations than the result from
       AttributeMap::operator().
       \todo The iterator is not really const as AttributeProxy allows
       the modification of nested attribute maps.
     */
    class const_iterator;

    /** @internal */
    const_iterator begin() const;

    /** @internal */
    const_iterator end() const;

    /** @internal */
    uint32_t hash() const;

    /**
       Quick attribute accessor with automatic type conversion and exception
       throwing if the attribute doesn't exists or there's a type mismatch.

       Being m an AtributeMap and s and string, the result of m(s) is a model
       of an object that supports the following operations:
       - Automatic conversion of the first parameter of the retrieved
         parameter to any type. An exception is thrown if the parameter is not
         convertible to the type of the rvalue.
         \verbatim
         m.set("test", 10);
         int a = m("test"); // OK
         bool b = m("test"); // throws
         b = m("nonexistent", false); // OK
         \endverbatim
       - Indexing with operator()(unsigned int) to access the nth parameter.
         \verbatim
         m.set("test", 10, 20);
         int a = m("test")(0); // a == 10
         \endverbatim
       - Indexing with operator()(const string& name) to access an attribute
         of a nested attribute map. This operation will throw if the held
         attribute is not an AttributeMap. The result of this indexing is an
         object that supports all the operations listed here.
         Nested attributes can be accessed using dots (this nested access
         using . is only performed by this operation).
         \verbatim
         m.set("test", AttributeMapPtr(new AttributeMap));
         int a = m("test")("foo", 10); // OK
         int a = m("test.foo", 10); // OK
         \endverbatim
       - get and set methods. Will throw if the parameter accessed is not an
         attribute map.
         \verbatim
         int a = m("test").get("foo", 10);
         m("not_a_map").set("foo", 10); // throws
         \endverbatim
    */
    AttributeProxy operator()(const std::string& name) const;

    /**
       Overloaded version of the function above in which a default value
       can be provided.

       This default value is encapsulated by the return proxy when the
       requested attribute does not exist,
     */
    template <typename T>
    inline AttributeProxy operator()(const std::string& name,
                                     const T& def) const;

    /**
       @internal
       Signal emitted when an attribute has been changed.

       The attribute map itself and a string with the name of the modified
       attributes are passed to the slots.

       When the attribute belongs a nested attribute map the attribute name
       is compound using dots.
     */
    AttributeMapChangedSignal attributeMapChanged;

    /**
       Signal emitted when an attribute has been changed.

       The name of the changed attribute is passed as the signal parameter.
     */
    AttributeChangedSignal attributeChanged;

    /**
       @internal
       Signal emitted when an attribute is about to change.

       The attribute map itself, a string with the name of the attribute and
       a vector with the parameters is passed to the slot.

       When the attribute belongs a nested attribute map the attribute name
       is compound using dots.

       See AttributeMap::operator() for details about how to access the
       parameters to be set from the AttributeProxy.
     */
    AttributeMapChangingSignal attributeMapChanging;

    bool operator==(const AttributeMap& attributes) const;
    bool operator!=(const AttributeMap& attributes) const
    {
        return !operator==(attributes);
    }

    /**
       @internal
       Returns the additional docstring
       @sa setExtraDocstring
     */
    const std::string& getExtraDocstring() const;

    /**
       @internal
       Sets an extra docstring to be concatenated to the Python wrapping
       documentation to include details specific about this attribute map.

       The docstring is neither considered in the hash and comparison operators
       nor it is serialized.
     */
    void setExtraDocstring(const std::string& docstring);

    /** @internal
        Disconnects all slots, ensuring the callables are referenced no more.
        It doesn't consider any side effects related to nesting.
    */
    void disconnectAllSlots();

    /**
       @internal
       Registers a type to be fully supported by AttributeMap.

       This support includes:
       - Hash value calculation (AttributeMap::hash() throws when it finds an
         unregistered type).
       - Optional serialization support.
       - Optional to/from Python conversions.

       The registered type must:
       - Be default constructible
       - Provide a public operator==
       - Implement the serialize function as required by boost::serialization.
       An alternative for non-copy constructible types is to register a
       std::shared_ptr<Type> instead.

       For full serialization support of a type named Type,
       BOOST_CLASS_EXPORT(Type) and
       BOOST_CLASS_EXPORT(lunchbox::Any::holder<Type>)
       must be provided. Additional template instantiations of the serialize
       function may be required depending on the usage.

       When a user defined type is modified, the attribute map will not detect
       the change and trigger the attributeChanged signal by default. The
       trigger may be enabled if the registered type provides a dirty signal
       with void() signature. To enable it use the macro
       ATTRIBUTE_MAP_HAS_DIRTY_SIGNAL with the type as argument (no shared_ptr
       if registering a pointer type). The macro expands to a struct
       declaration and must be inside the bbp::rtneuron namespace. The macro
       is declared in the header rtneuron/detail/attributeMapTypeRegistration.h.

       @param pythonName Takes the names with which the registered type has
              been wrapped in Python. This function will enable the wrapping
              of AttributeMap to perform the to/from Python conversion of
              the registered type when used as an attribute parameter.
              If empty the conversions won't be available.
    */
    template <typename T>
    inline static void registerType(const std::string& pythonName = "");

    //! @internal Used only by the Python wrapping
    static TypeRegistrationPtr getTypeRegistration(const std::string& name);
    //! @internal Used only by the Python wrapping
    static TypeRegistrationPtr getTypeRegistration(const std::type_info& type);

private:
    /*--- Private declarations ---*/
    template <typename T>
    class TypeRegistration;

    /*--- Private member attributes ---*/

    class _Impl;
    _Impl* _impl;

    /*--- Private member functions ---*/

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version);

    template <class Archive>
    void load(Archive& ar, const unsigned int version);
    template <class Archive>
    void save(Archive& ar, const unsigned int version) const;

    static void _registerType(const std::type_info& type,
                              TypeRegistrationPtr registration);
    static void _registerType(const std::string& pythonName,
                              TypeRegistrationPtr registration);

    static uint32_t _computeBufferHash(const std::string& buffer);
};
}
}
#include "AttributeMap.ipp"
#endif
