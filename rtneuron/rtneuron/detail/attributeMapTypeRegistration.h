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

#ifndef RTNEURON_API_DETAIL_ATTRIBUTEMAPTYPEREGISTRATION_H
#define RTNEURON_API_DETAIL_ATTRIBUTEMAPTYPEREGISTRATION_H

#include "../AttributeMap.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/utility/enable_if.hpp>

namespace bbp
{
namespace rtneuron
{
/**
   @internal
   User defined type registration for AttributeMap

   This class contains a collection of virtual functions that make it possible
   to operate on lunchbox::Any objects without knowing the actual type that
   it stores and hiding the necessary any_casts internally.

   The functions used by the Python wrapping take void* instead of PyObject
   to isolate the core library from the Python headers and libraries.
   The definition of the internal function is an empty implementation unless
   Python.h has been included before this header. This seems the only way
   of not making this header depend on Python except where needed. Since the
   functions are inline template functions this shouldn't cause any double
   definition error.
*/
class AttributeMap::BaseTypeRegistration
{
public:
    template <typename T>
    struct isPrintable
    {
        const static bool value = false;
    };

    template <typename T>
    struct hasDirtySignal
    {
        const static bool value = false;
    };

    typedef boost::signals2::signal<void()> DirtySignal;

    virtual ~BaseTypeRegistration() {}
    virtual bool compare(const lunchbox::Any& left,
                         const lunchbox::Any& right) const = 0;
    virtual uint32_t hash(const lunchbox::Any& value) const = 0;

    template <typename Callback>
    void connectDirtySignal(const Callback& callback, lunchbox::Any& value)
    {
        DirtySignal* signal = getDirtySignal(value);
        if (signal)
            signal->connect(callback);
    }

    template <typename Callback>
    void disconnectDirtySignal(const Callback& callback, lunchbox::Any& value)
    {
        DirtySignal* signal = getDirtySignal(value);
        if (signal)
            signal->disconnect(callback);
    }

    virtual void print(std::ostream& out, const lunchbox::Any& value) const = 0;

    /** Takes a PyObject* and returns a lunchbox::Any containing the C++
        value of the registered type if the conversion is possible.
        Throws otherwise. */
    virtual lunchbox::Any getFromPython(const void* pyobject) const = 0;
    /** Returns a new reference to the Python object that wraps the object
        contained by value */
    virtual void* convertToPython(const lunchbox::Any& value) const = 0;

protected:
    virtual DirtySignal* getDirtySignal(lunchbox::Any& value) const = 0;

    template <typename T>
    typename boost::enable_if<isPrintable<T>>::type _print(std::ostream& out,
                                                           const T& value) const
    {
        out << value;
    }

    template <typename T>
    typename boost::disable_if<isPrintable<T>>::type _print(
        std::ostream& out, const T& value) const
    {
        out << lunchbox::className(*((T*)0)) << ' ' << &value;
    }

    template <typename T>
    typename boost::enable_if<hasDirtySignal<T>, DirtySignal*>::type
        _getDirtySignal(T& value) const
    {
        return &value.dirty;
    }

    template <typename T>
    typename boost::disable_if<hasDirtySignal<T>, DirtySignal*>::type
        _getDirtySignal(T&) const
    {
        return 0;
    }

    template <typename T>
    inline lunchbox::Any _getFromPython(const void* pyobject) const;

    template <typename T>
    inline void* _convertToPython(const lunchbox::Any& pyobject) const;
};

#define ATTRIBUTE_MAP_HAS_DIRTY_SIGNAL(Type)                        \
    template <>                                                     \
    struct AttributeMap::BaseTypeRegistration::hasDirtySignal<Type> \
    {                                                               \
        static const bool value = true;                             \
    };

#define ATTRIBUTE_MAP_IS_PRINTABLE(Type)                         \
    template <>                                                  \
    struct AttributeMap::BaseTypeRegistration::isPrintable<Type> \
    {                                                            \
        static const bool value = true;                          \
    };

template <typename T>
class AttributeMap::TypeRegistration : public AttributeMap::BaseTypeRegistration
{
public:
    bool compare(const lunchbox::Any& left,
                 const lunchbox::Any& right) const final
    {
        try
        {
            return (lunchbox::any_cast<T>(left) ==
                    lunchbox::any_cast<T>(right));
        }
        catch (...)
        {
            return false;
        }
    }

    uint32_t hash(const lunchbox::Any& value) const final
    {
        /* User defined objects are required to be serializable, so we
           compute the hash from the serialization */
        std::stringstream buffer;
        boost::archive::binary_oarchive os(buffer);
        os& const_cast<T&>(lunchbox::any_cast<const T&>(value));
        return AttributeMap::_computeBufferHash(buffer.str());
    }

    DirtySignal* getDirtySignal(lunchbox::Any& value) const
    {
        return _getDirtySignal(lunchbox::any_cast<T&>(value));
    }

    void print(std::ostream& out, const lunchbox::Any& value) const
    {
        _print(out, lunchbox::any_cast<const T&>(value));
    }
    lunchbox::Any getFromPython(const void* pyobject) const final
    {
        return _getFromPython<T>(pyobject);
    }
    void* convertToPython(const lunchbox::Any& value) const final
    {
        return _convertToPython<T>(value);
    }
};

template <typename T>
class AttributeMap::TypeRegistration<std::shared_ptr<T>>
    : public AttributeMap::BaseTypeRegistration
{
private:
    typedef std::shared_ptr<T> Tptr;

public:
    bool compare(const lunchbox::Any& left,
                 const lunchbox::Any& right) const final
    {
        try
        {
            return (*lunchbox::any_cast<Tptr>(left) ==
                    *lunchbox::any_cast<Tptr>(right));
        }
        catch (...)
        {
            return false;
        }
    }

    uint32_t hash(const lunchbox::Any& value) const final
    {
        /* User defined objects are required to be serializable, so we
           compute the hash from the serialization */
        std::stringstream buffer;
        boost::archive::binary_oarchive os(buffer);
        os&* lunchbox::any_cast<Tptr>(value);
        return AttributeMap::_computeBufferHash(buffer.str());
    }

    DirtySignal* getDirtySignal(lunchbox::Any& value) const
    {
        return _getDirtySignal(*lunchbox::any_cast<Tptr>(value));
    }

    void print(std::ostream& out, const lunchbox::Any& value) const
    {
        _print(out, *lunchbox::any_cast<const Tptr&>(value));
    }
    lunchbox::Any getFromPython(const void* pyobject) const final
    {
        return _getFromPython<Tptr>(pyobject);
    }
    void* convertToPython(const lunchbox::Any& value) const final
    {
        return _convertToPython<Tptr>(value);
    }
};

template <typename T>
lunchbox::Any AttributeMap::BaseTypeRegistration::_getFromPython(
    const void* pyobject) const
{
#ifdef PyObject_HEAD
    boost::python::extract<std::shared_ptr<T>> pointer((PyObject*)pyobject);
    if (pointer.check())
        return lunchbox::Any((std::shared_ptr<T>)pointer);
    boost::python::extract<T> rvalue((PyObject*)pyobject);
    assert(rvalue.check());
    return (T)rvalue;
#else
    (void)pyobject;
    LBUNIMPLEMENTED;
    return lunchbox::Any();
#endif
}

template <typename T>
void* AttributeMap::BaseTypeRegistration::_convertToPython(
    const lunchbox::Any& value) const
{
#ifdef PyObject_HEAD
    boost::python::object object(lunchbox::any_cast<const T&>(value));
    boost::python::incref(object.ptr());
    return object.ptr();
#else
    (void)value;
    LBUNIMPLEMENTED;
    return 0;
#endif
}

template <typename T>
void AttributeMap::registerType(const std::string& typeName)
{
    TypeRegistrationPtr registration(new TypeRegistration<T>());
    AttributeMap::_registerType(typeid(T), registration);
    if (!typeName.empty())
        AttributeMap::_registerType(typeName, registration);
}
}
}

#endif
