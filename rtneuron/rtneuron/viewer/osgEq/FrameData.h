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

#ifndef RTNEURON_OSGEQ_FRAMEDATA_H
#define RTNEURON_OSGEQ_FRAMEDATA_H

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/lambda/construct.hpp>
#include <boost/lambda/lambda.hpp>

#include <eq/eq.h>

#include <osg/CullSettings>
#include <osg/Matrix>
#include <osgGA/EventQueue>

#include <mutex>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*! \brief Distributed object with per-frame data broadcast from the
  application node to rendering client nodes */
class FrameData : public co::Object
{
public:
    /*--- Public declarations ---*/

    enum AttributeType
    {
        BOOL,
        FLOAT,
        DOUBLE,
        INT,
        INVALID
    };

    class FrameAttributeGetter;
    class FrameAttributeSetter;
    class ObjectAttribute;
    typedef lunchbox::RefPtr<ObjectAttribute> ObjectAttributePtr;

    typedef std::map<co::uint128_t, co::uint128_t> ObjectUUIDCommitMap;

    /*--- Public member attributes */
public:
    bool statistics;

    bool isIdle;

    typedef std::vector<unsigned int> SceneIDs;
    SceneIDs activeScenes;

#ifndef NDEBUG
    bool drawViewportBorder;
#endif

    ObjectUUIDCommitMap commits;

    /*--- Public constructors/destructor */

    FrameData();

    /*--- Public member functions ---*/

    //! Get the value of an attribute set for this frame.
    /*! This class relies on the conversion constructors of
      FrameAttributeGetter to simplify the user code.
      @param name The name of the attribute to search.
      @param attribute A special type that will be constructed from any of
      a bool, int, double or float lvalue where the result will be returned.
      @return True if and only the attribute exists and it is of the type
      passed.
    */
    bool getFrameAttribute(const char* name,
                           FrameAttributeGetter attribute) const;

    //! Sets the value of an attribute to be broadcasted in next commit.
    /*! The attribute will be set for next commit but won't be available for
      subsequent frames unless reset.

      This class relies on the conversion constructors of FrameAttributeGetter
      to simplify the user code.

      @param name The name of the attribute to set.
      @param attribute An object constructed from one of a bool, int, double
      or float rvalue.
    */
    void setFrameAttribute(const char* name,
                           const FrameData::FrameAttributeSetter& attribute);

    //! Get and object attribute with the given name
    /* @param name
       @return a valid pointer if the object exists or 0 if it doesn't
    */
    ObjectAttribute* getFrameAttribute(const char* name);

    template <typename T>
    T* getFrameAttribute(const char* name)
    {
        return dynamic_cast<T*>(getFrameAttribute(name));
    }

    //! Sets an object to be broadcasted in next commit.
    /*! The object will be set for next commit but won't be available for
       subsequent frames unless reset.

       @param name The name of the object attribute to set.
       @param object An object deriving from ObjectAttribute and registered
       for proper deserialization.
     */
    void setFrameAttribute(const char* name, ObjectAttribute* object);

protected:
    /*--- Protected member functions ---*/

    virtual ChangeType getChangeType() const { return INSTANCE; }
    virtual void getInstanceData(co::DataOStream& out);

    virtual void applyInstanceData(co::DataIStream& in);

protected:
    std::mutex _mutex;

    typedef std::map<std::string, FrameAttributeSetter> AttributeMap;
    AttributeMap _nextFrameAttributes;
    AttributeMap _currentFrameAttributes;

    typedef std::map<std::string, ObjectAttributePtr> ObjectAttributeMap;
    ObjectAttributeMap _nextFrameObjectAttributes;
    ObjectAttributeMap _currentFrameObjectAttributes;
};

/*! \brief Helper class for type-safe polymorphic serialization of some POD
  types.

  \sa FrameAttributeGetter
*/
class FrameData::FrameAttributeSetter
{
    friend co::DataOStream& operator<<(co::DataOStream&, FrameAttributeSetter&);

    friend co::DataIStream& operator>>(co::DataIStream&, FrameAttributeSetter&);

    friend class FrameAttributeGetter;

public:
    FrameAttributeSetter()
        : _type(INVALID)
    {
    }

    FrameAttributeSetter(bool value)
    {
        _type = BOOL;
        _value._bool = value;
    }

    FrameAttributeSetter(float value)
    {
        _type = FLOAT;
        _value._float = value;
    }

    FrameAttributeSetter(double value)
    {
        _type = DOUBLE;
        _value._double = value;
    }

    FrameAttributeSetter(int value)
    {
        _type = INT;
        _value._int = value;
    }

protected:
    AttributeType _type;
    union {
        bool _bool;
        float _float;
        double _double;
        int _int;
    } _value;
};

/*! \brief Helper class for type-safe polymorphic retrieval of POD
  frame attributes.

  \sa FrameAttributeSetter
*/
class FrameData::FrameAttributeGetter
{
public:
    FrameAttributeGetter(bool& value)
    {
        _type = BOOL;
        _value._bool = &value;
    }

    FrameAttributeGetter(float& value)
    {
        _type = FLOAT;
        _value._float = &value;
    }

    FrameAttributeGetter(double& value)
    {
        _type = DOUBLE;
        _value._double = &value;
    }

    FrameAttributeGetter(int& value)
    {
        _type = INT;
        _value._int = &value;
    }

    FrameAttributeGetter& operator=(const FrameAttributeSetter& rhs);

protected:
    AttributeType _type;
    union {
        bool* _bool;
        float* _float;
        double* _double;
        int* _int;
    } _value;
};

//! Base class for frame attributes of generic non-POD type
/*! Derived classes must implement the serialization and deserialization methods
  In order to be used they also must register themselves to be recognized
  by the deserialization routines. A requirement for registration is that
  the class has a default constructor.
*/
class FrameData::ObjectAttribute : public lunchbox::Referenced
{
    /* Declarations */
public:
    typedef std::map<std::string, boost::function0<ObjectAttribute*>>
        ConstructorMap;

protected:
    //! Helper class for registration of ObjectAttribute derived classes
    template <typename Derived>
    class RegisterProxy
    {
    public:
        RegisterProxy() { registerType<Derived>(); }
    };

    /* Destructor */
protected:
    ~ObjectAttribute() {}
    /* Member functions */
protected:
    //! Registration function
    /* This function stores a functor associated to the implementation
       dependent name of the template parameter. This functor invokes the
       default constructor using the operator new */
    template <typename Derived>
    static void registerType()
    {
        /* Storing constructor functor for this type in constructors table */
        boost::function0<ObjectAttribute*> constructor =
            boost::bind<ObjectAttribute*>(boost::lambda::new_ptr<Derived>());
        registerType_(typeid(Derived).name(), constructor);
    }

public:
    void serialize(co::DataOStream& out)
    {
        out << std::string(typeid(*this).name());
        serializeImplementation(out);
    }

    static ObjectAttributePtr deserialize(co::DataIStream& in);

protected:
    virtual void serializeImplementation(co::DataOStream& out) = 0;
    virtual void deserializeImplementation(co::DataIStream& in) = 0;

private:
    static void registerType_(const std::string& name,
                              const boost::function0<ObjectAttribute*>& ctor);
};

#define FRAMEOBJECTATTRIBUTEWRAPPER_ACCESS_OPERATORS(ref, pointer) \
    operator T&() { return ref; }                                  \
    operator const T&() const { return ref; }                      \
    operator T*() { return pointer; }                              \
    operator const T*() const { return pointer; }                  \
    T& operator*() { return ref; }                                 \
    const T& operator*() const { return ref; }                     \
    T* operator->() { return pointer; }                            \
    const T* operator->() const { return pointer; }
//! Helper class to wrap an arbitrary object as an ObjectAttribute.
/**
   The object is required to be copy constructable and a copy will
   be internally stored. Don't use with objects for which the identity
   of the object (e.g. the memory address) is important.
*/
template <typename T>
class FrameObjectAttributeWrapper : public FrameData::ObjectAttribute
{
    /* Declarations */
public:
    typedef RegisterProxy<FrameObjectAttributeWrapper<T>> RegisterProxy;

    /* Constructors */
public:
    FrameObjectAttributeWrapper() {}
    FrameObjectAttributeWrapper(const T& object)
        : _object(object)
    {
    }

    /* Destructor */
protected:
    ~FrameObjectAttributeWrapper() {}
    /* Member functions */
public:
    FRAMEOBJECTATTRIBUTEWRAPPER_ACCESS_OPERATORS(_object, &_object)

protected:
    virtual void serializeImplementation(co::DataOStream& out)
    {
        _object.serialize(out);
    }

    virtual void deserializeImplementation(co::DataIStream& in)
    {
        _object.deserialize(in);
    }

    /* Member attributes */
protected:
    T _object;
};

/*! \brief Partial specialization of FrameObjectAttributeWrapper to store the
  actual object with a ref_ptr instead of by value.
*/
template <typename T>
class FrameObjectAttributeWrapper<osg::ref_ptr<T>>
    : public FrameData::ObjectAttribute
{
    /* Declarations */
public:
    typedef FrameObjectAttributeWrapper<osg::ref_ptr<T>> this_type;
    typedef RegisterProxy<this_type> RegisterProxy;

    /* Constructors */
public:
    FrameObjectAttributeWrapper()
        : _object(new T())
    {
    }

    FrameObjectAttributeWrapper(T* object)
        : _object(object)
    {
    }

    /* Destructor */
protected:
    ~FrameObjectAttributeWrapper() {}
    /* Member functions */
public:
    FRAMEOBJECTATTRIBUTEWRAPPER_ACCESS_OPERATORS(*_object, _object.get());

protected:
    virtual void serializeImplementation(co::DataOStream& out)
    {
        _object->serialize(out);
    }

    virtual void deserializeImplementation(co::DataIStream& in)
    {
        _object = new T();
        _object->deserialize(in);
    }

    /* Member attributes */
protected:
    /* Definitions needs to be done somewhere else */
    static RegisterProxy s_registry;

    osg::ref_ptr<T> _object;
};

#undef FRAMEOBJECTATTRIBUTEWRAPPER_ACCESS_OPERATORS
}
}
}

#endif
