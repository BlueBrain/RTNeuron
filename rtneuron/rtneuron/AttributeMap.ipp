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

#ifndef RTNEURON_RTNEURON_ATTRIBUTE_MAP_INC
#define RTNEURON_RTNEURON_ATTRIBUTE_MAP_INC

#include <sstream>
#include <typeinfo>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

namespace bbp
{
namespace rtneuron
{
namespace detail
{
template <typename From, typename To>
struct is_convertible : public boost::false_type
{
};

template <typename T>
struct is_convertible<T, T> : public boost::true_type
{
};
template <typename T>
struct is_convertible<const T&, T> : public boost::true_type
{
};
template <typename T>
struct is_convertible<T&, T> : public boost::true_type
{
};

template <>
struct is_convertible<bool, int> : public boost::true_type
{
};
template <>
struct is_convertible<bool, unsigned int> : public boost::true_type
{
};
template <>
struct is_convertible<bool, float> : public boost::true_type
{
};
template <>
struct is_convertible<bool, double> : public boost::true_type
{
};

template <>
struct is_convertible<int, float> : public boost::true_type
{
};
template <>
struct is_convertible<int, double> : public boost::true_type
{
};

template <>
struct is_convertible<unsigned int, float> : public boost::true_type
{
};
template <>
struct is_convertible<unsigned int, double> : public boost::true_type
{
};

template <>
struct is_convertible<float, double> : public boost::true_type
{
};

template <>
struct is_convertible<AttributeMapPtr, bool> : public boost::true_type
{
};

template <typename From, typename To = From,
          bool is_convertible_ = is_convertible<From, To>::value>
struct convert;

template <typename From, typename To>
struct convert<From, To, true>
{
    void operator()(const lunchbox::Any& in, To& out) const
    {
        try
        {
            out = To(lunchbox::any_cast<const From&>(in));
        }
        catch (lunchbox::bad_any_cast&)
        {
            std::stringstream s;
            s << "Can't convert from " << lunchbox::className(From()) << " to "
              << lunchbox::className(To());
            throw std::runtime_error(s.str());
        }
    }
};

template <typename From, typename To>
struct convert<From, To, false>
{
    void operator()(const lunchbox::Any&, To&) const
    {
        std::stringstream s;
        s << "Can't convert from " << lunchbox::className(From()) << " to "
          << lunchbox::className(To());
        throw std::runtime_error(s.str());
    }
};
}

class AttributeMap::OutWrapper
{
public:
    template <typename T>
    friend void converter(void*, const lunchbox::Any&);

    OutWrapper()
        : _ref(0)
    {
    }

    /* Any type is accepted but full support is only given for bool, int,
       unsigned int, float, double, std::string and  AttributeMapPtr.
       For other types, automatic conversions are not supported, even if
       the proper conversion operators exist. */
    template <typename T>
    OutWrapper(T& value)
        : _converter(converter<T>)
        , _ref(&value)
    {
    }

    OutWrapper& operator=(const lunchbox::Any& in)
    {
        assert(_ref != 0);
        /* Exception thrown if conversion not possible */
        _converter(_ref, in);
        return *this;
    }

protected:
    template <typename T>
    static void converter(void* ref, const lunchbox::Any& param)
    {
        using namespace detail;

        T* out = static_cast<T*>(ref);
        if (param.type() == typeid(bool))
            convert<bool, T>()(param, *out);
        else if (param.type() == typeid(int))
            convert<int, T>()(param, *out);
        else if (param.type() == typeid(unsigned int))
            convert<unsigned int, T>()(param, *out);
        else if (param.type() == typeid(float))
            convert<float, T>()(param, *out);
        else if (param.type() == typeid(double))
            convert<double, T>()(param, *out);
        else if (param.type() == typeid(std::string))
            convert<std::string, T>()(param, *out);
        else if (param.type() == typeid(AttributeMapPtr))
        {
            /* This is not strictly necessary but enables the convenient
               implicit conversion from AttributeMapPtr to bool that
               std::shared_ptr provides. */
            convert<AttributeMapPtr, T>()(param, *out);
        }
        else
        {
            try
            {
                convert<T, T>()(param, *out);
            }
            catch (const std::runtime_error&)
            {
                std::stringstream s;
                s << "Can't convert to " << lunchbox::className(T());
                throw std::runtime_error(s.str());
            }
        }
    }

    boost::function<void(void*, const lunchbox::Any&)> _converter;
    void* const _ref;
};

class AttributeMap::ParameterProxy
{
public:
    template <typename T>
    explicit ParameterProxy(const T& default_)
        : _default(default_)
        , _parameter(0)
    {
    }

    explicit ParameterProxy(const char* default_)
        : _default(std::string(default_))
        , _parameter(0)
    {
    }

    explicit ParameterProxy(const lunchbox::Any& parameter)
        : _parameter(&parameter)
    {
    }

    template <typename T>
    operator const T() const
    {
        using namespace detail;
        T out;
        const lunchbox::Any* from = _default.empty() ? _parameter : &_default;
        assert(from != 0);

        if (from->type() == typeid(bool))
            convert<bool, T>()(*from, out);
        else if (from->type() == typeid(int))
            convert<int, T>()(*from, out);
        else if (from->type() == typeid(unsigned int))
            convert<unsigned int, T>()(*from, out);
        else if (from->type() == typeid(float))
            convert<float, T>()(*from, out);
        else if (from->type() == typeid(double))
            convert<double, T>()(*from, out);
        else if (from->type() == typeid(std::string))
            convert<std::string, T>()(*from, out);
        else if (from->type() == typeid(AttributeMapPtr))
        {
            /* This is not strictly necessary but enables the convenient
               implicit conversion from AttributeMapPtr to bool that
               std::shared_ptr provides. */
            convert<AttributeMapPtr, T>()(*from, out);
        }
        else
        {
            try
            {
                convert<T, T>()(*from, out);
            }
            catch (const std::runtime_error&)
            {
                std::stringstream s;
                s << "Can't convert to " << lunchbox::className(T());
                throw std::runtime_error(s.str());
            }
        }
        return out;
    }

    template <typename T>
    friend bool operator==(const ParameterProxy& proxy, const T& other)
    {
        T t = proxy;
        return t == other;
    }

    template <typename T>
    friend bool operator==(const T& other, const ParameterProxy& proxy)
    {
        T t = proxy;
        return t == other;
    }

    /* Specializations for const char */
    bool operator==(const char* other) const
    {
        std::string str = *this;
        return str == other;
    }
    friend bool operator==(const char* other, const ParameterProxy& proxy)
    {
        std::string str = proxy;
        return str == other;
    }

    inline AttributeProxy operator()(const std::string& data);

    inline const AttributeProxy operator()(const std::string& data) const;

    template <typename T>
    inline AttributeProxy operator()(const std::string& data,
                                     const T& default_) const;

    void set(const std::string& name, const Parameter& p1)
    {
        parameterAsAttribute().set(name, p1);
    }

    void set(const std::string& name, const Parameter& p1, const Parameter& p2)
    {
        parameterAsAttribute().set(name, p1, p2);
    }

    void set(const std::string& name, const Parameter& p1, const Parameter& p2,
             const Parameter& p3)
    {
        parameterAsAttribute().set(name, p1, p2, p3);
    }

    void set(const std::string& name, const Parameter& p1, const Parameter& p2,
             const Parameter& p3, const Parameter& p4)
    {
        parameterAsAttribute().set(name, p1, p2, p3, p4);
    }

    void set(const std::string& name, const Parameter& p1, const Parameter& p2,
             const Parameter& p3, const Parameter& p4, const Parameter& p5)
    {
        parameterAsAttribute().set(name, p1, p2, p3, p4, p5);
    }

    void set(const std::string& name, const Parameters& params)
    {
        parameterAsAttribute().set(name, params);
    }

    void set(const std::string& name, const AttributeProxy& params)
    {
        parameterAsAttribute().set(name, params);
    }

    template <typename T>
    int get(const std::string& name, T& p1) const
    {
        return parameterAsAttribute().get(name, p1);
    }

    template <typename T, typename U>
    int get(const std::string& name, T& p1, U& p2) const
    {
        return parameterAsAttribute().get(name, p1, p2);
    }

    template <typename T, typename U, typename V>
    int get(const std::string& name, T& p1, U& p2, V& p3) const
    {
        return parameterAsAttribute().get(name, p1, p2, p3);
    }

    template <typename T, typename U, typename V, typename W>
    int get(const std::string& name, T& p1, U& p2, V& p3, W& p4) const
    {
        return parameterAsAttribute().get(name, p1, p2, p3, p4);
    }

    template <typename T, typename U, typename V, typename W, typename X>
    int get(const std::string& name, T& p1, U& p2, V& p3, W& p4, X& p5) const
    {
        return parameterAsAttribute().get(name, p1, p2, p3, p4, p5);
    }

    int get(const std::string& name, std::vector<OutWrapper>& params) const
    {
        return parameterAsAttribute().get(name, params);
    }

protected:
    AttributeMap& parameterAsAttribute() const
    {
        if (_parameter->type() != typeid(AttributeMapPtr))
            LBTHROW(std::runtime_error("Parameter is not an attribute map"));

        return *lunchbox::any_cast<const AttributeMapPtr&>(*_parameter);
    }

protected:
    lunchbox::Any _default;
    const lunchbox::Any* _parameter;
};

class AttributeMap::AttributeProxy : public ParameterProxy
{
public:
    friend class AttributeMap;

    explicit AttributeProxy(const Parameters& parameters)
        : ParameterProxy(parameters[0])
        , _parameters(&parameters)
    {
    }

    template <typename T>
    explicit AttributeProxy(const T& default_)
        : ParameterProxy(default_)
        , _parameters(0)
    {
    }

    using ParameterProxy::operator();

    ParameterProxy operator()(size_t index)
    {
        if (_parameters == 0)
        {
            /* The attribute wasn't found we return the default value
               wrapped by the base class. */
            return *this;
        }

        if (_parameters->size() <= index)
            LBTHROW(std::runtime_error("Parameter index out of bounds"));

        return ParameterProxy((*_parameters)[index]);
    }

    const ParameterProxy operator()(size_t index) const
    {
        if (_parameters == 0)
        {
            /* The attribute wasn't found we return the default value
               wrapped by the base class. */
            return *this;
        }

        if (_parameters->size() <= index)
            LBTHROW(std::runtime_error("Parameter index out of bounds"));

        return ParameterProxy((*_parameters)[index]);
    }

    size_t getSize() const { return _parameters->size(); }
protected:
    const Parameters* _parameters;
};

AttributeMap::AttributeProxy AttributeMap::ParameterProxy::operator()(
    const std::string& data)
{
    AttributeMapPtr map;
    try
    {
        map = lunchbox::any_cast<const AttributeMapPtr&>(*_parameter);
    }
    catch (lunchbox::bad_any_cast&)
    {
        LBTHROW(std::runtime_error("Parameter is not an attribute map"));
    }
    return (*map)(data);
}

const AttributeMap::AttributeProxy AttributeMap::ParameterProxy::operator()(
    const std::string& data) const
{
    AttributeMapPtr map;
    try
    {
        map = lunchbox::any_cast<const AttributeMapPtr&>(*_parameter);
    }
    catch (lunchbox::bad_any_cast&)
    {
        LBTHROW(std::runtime_error("Parameter is not an attribute map"));
    }
    return (*map)(data);
}

template <typename T>
AttributeMap::AttributeProxy AttributeMap::ParameterProxy::operator()(
    const std::string& data, const T& default_) const
{
    AttributeMap& map = parameterAsAttribute();
    return map(data, default_);
}

template <typename T>
AttributeMap::AttributeProxy AttributeMap::operator()(const std::string& name,
                                                      const T& default_) const
{
    try
    {
        return operator()(name);
    }
    catch (std::runtime_error&)
    {
        return AttributeProxy(default_);
    }
}

class AttributeMap::const_iterator
    : public boost::iterator_facade<
          const_iterator, std::pair<std::string, const AttributeProxy>,
          std::bidirectional_iterator_tag,
          std::pair<std::string, const AttributeProxy>>
{
    friend class AttributeMap;
    friend class boost::iterator_core_access;
    typedef std::pair<std::string, const AttributeProxy> Dereferenced;

    /* Constructors */
public:
    const_iterator() {}
private:
    explicit const_iterator(
        const std::map<std::string, Parameters>::const_iterator& iter)
        : _iter(iter)
    {
    }

    /* Member functions */
private:
    Dereferenced dereference() const
    {
        return Dereferenced(_iter->first, AttributeProxy(_iter->second));
    }

    bool equal(const const_iterator& other) const
    {
        return _iter == other._iter;
    }

    void increment() { ++_iter; }
    void decrement() { --_iter; }
private:
    std::map<std::string, Parameters>::const_iterator _iter;
};
}
}
#endif
