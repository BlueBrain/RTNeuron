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

#ifndef RTNEURON_BOOST_SIGNAL_CONNECT_WRAPPER_H
#define RTNEURON_BOOST_SIGNAL_CONNECT_WRAPPER_H

#include "gil.h"

#include <boost/function_types/function_arity.hpp>
#include <boost/function_types/function_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/signals2/signal.hpp>

namespace bbp
{
namespace rtneuron
{
/**
   This class wraps a boost::python::object to ensure that the GIL is
   acquired during its destruction.
   This is needed because in boost.signals2 the disconnection of signals
   is deferred and it can occur from threads other than the main thread.
   During disconnection it's necessary to destroy the proxy of the Python
   callable that was attached to the signal, and this may require invoking
   the Python interpreter, so we need the GIL. */
class CallableWrapper
{
public:
    CallableWrapper(const boost::python::object& callable)
        : _callable(callable)
    {
    }

    ~CallableWrapper()
    {
        _releaser._state = PyGILState_Ensure();
        /* Next, _object will be destroyed, and finally the Release destructor
           will return to the previous lock state. */
    }

    bool operator==(const CallableWrapper& other) const
    {
        return _callable.ptr() == other._callable.ptr();
    }

    template <typename... Args>
    void operator()(Args... args) const
    {
        /* Acquiring the GIL this way because this may be called
           from a C++ thread unknown to the interpreter */
        EnsureGIL lock;
        try
        {
            _callable(args...);
        }
        catch (boost::python::error_already_set& e)
        {
            PyErr_Print();
            PyErr_SetString(PyExc_TypeError,
                            "Invalid function signature "
                            "invoking slot.");
        }
    }

private:
    struct Releaser
    {
        PyGILState_STATE _state;
        ~Releaser() { PyGILState_Release(_state); }
    };

    /* Destruction order is important. So the order of these member variables
       mustn't be changed. */
    Releaser _releaser;
    boost::python::object _callable;
};

template <typename Signature,
          int arity = boost::function_types::function_arity<Signature>::value>
class signal_connector;

template <typename Signature>
class signal_connector<Signature, 0>
{
public:
    typedef signal_connector<Signature, 0> This;

    void static connect(boost::signals2::signal<Signature>* signal,
                        boost::python::object& callable)
    {
        /* This function needs explicit overload resolution */
        void (CallableWrapper::*op)() const = &CallableWrapper::operator();
        signal->connect(boost::bind(op, CallableWrapper(callable)));
    }

    void static disconnect(boost::signals2::signal<Signature>* signal,
                           boost::python::object& callable)
    {
        /* This function needs explicit overload resolution */
        void (CallableWrapper::*op)() const = &CallableWrapper::operator();
        signal->disconnect(boost::bind(op, CallableWrapper(callable)));
    }
};

template <typename Signature>
class signal_connector<Signature, 1>
{
public:
    typedef signal_connector<Signature, 1> This;
    typedef typename boost::mpl::at_c<
        boost::function_types::parameter_types<Signature>, 0>::type Arg1;

    void static connect(boost::signals2::signal<Signature>* signal,
                        boost::python::object& callable)
    {
        signal->connect(boost::bind(&CallableWrapper::operator() < Arg1 >,
                                    CallableWrapper(callable), _1));
    }

    void static disconnect(boost::signals2::signal<Signature>* signal,
                           boost::python::object& callable)
    {
        signal->disconnect(boost::bind(&CallableWrapper::operator() < Arg1 >,
                                       CallableWrapper(callable), _1));
    }
};

template <typename Signature>
class signal_connector<Signature, 2>
{
public:
    typedef typename boost::mpl::at_c<
        boost::function_types::parameter_types<Signature>, 0>::type Arg1;
    typedef typename boost::mpl::at_c<
        boost::function_types::parameter_types<Signature>, 1>::type Arg2;

    void static connect(boost::signals2::signal<Signature>* signal,
                        boost::python::object& callable)
    {
        signal->connect(boost::bind(&CallableWrapper::operator() < Arg1, Arg2 >,
                                    CallableWrapper(callable), _1, _2));
    }

    void static disconnect(boost::signals2::signal<Signature>* signal,
                           boost::python::object& callable)
    {
        signal->disconnect(boost::bind(&CallableWrapper::operator() < Arg1,
                                       Arg2 >, CallableWrapper(callable), _1,
                                       _2));
    }
};

template <typename Signature>
class signal_connector<Signature, 3>
{
public:
    typedef typename boost::mpl::at_c<
        boost::function_types::parameter_types<Signature>, 0>::type Arg1;
    typedef typename boost::mpl::at_c<
        boost::function_types::parameter_types<Signature>, 1>::type Arg2;
    typedef typename boost::mpl::at_c<
        boost::function_types::parameter_types<Signature>, 2>::type Arg3;

    void static connect(boost::signals2::signal<Signature>* signal,
                        boost::python::object& callable)
    {
        signal->connect(boost::bind(&CallableWrapper::operator() < Arg1, Arg2,
                                    Arg3 >, CallableWrapper(callable), _1, _2,
                                    _3));
    }

    void static disconnect(boost::signals2::signal<Signature>* signal,
                           boost::python::object& callable)
    {
        signal->disconnect(boost::bind(&CallableWrapper::operator() < Arg1,
                                       Arg2, Arg3 >, CallableWrapper(callable),
                                       _1, _2, _3));
    }
};

#define WRAP_MEMBER_SIGNAL(class, name)                                     \
    class_<class ::name, boost::noncopyable>("__" #name "__", no_init)      \
        .def("connect", signal_connector<class ::name##Signature>::connect) \
        .def("disconnect",                                                  \
             signal_connector<class ::name##Signature>::disconnect)         \
        .def("disconnectAll", &class ::name::disconnect_all_slots);
}
}
#endif
