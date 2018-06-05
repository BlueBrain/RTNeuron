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

#ifndef RTNEURON_CUDA_CUDACONTEXT_H
#define RTNEURON_CUDA_CUDACONTEXT_H

#include <osg/GraphicsContext>
#include <osg/Referenced>

#include <boost/function/function1.hpp>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <functional>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace detail
{
class CUDAContext;
}

/**
   This class is not thread-safe on purpose, synchronization must be
   external.
*/
class CUDAContext : public osg::Referenced, boost::noncopyable
{
public:
    /*--- Public static functions ---*/

    /** Return the ID of a device which is interoperable with the given GL
        context.

        The GL context may be made current if it's not already.
    */
    static unsigned int getDeviceID(osg::GraphicsContext* context);

    /**
       Returns a new or existing context for the device associated with
       a given graphics context.

       The CUDAContext returned is the user data pointer assigned to
       context. The GL context is made current if it's not already.
    */
    static CUDAContext* getOrCreateContext(osg::GraphicsContext* context);

    /**
       Returns a pointer to the CUDA context currently bound by the caller
       thread or 0 if no context is bound.
    */
    static CUDAContext* getCurrentContext();

    static void printAllDeviceInfo();

    /*--- Public constructors ---*/

    explicit CUDAContext(const unsigned int deviceID);

    /*--- Public member functions ---*/

    /**
       Helper function that returns a functor object to be passed as a
       custom deallocator to smart pointers holding device pointers allocated
       in this context device.
       The function parameter is the cuda fuction to use for deallocation
       (e.g. to distinguish between host and device arrays).
     */
    std::function<void(void*)> cudaFreeFunctor(cudaError_t (*free)(void*));

    /**
       \brief Creates a new CUDA stream object that will be deallocated when
       this context is destroyed.
       Throws an exception is an error occurs.
    */
    cudaStream_t createStream();

    /**
       Destroys the given CUDA stream.
       Throws an exception is an error occurs.
    */
    void destroyStream(cudaStream_t stream);

    /**
       \brief Returns a stream different from stream 0 to be used as the
       default stream for asynchronous processing.
     */
    cudaStream_t defaultStream();

    /**
       \brief Creates a new CUDA event object that will be deallocated when
       this context is destroyed.
    */
    cudaEvent_t createEvent();

    /**
       Sets the device pointed by this context as the current context for
       this thread.
     */
    void makeCurrent();

    /**
       Releases the device from this thread so other thread can use it.
     */
    void releaseContext();

    unsigned int getDeviceID() const;

protected:
    /*--- Protected destructor ---*/

    ~CUDAContext();

    /*--- Private member variables ---*/
private:
    boost::scoped_ptr<detail::CUDAContext> _impl;

    void _baseConstruction();
    void _cudaFreeHelper(cudaError_t (*free)(void*), void* ptr);

    /*--- Private member functions ---*/

    void _makeCurrentOrWait();
};

/**
   Scoped based management of CUDA contexts

   The constructor only makes the context current if not already.
   The destructor releases the context if it wasn't the current one and
   returns to a previous context if necessary.
   This policies provide a limited ability for nesting CUDA contexts.
 */
class ScopedCUDAContext
{
public:
    /*--- Public constructors/destructor ---*/
    /**
       The reference count of context is not increased by this method.
     */
    ScopedCUDAContext(CUDAContext* context)
        : _context(context)
    {
        /* Previous must me a hard reference somewhere else in the calling
           thread because it cannot be current if it was already destroyed (
           the destructor releases it). */
        _previous = CUDAContext::getCurrentContext();
        if (_previous != context)
        {
            if (_previous != 0)
                _previous->releaseContext();
            _context->makeCurrent();
        }
    }

    /**
       The contained context won't be deallocated here.
     */
    ~ScopedCUDAContext()
    {
        if (_previous != _context)
        {
            _context->releaseContext();
            if (_previous)
                _previous->makeCurrent();
        }
    }

    /*--- Public member functions ---*/

    CUDAContext& operator*() { return *_context; }
    const CUDAContext& operator*() const { return *_context; }
    CUDAContext* operator->() { return _context; }
    const CUDAContext* operator->() const { return _context; }
private:
    /*--- Private member variables ---*/
    CUDAContext* _context;
    CUDAContext* _previous;
};
}
}
}
#endif
