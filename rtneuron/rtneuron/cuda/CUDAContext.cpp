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

#include "CUDAContext.h"

#include <lunchbox/debug.h>
#include <lunchbox/log.h>

#include <condition_variable>
#include <mutex>
#include <thread>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Static declarations
*/
namespace
{
std::mutex _globalMutex;
std::map<std::thread::id, CUDAContext*> _currentContextMap;
std::map<std::string, int> _interoperableDeviceCache;
std::map<unsigned int, unsigned int> _deviceRefCounts;

/*
  Helper functions
*/

void _checkAndRefDeviceID(unsigned int device)
{
    int deviceCount;
    const cudaError_t errorCode = cudaGetDeviceCount(&deviceCount);
    if (errorCode != 0)
    {
        std::stringstream msg;
        msg << "Creating CUDA Context: Error " << (unsigned int)errorCode
            << ": " << cudaGetErrorString(errorCode);
        LBTHROW(std::runtime_error(msg.str()));
    }
    if (device >= (unsigned int)deviceCount)
    {
        std::stringstream msg;
        msg << "Creating CUDA Context: invalid device ID " << device;
        LBTHROW(std::runtime_error(msg.str()));
    }

    std::unique_lock<std::mutex> lock(_globalMutex);
    ++_deviceRefCounts[device];
}

/*
  Resets a device if there are no more context alive using the device.
  The device must be current on the calling thread.
*/
void _resetDevice(unsigned int device)
{
    std::unique_lock<std::mutex> lock(_globalMutex);
    if (--_deviceRefCounts[device] == 0)
    {
        const auto error = cudaDeviceReset();
        if (error != cudaSuccess)
            std::cerr << "Could not reset CUDA device " << device << std::endl;
    }
}
}

namespace detail
{
class CUDAContext
{
public:
    CUDAContext(const unsigned int deviceID, core::CUDAContext* context)
        : _deviceID(deviceID)
        , _parent(context)
    {
        _checkAndRefDeviceID(deviceID);
        makeCurrent(context);
        if (cudaStreamCreate(&_defaultStream) != cudaSuccess)
        {
            std::cerr << "RTNeuron: Error creating CUDA stream: "
                      << cudaGetErrorString(cudaGetLastError()) << std::endl;
            abort();
        }
        _streams.push_back(_defaultStream);
        releaseContext();
    }

    ~CUDAContext()
    {
        makeCurrent(_parent);

        /* Deallocate all streams and events allocated and managed by this
           context */
        for (std::vector<cudaStream_t>::const_iterator i = _streams.begin();
             i != _streams.end(); ++i)
        {
            cudaError_t error = cudaStreamDestroy(*i);
            if (error != cudaSuccess && error != cudaErrorCudartUnloading)
            {
                std::cerr << "RTNeuron: Error destroying CUDA stream " << *i
                          << ": " << cudaGetErrorString(cudaGetLastError())
                          << std::endl;
            }
        }

        for (std::vector<cudaEvent_t>::const_iterator i = _events.begin();
             i != _events.end(); ++i)
        {
            cudaError_t error = cudaEventDestroy(*i);
            if (error != cudaSuccess && error != cudaErrorCudartUnloading)
            {
                std::cerr << "RTNeuron: Error destroying CUDA event " << *i
                          << ": " << cudaGetErrorString(cudaGetLastError())
                          << std::endl;
            }
        }

        _resetDevice(_deviceID);
        releaseContext();
    }

    void makeCurrent(core::CUDAContext* context)
    {
        std::unique_lock<std::mutex> lock(_mutex);

        if (_currentThread == std::thread::id())
            _makeCurrent(context);
        else if (_currentThread != std::this_thread::get_id())
            LBTHROW(
                std::runtime_error("CUDA context already current in "
                                   "another thread"));
    }

    /* A special version of makeCurrent that waits instead of throwing if
       another thread is holding the context current */
    void makeCurrentOrWait(core::CUDAContext* context)
    {
        std::unique_lock<std::mutex> lock(_mutex);

        if (_currentThread == std::this_thread::get_id())
            return;

        _condition.wait(lock,
                        [this] { return _currentThread == std::thread::id(); });

        _makeCurrent(context);
    }

    void releaseContext()
    {
        std::unique_lock<std::mutex> lock(_mutex);

        if (_currentThread != std::this_thread::get_id())
            LBTHROW(
                std::runtime_error("Can't release non current CUDA context"));
        {
            std::unique_lock<std::mutex> globalLock(_globalMutex);
            _currentContextMap[_currentThread] = 0;
        }
        _currentThread = std::thread::id();
        _condition.notify_all();
    }

    cudaStream_t createStream()
    {
        cudaStream_t stream;
        cudaError_t error = cudaStreamCreate(&stream);
        if (error != cudaSuccess)
        {
            std::stringstream str;
            str << "RTNeuron: Error creating CUDA stream: "
                << cudaGetErrorString(error) << std::endl;
            LBTHROW(std::runtime_error(str.str()));
        }
        _streams.push_back(stream);
        return stream;
    }

    void destroyStream(cudaStream_t stream)
    {
        cudaError_t error = cudaStreamDestroy(stream);
        if (error != cudaSuccess)
        {
            std::stringstream str;
            str << "RTNeuron: Error destroying CUDA stream: "
                << cudaGetErrorString(error) << std::endl;
            LBTHROW(std::runtime_error(str.str()));
        }
        _streams.erase(std::find(_streams.begin(), _streams.end(), stream));
    }

    cudaEvent_t createEvent()
    {
        cudaEvent_t event;
        cudaError_t error = cudaEventCreate(&event);
        if (error != cudaSuccess)
        {
            std::stringstream str;
            str << "RTNeuron: Error creating CUDA event: "
                << cudaGetErrorString(error) << std::endl;
            LBTHROW(std::runtime_error(str.str()));
        }
        _events.push_back(event);
        return event;
    }

    std::mutex _mutex;
    std::condition_variable _condition;

    std::vector<cudaStream_t> _streams;
    std::vector<cudaEvent_t> _events;

    cudaStream_t _defaultStream;

    unsigned int _deviceID;
    core::CUDAContext* _parent;
    std::thread::id _currentThread;

private:
    void _makeCurrent(core::CUDAContext* context)
    {
        _currentThread = std::this_thread::get_id();
        cudaError_t error = cudaSetDevice(_deviceID);
        if (error != cudaSuccess)
        {
            std::stringstream msg;
            msg << "Making current CUDA context for device " << _deviceID
                << ": " << cudaGetErrorString(error);
            LBTHROW(std::runtime_error(msg.str()));
        }
        std::unique_lock<std::mutex> globalLock(_globalMutex);
        _currentContextMap[_currentThread] = context;
    }
};
}

/*
  Constructors/destructor
*/
// cppcheck-suppress uninitMemberVar
CUDAContext::CUDAContext(const unsigned int deviceID)
    : _impl(new detail::CUDAContext(deviceID, this))
{
}

CUDAContext::~CUDAContext()
{
    /* If the context is current in any other thread different from the
       thread attempting to dispose it, an exception will be thrown. */
    _impl->makeCurrent(this);
    /* Release context is called by the destructor. */
    _impl.reset();
}

/*
  Member fuctions
*/
void CUDAContext::printAllDeviceInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device " << prop.name
                  << "\n---------------------------------"
                  << "\ntotalGlobalMem "
                  << prop.totalGlobalMem / 1024.0 / 1024.0 << "MB"
                  << "\nsharedMemPerBlock " << prop.sharedMemPerBlock / 1024.0
                  << "KB"
                  << "\nregsPerBlock " << prop.regsPerBlock << "\nwarpSize "
                  << prop.warpSize << "\nmemPitch " << prop.memPitch
                  << "\nmaxThreadsPerBlock " << prop.maxThreadsPerBlock
                  << "\nmaxThreadsDim " << prop.maxThreadsDim[0] << ' '
                  << prop.maxThreadsDim[1] << ' ' << prop.maxThreadsDim[2]
                  << "\nmaxGridSize " << ' ' << prop.maxGridSize[0] << ' '
                  << prop.maxGridSize[1] << ' ' << prop.maxGridSize[2]
                  << "\ntotalConstMem " << prop.totalConstMem << "\nmajor "
                  << prop.major << "\nminor " << prop.minor << "\nclockRate "
                  << prop.clockRate << "\ntextureAlignment "
                  << prop.textureAlignment << "\ndeviceOverlap "
                  << prop.deviceOverlap << "\nmultiProcessorCount "
                  << prop.multiProcessorCount << std::endl;
    }
}

unsigned int CUDAContext::getDeviceID(osg::GraphicsContext* context)
{
    std::unique_lock<std::mutex> lock(_globalMutex);
    /* Check first if we know the device for this display.
       This cache is needed because with Tesla GPUs, cudaGLGetDevices fails
       the second time it's called unless the X display connection used in the
       first call is left open. */
    std::string displayName = context->getTraits()->displayName();
    auto entry = _interoperableDeviceCache.find(displayName);
    if (entry != _interoperableDeviceCache.end())
        return entry->second;

    if (!context->isCurrent())
        context->makeCurrent();

    unsigned int count;
    int devices[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    const cudaError_t error =
        cudaGLGetDevices(&count, devices, 10, cudaGLDeviceListAll);
    if (error != cudaSuccess || count < 1)
    {
        std::stringstream msg;
        msg << "Couldn't find any CUDA device interoperable with display "
            << context->getTraits()->displayName() << ": "
            << (error == cudaSuccess ? "No device found"
                                     : cudaGetErrorString(error))
            << std::endl;
        LBTHROW(std::runtime_error(msg.str()));
    }
    _interoperableDeviceCache[displayName] = devices[0];
    return devices[0];
}

CUDAContext* CUDAContext::getOrCreateContext(osg::GraphicsContext* context)
{
    static std::mutex creationMutex;
    std::unique_lock<std::mutex> lock(creationMutex);

    CUDAContext* cudaContext =
        dynamic_cast<CUDAContext*>(context->getUserData());
    assert(context->getUserData() != 0 || cudaContext == 0);
    if (cudaContext == 0)
    {
        const unsigned int deviceID = getDeviceID(context);
        cudaContext = new CUDAContext(deviceID);
        context->setUserData(cudaContext);
    }
    return cudaContext;
}

CUDAContext* CUDAContext::getCurrentContext()
{
    std::unique_lock<std::mutex> lock(_globalMutex);
    return _currentContextMap[std::this_thread::get_id()];
}

std::function<void(void*)> CUDAContext::cudaFreeFunctor(
    cudaError_t (*free)(void*))
{
    return std::bind(&CUDAContext::_cudaFreeHelper,
                     osg::ref_ptr<CUDAContext>(this), free,
                     std::placeholders::_1);
}

cudaStream_t CUDAContext::createStream()
{
    return _impl->createStream();
}

void CUDAContext::destroyStream(cudaStream_t stream)
{
    return _impl->destroyStream(stream);
}

cudaStream_t CUDAContext::defaultStream()
{
    return _impl->_defaultStream;
}

cudaEvent_t CUDAContext::createEvent()
{
    return _impl->createEvent();
}

void CUDAContext::makeCurrent()
{
    _impl->makeCurrent(this);
}

void CUDAContext::releaseContext()
{
    _impl->releaseContext();
}

unsigned int CUDAContext::getDeviceID() const
{
    return _impl->_deviceID;
}

void CUDAContext::_cudaFreeHelper(cudaError_t (*free)(void*), void* ptr)
{
    CUDAContext* previous = getCurrentContext();
    if (previous != this)
    {
        if (previous)
            previous->releaseContext();
        _impl->makeCurrentOrWait(this);
    }

    free(ptr);

    if (previous != this)
    {
        releaseContext();
        if (previous)
            previous->makeCurrent();
    }
}
}
}
}
