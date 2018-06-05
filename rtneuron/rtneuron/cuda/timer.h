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

#ifndef RTNEURON_CUDA_TIMER_H
#define RTNEURON_CUDA_TIMER_H

#include <cuda.h>

#include <chrono>
#include <iostream>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace cuda
{
/** Timer class for CUDA for testing purposes */
class Timer
{
public:
    Timer(cudaStream_t stream = 0, bool withGPU = false)
        : _cpuStart(std::chrono::high_resolution_clock::now())
        , _stream(stream)
        , _withGPU(withGPU)
    {
        if (_withGPU)
        {
            if (cudaEventCreate(&_gpuStart) != cudaSuccess ||
                cudaEventCreate(&_gpuStop) != cudaSuccess ||
                cudaEventRecord(_gpuStart, _stream) != cudaSuccess)
            {
                std::cerr << "Error starting CUDA timers" << std::endl;
                exit(-1);
            }
        }
    }

    ~Timer()
    {
        const auto now = std::chrono::high_resolution_clock::now();
        if (_withGPU)
        {
            cudaEventRecord(_gpuStop, _stream);
            cudaEventSynchronize(_gpuStop);
        }

        const double cpuElapsed = (now - _cpuStart).count();
        std::cout << "  CPU time elapsed: " << cpuElapsed / 1000000.0 << " ms"
                  << std::endl;

        if (_withGPU)
        {
            float gpuElapsed;
            cudaEventElapsedTime(&gpuElapsed, _gpuStart, _gpuStop);
            std::cout << "  GPU time elapsed: " << gpuElapsed << " ms"
                      << std::endl;
            cudaEventDestroy(_gpuStart);
            cudaEventDestroy(_gpuStop);
        }
    }

private:
    std::chrono::time_point<std::chrono::system_clock> _cpuStart;
    cudaStream_t _stream;
    cudaEvent_t _gpuStart;
    cudaEvent_t _gpuStop;
    bool _withGPU;
};
}
}
}
}
#endif
