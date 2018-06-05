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

#include "util/cameraToID.h"

#include <lunchbox/thread.h>

#include <osg/Camera>
#include <osg/ref_ptr>

#include <iostream>
#include <vector>

std::vector<uint32_t> counts;
size_t repetitions = 10000;
int cameras = 4;

class Thread : public lunchbox::Thread
{
public:
    Thread()
        : _camera(new osg::Camera())
    {
    }

    virtual void run()
    {
        for (size_t i = 0; i < repetitions; ++i)
        {
            uint32_t id = bbp::rtneuron::core::getUniqueCameraID(_camera);
            ++counts[id];
        }
    }

private:
    osg::ref_ptr<osg::Camera> _camera;
};

void sequentialTestNoThreads()
{
    counts.clear();
    counts.resize(cameras, 0);

    for (int i = 0; i < cameras; ++i)
    {
        osg::ref_ptr<osg::Camera> camera(new osg::Camera());
        for (size_t j = 0; j < repetitions; ++j)
        {
            uint32_t id = bbp::rtneuron::core::getUniqueCameraID(camera);
            ++counts[id];
        }
    }
}

void sequentialTest()
{
    counts.clear();
    counts.resize(cameras, 0);

    for (int i = 0; i < cameras; ++i)
    {
        Thread thread;
        thread.start();
        thread.join();
    }
}

void sequentialTestAllCameras()
{
    counts.clear();
    counts.resize(cameras, 0);

    Thread* threads = new Thread[cameras];
    for (int i = 0; i < cameras; ++i)
    {
        threads[i].start();
        threads[i].join();
    }
    delete[] threads;
}

void parallelTest()
{
    counts.clear();
    counts.resize(cameras, 0);

    Thread* threads = new Thread[cameras];
    for (int i = 0; i < cameras; ++i)
        threads[i].start();
    for (int i = 0; i < cameras; ++i)
        threads[i].join();
    delete[] threads;
}

int main()
{
    {
        osg::ElapsedTime timer;
        sequentialTestNoThreads();
        for (int i = 0; i < cameras; ++i)
        {
            assert(counts[i] == repetitions * cameras || i != 0);
            assert(counts[i] == 0 || i == 0);
            std::cout << counts[i] << std::endl;
        }
        double elapsed = timer.elapsedTime();
        std::cout << "Time taken " << timer.elapsedTime() << std::endl;
        std::cout << "Per operation "
                  << (elapsed / (repetitions * cameras) * 1000000000) << " ns"
                  << std::endl;
    }

    {
        osg::ElapsedTime timer;
        sequentialTest();
        for (int i = 0; i < cameras; ++i)
        {
            assert(counts[i] == repetitions * cameras || i != 0);
            assert(counts[i] == 0 || i == 0);
            std::cout << counts[i] << std::endl;
        }
        std::cout << "Time taken " << timer.elapsedTime() << std::endl;
    }

    {
        osg::ElapsedTime timer;
        sequentialTestAllCameras();
        for (int i = 0; i < cameras; ++i)
        {
            assert(counts[i] == repetitions);
            std::cout << counts[i] << std::endl;
        }
        std::cout << "Time taken " << timer.elapsedTime() << std::endl;
    }

    {
        osg::ElapsedTime timer;
        parallelTest();
        for (int i = 0; i < cameras; ++i)
        {
            assert(counts[i] == repetitions);
            std::cout << counts[i] << std::endl;
        }
        std::cout << "Time taken " << timer.elapsedTime() << std::endl;
    }
}
