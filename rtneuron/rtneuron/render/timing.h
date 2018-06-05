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

#ifndef RTNEURON_TIMING_H
#define RTNEURON_TIMING_H

#include "Stats.h"

#include "util/extensions.h"

#include <osg/Timer>

#include <lunchbox/compiler.h>

#include <cassert>
#include <iostream>
#include <list>

namespace bbp
{
namespace rtneuron
{
namespace core
{
//#define TIMING
//#define DP_TIMING // Depth peeling timing
//#define CULL_TIMING

/**
   All times are in millisecons.
 */
class Tick
{
public:
    /**
       Will output messages to std::cout is the Stats objects is null,
       otherwise it will add measures to the last frame stored in the stats.
     */
    Tick(Stats* stats = 0)
        : _stats(stats)
    {
#ifdef TIMING
        _tick = osg::Timer::instance()->tick();
#endif
    }

    /**
       Updates the reference time and emits/stores the difference between
       the previous reference and the new one.
       If the name of the measure if not empty, the difference between the
       old timestamp and the new one, referred as the measure, is computed.
       If the internal stats object is null, the measure is reported to
       std::cout, otherwise it is stored in the stats object.
    */
    double tick(const std::string& name LB_UNUSED = "")
    {
        double result = 0;
#ifdef TIMING
        osg::Timer_t now = osg::Timer::instance()->tick();
        result = osg::Timer::instance()->delta_s(_tick, now) * 1000;
        if (name != "")
        {
            if (_stats != 0)
                _stats->setMeasure(name, result);
            else
                std::cout << name << ' ' << result << std::endl;
        }
        _tick = now;
#endif
        return result;
    }

protected:
    osg::Timer_t _tick;
    Stats* _stats;
};

/**
 */
class GPUTimer
{
    /* Constructor */
public:
    /**
     */
    GPUTimer(Stats* stats = 0)
        : _stats(stats)
    {
    }

    /* Destructor */
public:
    /**
       \bug Query objects are not cleaned.
     */
    ~GPUTimer() {}
    /* Member functions */
public:
    void start(DrawExtensions* extensions, const std::string& message,
               unsigned int frameNumber, bool accumulate = false);

    void stop(DrawExtensions* extensions);

    bool checkQueries(DrawExtensions* extensions,
                      bool waitForCompletion = false);

    bool pendingQueriesMinimumFrame(unsigned int& frame) const;

    /* Member attributes */
protected:
    static bool s_started;

    struct Query
    {
        GLuint id;
#if defined TIMGING && defined GPU_TIMING
        unsigned int frame;
        std::string name;
        bool accumulate;
#endif
    };
    typedef std::list<Query> QueryList;
    QueryList _available;
    QueryList _pending;

    Stats* _stats;
};
}
}
}
#endif
