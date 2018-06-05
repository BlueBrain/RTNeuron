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

#include "timing.h"

#include "util/extensions.h"

#include <osg/Timer>

#include <iostream>
#include <memory>
#include <vector>

#define GPU_TIMING

namespace bbp
{
namespace rtneuron
{
namespace core
{
bool GPUTimer::s_started = false;

void GPUTimer::start(DrawExtensions* ext LB_UNUSED,
                     const std::string& name LB_UNUSED,
                     unsigned int frameNumber LB_UNUSED,
                     bool accumulate LB_UNUSED)
{
#if defined TIMING && defined GPU_TIMING
    if (s_started)
    {
        std::cerr << "Trying to start an overlapping GPU timer. Ignoring"
                  << std::endl;
    }
    else
    {
        s_started = true;
        if (_available.empty())
        {
            /* Creating a new query object. */
            Query query;
            ext->glGenQueries(1, &query.id);
            _available.push_back(query);
        }
        /* Accesing the next available query object. */
        Query& query = _available.front();
        query.frame = frameNumber;
        query.name = name;
        query.accumulate = accumulate;
        ext->glBeginQuery(GL_TIME_ELAPSED_EXT, query.id);
        /* Moving the available query to the pending list. */
        _pending.splice(_pending.end(), _available, _available.begin());
    }
#endif
}

void GPUTimer::stop(DrawExtensions* ext LB_UNUSED)
{
#if defined TIMING && defined GPU_TIMING
    assert(s_started);
    ext->glEndQuery(GL_TIME_ELAPSED_EXT);
    s_started = false;
#endif
}

bool GPUTimer::checkQueries(DrawExtensions* ext LB_UNUSED,
                            bool waitForCompletion LB_UNUSED)
{
/* \todo missing ext argument */
#if defined TIMING && defined GPU_TIMING
    QueryList::iterator q = _pending.begin();
    while (q != _pending.end())
    {
        /* Checking query availability only if waitForCompletion is false. */
        GLuint available = waitForCompletion;
        if (!waitForCompletion)
            ext->glGetQueryObjectuiv(q->id, GL_QUERY_RESULT_AVAILABLE_ARB,
                                     &available);
        if (!available)
            /* No more queries to consider because they won't be available. */
            break;

        /* Getting the query result.  */
        GLuint64EXT elapsed;
/* Though valid, this code produces a link error in Linux with
   OSG 2.8.2, not sure what happens in Windows. */
#ifdef _WIN32
        ext->glGetQueryObjectui64v(q->id, GL_QUERY_RESULT_ARB, &elapsed);
#else
        glGetQueryObjectui64vEXT(q->id, GL_QUERY_RESULT_ARB, &elapsed);
#endif
        double elapsedMs = double(elapsed) * 1e-6;
        /* Storing it in the stats object. */
        if (_stats)
        {
            bool missed =
                q->accumulate
                    ? !_stats->incrementMeasure(q->frame, q->name, elapsedMs)
                    : !_stats->setMeasure(q->frame, q->name, elapsedMs);
            if (missed)
            {
                std::cerr << "Missing measure: " << q->name << ' ' << q->frame
                          << std::endl;
            }
        }
        else
        {
            std::cout << q->name << ' ' << elapsedMs << std::endl;
        }

        /* Moving the node pointed by q to the end of the available list.
           After this call q points to the node that was following the
           moved one (which hasn't been invalidated at all). */
        _available.splice(_available.end(), _pending, q++);
    }
#endif
    return _pending.empty();
}

bool GPUTimer::pendingQueriesMinimumFrame(unsigned int& frame LB_UNUSED) const
{
#if defined TIMING && defined GPU_TIMING
    if (_pending.empty())
        return false;

    frame = _pending.front().frame;
    return true;
#else
    return false;
#endif
}
}
}
}
