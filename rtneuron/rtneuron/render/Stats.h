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

#ifndef RTNEURON_STATS_H
#define RTNEURON_STATS_H

#include <boost/circular_buffer.hpp>
#include <iostream>
#include <map>
#include <string>

#ifndef NDEBUG
#include <set>
#endif

namespace bbp
{
namespace rtneuron
{
namespace core
{
class Stats
{
public:
    /*--- Public declarations ---*/
    typedef std::map<std::string, double> MeasureMap;

    /*--- Public constructors/destructor ---*/

    Stats(const std::string& name, unsigned int maxFrames = 5)
        : _name(name)
        , _stats(maxFrames)
        , _firstFrame(0)
        , _lastFrame(0)
    {
        _stats.resize(1);
    }

    Stats(unsigned int maxFrames = 5)
        : _stats(maxFrames)
        , _firstFrame(0)
        , _lastFrame(0)
    {
        _stats.resize(1);
    }

    /*--- Public member functions ---*/

    bool setMeasure(int frameNumber, const std::string& name, double value)
    {
#ifndef NDEBUG
        assert(_clearedFrames.find(frameNumber) == _clearedFrames.end());
#endif
        MeasureMap* measures = getFrameMeasures(frameNumber);
        if (measures)
        {
            measures->insert(std::make_pair(name, value));
            return true;
        }
        else
        {
            return false;
        }
    }

    bool incrementMeasure(int frameNumber, const std::string& name,
                          double value)
    {
#ifndef NDEBUG
        assert(_clearedFrames.find(frameNumber) == _clearedFrames.end());
#endif
        MeasureMap* measures = getFrameMeasures(frameNumber);
        if (measures)
        {
            double& measure = (*measures)[name];
            measure += value;
            return true;
        }
        else
        {
            return false;
        }
    }

    void setMeasure(const std::string& name, double value)
    {
        MeasureMap& measures = getLastFrameMeasures();
        measures.insert(std::make_pair(name, value));
    }

    void incrementMeasure(const std::string& name, double value)
    {
        double& measure = getLastFrameMeasures()[name];
        measure += value;
    }

    bool getMeasure(int frameNumber, const std::string& name,
                    double& value) const
    {
        const MeasureMap* measures = getFrameMeasures(frameNumber);
        if (measures)
        {
            MeasureMap::const_iterator m = measures->find(name);
            if (m != measures->end())
            {
                value = m->second;
                return true;
            }
        }
        return false;
    }

    bool getMeasure(const std::string& name, double& value) const
    {
        const MeasureMap& measures = getLastFrameMeasures();
        MeasureMap::const_iterator m = measures.find(name);
        if (m != measures.end())
        {
            value = m->second;
            return true;
        }
        return false;
    }

    MeasureMap& getLastFrameMeasures() { return _stats.back(); }
    const MeasureMap& getLastFrameMeasures() const { return _stats.back(); }
    MeasureMap* getFrameMeasures(unsigned int frameNumber)
    {
        if (frameNumber < _firstFrame)
        {
            /* Nothing to do, this frame has been removed and any measure
               left missed */
            return 0;
        }

        /** Pushing new measuremap until we reach the desired frame.
            \todo This doesn't work as expected if there is some frame
            skipping as in time multiplexing (anyways this is for
            profiling. */
        while (frameNumber > _lastFrame)
        {
            if (_stats.size() == _stats.capacity())
                /** First frame is removed */
                ++_firstFrame;
            _stats.push_back(MeasureMap());
            ++_lastFrame;
        }

        return &_stats[frameNumber - _firstFrame];
    }

    const MeasureMap* getFrameMeasures(unsigned int frameNumber) const
    {
        return const_cast<Stats*>(this)->getFrameMeasures(frameNumber);
    }

    void report(std::ostream& out, unsigned int frameNumber) const
    {
        const MeasureMap* measures = getFrameMeasures(frameNumber);
        if (measures)
        {
            for (MeasureMap::const_iterator m = measures->begin();
                 m != measures->end(); ++m)
            {
                out << _name << m->first << ' ' << m->second << std::endl;
            }
        }
    }

    void print(std::ostream& out) const
    {
        unsigned int frame = _firstFrame;
        for (boost::circular_buffer<MeasureMap>::const_iterator
                 i = _stats.begin();
             i != _stats.end(); ++i, ++frame)
        {
            out << "Frame " << frame << std::endl;
            for (MeasureMap::const_iterator m = i->begin(); m != i->end(); ++m)
                out << _name << m->first << ' ' << m->second << std::endl;
        }
    }

    void clearFrame(unsigned int frameNumber)
    {
        MeasureMap* measureMap = getFrameMeasures(frameNumber);
        if (measureMap)
            measureMap->clear();
#ifndef NDEBUG
        _clearedFrames.insert(frameNumber);
#endif
    }

private:
/*--- Private member attributes ---*/

#ifndef NDEBUG
    std::set<unsigned int> _clearedFrames;
#endif
    std::string _name;
    boost::circular_buffer<MeasureMap> _stats;
    unsigned int _firstFrame;
    unsigned int _lastFrame;
};
}
}
}
#endif
