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

#include "SpikeReport.h"

#include <brain/spikeReportReader.h>

#include <cmath>
#include <mutex>

namespace bbp
{
namespace rtneuron
{
namespace core
{
typedef std::unique_lock<std::mutex> Lock;

class SpikeReport::_Impl
{
public:
    _Impl(const SpikeReportReaderPtr& report)
        : _report(report)
        , _endTime(report->getEndTime())
        , _hasEnded(report->hasEnded())
    {
    }

    void update()
    {
        _endTime = _report->getEndTime();
        _hasEnded = _report->hasEnded();
    }

    std::mutex _reportMutex;
    std::mutex _stateMutex;
    SpikeReportReaderPtr _report;
    double _endTime;
    bool _hasEnded;
};

SpikeReport::SpikeReport(const SpikeReportReaderPtr& report)
    : _impl(new _Impl(report))
{
}

SpikeReport::~SpikeReport()
{
    delete _impl;
}

brain::Spikes SpikeReport::getSpikes(const double start, const double end)
{
    Lock lock(_impl->_reportMutex);
    auto spikes = _impl->_report->getSpikes(start, end);
    Lock lock2(_impl->_stateMutex);
    _impl->update();
    return spikes;
}

void SpikeReport::tryUpdate()
{
    Lock lock(_impl->_reportMutex, std::try_to_lock);
    if (!lock)
        return;
    const float endtime{_impl->_report->getEndTime()};
    /* We ask for a ridiculously small window just to get the end time updated
       by the implementation. */
    _impl->_report->getSpikes(std::nextafter(endtime, -INFINITY), endtime);
    Lock lock2(_impl->_stateMutex);
    _impl->update();
}

double SpikeReport::getEndTime() const
{
    Lock lock(_impl->_stateMutex);
    return _impl->_endTime;
}

bool SpikeReport::hasEnded() const
{
    Lock lock(_impl->_stateMutex);
    return _impl->_report->hasEnded();
}

void SpikeReport::close()
{
    _impl->_report->close();
}
}
}
}
