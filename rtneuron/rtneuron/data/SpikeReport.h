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

#ifndef RTNEURON_SPIKEREPORT_H
#define RTNEURON_SPIKEREPORT_H

#include "types.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
/** A thread safe version of a spike report reader for RTNeuron

This class allows concurrent access to the report window and report updates
by many threads. Access to the spikes containers is assumed to occur only from
a single thread.
*/
class SpikeReport
{
public:
    /*--- Public constructors/destructor ---*/

    SpikeReport(const SpikeReportReaderPtr& reader);

    ~SpikeReport();

    SpikeReport(const SpikeReport&) = delete;
    SpikeReport& operator=(const SpikeReport&) = delete;

    /*--- Public member functions ---*/

    brain::Spikes getSpikes(const double start, const double end);

    /** Tries to update the internal spike buffer with the latest spikes
        but without locking in case some other thread is already waiting in
        getSpikes. */
    void tryUpdate();

    double getEndTime() const;

    double getStartTime() { return 0; }
    bool hasEnded() const;

    void close();

private:
    /*--- Private member attributes ---*/
    class _Impl;
    _Impl* _impl;
};
}
}
}
#endif
