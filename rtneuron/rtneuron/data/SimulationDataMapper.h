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

#ifndef RTNEURON_SIMULATIONDATAMAPPER_H
#define RTNEURON_SIMULATIONDATAMAPPER_H

#include "coreTypes.h"
#include "types.h"

#include <brain/compartmentReportView.h>

#include <boost/signals2/signal.hpp>

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/** This class manages the simulation data loading, preprocessing and
    mapping to vertex attributes of neurons.

    The data model access is provided by derived classes, which must
    define some protected pure virtual functions of this class.
*/
class SimulationDataMapper
{
public:
    /*--- Public declarations ---*/

    typedef void SimulationUpdatedSignature(const double);
    typedef boost::signals2::signal<SimulationUpdatedSignature>
        SimulationUpdatedSignal;

    /*--- Public constructors/destructor ---*/

    SimulationDataMapper();

    /**
       Destructor
       @sa Worker::~Worker for important thread-safety considerations when
       deriving this class.
    */
    virtual ~SimulationDataMapper();

    /*--- Public member functions ---*/

    /**
       Updates all mapping related structures thread-safely with regard
       to the inner thread.

       Updates the target of the simulation readers, the neuron indices
       and offsets the refer to the compartment and spike buffers and
       the GID to index table used SimulationRenderBuffer for spikes.
    */
    void updateMapping(Neurons& neurons);

    /**
       Returns the simulation buffer offset of a neuron according to the current
       simulation mapping.
     */
    uint64_t findAbsoluteOffset(const Neuron& neuron) const;

    /**
       Returns the simulation buffer offsets and axonal delays for
       morphological locations of a neuron.

       The function returns the absolute offset of the neuron and the
       array of relative offsets of the input section, position pairs.

       If the neuron is not present in the compartmental simulation or
       there's not compartmental simulation data source available
       LB_UNDEFINED_UINT64 is returned as the absoluteOffset. If a
       dendrite section is not reported in the mapping, an exception will be
       thrown.

       Axonal delays are returned for axonal sections for which there are no
       compartments specified in the current simuation mapping.
       Delays are computed assuming 300 um/ms of propagation velocity. To
       distinguish between delays and offsets, delays are represented by the
       negative range mapping delay 0 to -1. Delays are computed in ms and
       then quantized by multiplying by 1024 and truncating.

       The rationale behind this codification is to be able to use a single
       short int vertex attribute array for all the per vertex simulation
       parameters needed in the GLSL side.

       @param neuron
       @param relativeOffsetsAndDelays The output array with relative offsets
       and delays. Must be allocated by the caller.
       @param absoluteOffset
       @param sections Array of morphological sections
       @param positions Array of relative morphological positions within each
       respective section.
       @param length Length of the all the array parameters
    */
    void getSimulationOffsetsAndDelays(const Neuron& neuron,
                                       short* relativeOffsetsAndDelays,
                                       uint64_t& absoluteOffset,
                                       const uint16_t* sections,
                                       const float* positions,
                                       size_t length) const;

    /**
       Returns the absolute offset of a cell's soma in the simulation buffer.

       If the neuron is not present in the compartmental simulation or
       there's not compartmental simulation data source available
       LB_UNDEFINED_UINT64 is returned.
    */
    uint64_t getSomaSimulationOffset(const Neuron& neuron) const;

    /**
       Queues a new timestamp for the internal thread to process.

       The thread will update the compartment and spike data buffers to the
       given timestamp (or the most approximate one in the case of streaming).
       The mapping offsets and delays are updated if necessary.
    */
    void setSimulationAtTimestamp(double timestamp);

    /**
       Blocks the calling thread until a new timestamp is processed and the
       simulationUpdated signal has been emitted.
    */
    void waitForUpdate();

    /**
       Method to be called when the current simulation data has been properly
       buffered into the geometry objects that will display it for the
       next frame.
    */
    void unblockUpdate();

    /**
       Set the object used to store the compartment and spike data in
       rendering ready storage.
    */
    void setSimulationRenderBuffer(const SimulationRenderBufferPtr& buffer)
    {
        _simulationBuffer = buffer;
    }

    /**
       Sets the simulation report to be used next time a timestamp is
       processed (may be null).

       The mapping will be set as dirty and offsets/delays recomputed.
     */
    void setCompartmentReport(const CompartmentReportPtr& reader);

    /**
       Returns the current report reader.
     */
    CompartmentReportPtr getCompartmentReport() const
    {
        return _compartmentReport;
    }

    /**
       Set the spike report reader to be used next time a timestamp is
       processed (may be null).

       The mapping will be set as dirty if the availability of a reader
       has changed.
    */
    void setSpikeReport(const SpikeReportPtr& reader);

    const SpikeReportPtr& getSpikeReport() const { return _spikeReport; }
    void cancel();

    /*--- Public signals ---*/

    /** Callbacks for this signal shouldn't be blocking, otherwise
        cancelling the internal thread may deadlock. */
    SimulationUpdatedSignal simulationUpdated;

protected:
    /*--- Private member attributes ---*/

    std::mutex _workerMutex;
    std::mutex _stateMutex;
    std::condition_variable _condition;

    bool _cancel = false;

    double _timestamp;
    bool _timestampReady = false;

    std::condition_variable _updateFinished;
    unsigned long _updateCounter = 0;
    unsigned long _waitCounter = 0;
    bool _simulationDataPending = false;
    bool _mappingDirty = false;

    CompartmentReportPtr _compartmentReport;
    std::shared_ptr<brain::CompartmentReportView> _compartmentReportView;

    SpikeReportPtr _spikeReport;

    SimulationRenderBufferPtr _simulationBuffer;
    GIDToIndexMap _gidsToIndices;

    /* Don't move up as all other members must be created before the thread
       is started. */
    std::thread _thread;

    /*--- Private member functions ---*/

    void _run();

    bool _updateSimulation(double timestamp);

    /**
        Gets a compartmental data frame with the given timestamp and
        stores it.

        The selection of the data source and loading are responsibility of
        the derived classes.
        The timestamp is a hint of the desired timestamp, however if the
        simulation mapper is receiving data via streaming, the frame's
        timestamp may differ. The timestamp argument is an in-out argument
        to return the actual timestamp read.
        The returned value is true in case of success and false otherwise.
    */
    brion::Frame _readFrame(double timestamp);

    void _getSimulationOffsets(const Neuron& neuron, short* relativeOffsets,
                               size_t absoluteOffset, const uint16_t* sections,
                               const float* positions, size_t length) const;

    void _getAxonalDelays(const Neuron& neuron, short* delays,
                          const uint16_t* sections, const float* positions,
                          size_t length, bool assignDelayToSoma) const;
};
}
}
}
#endif
