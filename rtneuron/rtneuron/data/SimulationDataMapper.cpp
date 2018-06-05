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

#include "SimulationDataMapper.h"
#include "data/Neuron.h"
#include "data/Neurons.h"
#include "data/SpikeReport.h"
#include "scene/SimulationRenderBuffer.h"

#include <brain/compartmentReport.h>
#include <brain/compartmentReportMapping.h>
#include <brain/compartmentReportView.h>
#include <brain/neuron/morphology.h>
#include <brain/neuron/section.h>

#include <lunchbox/debug.h>
#include <lunchbox/types.h>

#include <OpenThreads/Thread>

#include <boost/lexical_cast.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper functions
*/
namespace
{
template <typename T>
static std::string lexical_cast(const T& t)
{
    return boost::lexical_cast<std::string>(t);
}
}

const float _maximumSpikeDelay = 25;                /* milliseconds */
const float _actionPotentialPropagationSpeed = 300; /* um/ms */

/*
  Constructors
*/
SimulationDataMapper::SimulationDataMapper()
    : _thread(&SimulationDataMapper::_run, this)
{
}

/*
  Destructor
*/
SimulationDataMapper::~SimulationDataMapper()
{
    if (_spikeReport)
        _spikeReport->close();

    cancel();
    _thread.join();
}

/*
  Member functions
*/

void SimulationDataMapper::updateMapping(Neurons& neurons)
{
    GIDSet target = neurons;

    /* Next the translation table is created */
    _gidsToIndices.clear();
    size_t index = 0;
    for (const auto gid : target)
        _gidsToIndices[gid] = index++;

    /* Then the cell offsets and indices are assigned after the compartment
       report is updated it it exsits. */
    if (_compartmentReport)
        _compartmentReportView.reset(new brain::CompartmentReportView(
            _compartmentReport->createView(target)));

#pragma omp parallel for
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        auto& neuron = *neurons[i];
        neuron.setSimulationBufferIndex(_gidsToIndices[neuron.getGID()]);
        neuron.setupSimulationOffsetsAndDelays(*this, false);
    }
}

uint64_t SimulationDataMapper::findAbsoluteOffset(const Neuron& neuron) const
{
    if (!_compartmentReportView)
        return LB_UNDEFINED_UINT64;

    const auto& mapping = _compartmentReportView->getMapping();

    /* Finding the minimum offset of all sections of this neuron */
    size_t index = neuron.getSimulationBufferIndex();
    assert(index != std::numeric_limits<size_t>::max());
    try
    {
        if (mapping.getOffsets().size() <= index)
            return LB_UNDEFINED_UINT64;

        const auto& offsets = mapping.getOffsets()[index];
        const auto& counts = mapping.getCompartmentCounts()[index];
        const size_t numSections = offsets.size();
        uint64_t offset = offsets[0];
        for (size_t s = 0; s < numSections; ++s)
        {
            if (counts[s] != 0)
                offset = std::min(offset, offsets[s]);
        }
        return offset;
    }
    catch (const std::exception&)
    {
        return LB_UNDEFINED_UINT64;
    }
}

void SimulationDataMapper::getSimulationOffsetsAndDelays(
    const Neuron& neuron, short* relativeOffsetsAndDelays,
    uint64_t& absoluteOffset, const uint16_t* sections, const float* positions,
    size_t length) const
{
    const size_t index = neuron.getSimulationBufferIndex();

    absoluteOffset = LB_UNDEFINED_UINT64;

    if (index == std::numeric_limits<size_t>::max())
        return;

    memset(relativeOffsetsAndDelays, 0, sizeof(short) * length);

    const bool hasCompartments = (bool)_compartmentReportView;
    if (hasCompartments)
    {
        absoluteOffset = findAbsoluteOffset(neuron);
        _getSimulationOffsets(neuron, relativeOffsetsAndDelays, absoluteOffset,
                              sections, positions, length);
    }
    _getAxonalDelays(neuron, relativeOffsetsAndDelays, sections, positions,
                     length, !hasCompartments);
}

uint64_t SimulationDataMapper::getSomaSimulationOffset(
    const Neuron& neuron) const
{
    if (!_compartmentReportView)
        return LB_UNDEFINED_UINT64;

    const size_t index = neuron.getSimulationBufferIndex();
    if (index == std::numeric_limits<size_t>::max())
        return LB_UNDEFINED_UINT64;

    const auto& mapping = _compartmentReportView->getMapping();
    const auto& counts = mapping.getCompartmentCounts()[index];
    if (counts[0] == 0)
        return LB_UNDEFINED_UINT64;

    /* This code assumes that the soma is section 0. There's no way to get rid
       of this assumption unles the morphology for the neuron is loaded, which
       is something we don't want to require for somas.
       The offsets from getOffsets() are already absolute. */
    return mapping.getOffsets()[index][0];
}

void SimulationDataMapper::setSimulationAtTimestamp(const double timestamp)
{
    std::unique_lock<std::mutex> lock(_workerMutex);
    if (!_thread.joinable())
        return;

    _updateCounter++;
    _timestamp = timestamp;
    _timestampReady = true;

    _condition.notify_all();
}

void SimulationDataMapper::waitForUpdate()
{
    std::unique_lock<std::mutex> lock(_workerMutex);
    while (_updateCounter > _waitCounter)
        _updateFinished.wait(lock);
}

void SimulationDataMapper::unblockUpdate()
{
    std::unique_lock<std::mutex> lock(_workerMutex);
    _simulationDataPending = false;
    _condition.notify_all();
}

void SimulationDataMapper::setCompartmentReport(
    const CompartmentReportPtr& report)
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    _compartmentReport = report;
}

void SimulationDataMapper::setSpikeReport(const SpikeReportPtr& report)
{
    std::unique_lock<std::mutex> lock(_stateMutex);
    _spikeReport = report;
}

void SimulationDataMapper::cancel()
{
    std::unique_lock<std::mutex> lock(_workerMutex);

    _cancel = true;
    _condition.notify_one();
    while (_updateCounter > _waitCounter)
        _updateFinished.wait(lock);
}

void SimulationDataMapper::_run()
{
    while (!_cancel)
    {
        double timestamp;
        {
            std::unique_lock<std::mutex> lock(_workerMutex);

            /* Wait until there's a mew timestamp to process and the last
               simulation data processed has been flushed to the geometry
               (that
               means unblockUpdate is called) or the thread has been asked to
               exit. */
            _condition.wait(lock, [&] {
                return _cancel || (_timestampReady && !_simulationDataPending);
            });

            if (_cancel)
            {
                _waitCounter = _updateCounter;
                _updateFinished.notify_all();
                break;
            }
            timestamp = _timestamp;
            _timestampReady = false;
        }

        bool simulationDirty = _updateSimulation(timestamp);

        {
            std::unique_lock<std::mutex> lock(_workerMutex);
            _simulationDataPending |= simulationDirty;
            _waitCounter = _updateCounter;
        }

        simulationUpdated(timestamp);

        LBVERB << "SimulationDataMapper: Timestamp done: "
               << lexical_cast(timestamp) << std::endl;

        _updateFinished.notify_all();
    }
}

bool SimulationDataMapper::_updateSimulation(double timestamp)
{
    if (!_simulationBuffer)
        return false;

    LBVERB << "SimulationDataMapper: Processing timestamp "
           << lexical_cast(timestamp) << std::endl;

    /* Only reading new compartment data if the front buffer (which must be
       the one displayed right now) doesn't contain the requested timestamp */
    assert(!_simulationDataPending);
    const bool upToDate =
        _simulationBuffer->getFrontBufferTimestamp() == timestamp;
    auto frame = upToDate ? brion::Frame() : _readFrame(timestamp);

    bool simulationDirty = false;
    SpikeReportPtr spikeReport;
    {
        std::unique_lock<std::mutex> lock(_stateMutex);
        spikeReport = _spikeReport;
        simulationDirty = frame.data || (spikeReport && !upToDate);
    }

    /* Updating simulation data buffers. */
    if (frame.data)
        _simulationBuffer->update(frame);

    if (spikeReport && !upToDate)
    {
        /* The purpose of this call is to update the internal spike
           buffer with the latest spikes, so the simulation window can
           be adjusted externally with the latest end time available. */
        spikeReport->tryUpdate();

        const auto& spikes =
            spikeReport->getSpikes(timestamp - _maximumSpikeDelay,
                                   std::nextafter((float)timestamp, INFINITY));
        _simulationBuffer->update(timestamp, spikes, _gidsToIndices);
    }

    return simulationDirty;
}

brion::Frame SimulationDataMapper::_readFrame(double timestamp)
{
    std::shared_ptr<brain::CompartmentReportView> view;
    double startTime;
    double endTime;
    {
        std::unique_lock<std::mutex> lock(_stateMutex);
        view = _compartmentReportView;
        if (!view)
            return brion::Frame();
        /* If there's a view, there's a _compartmentReport */
        startTime = _compartmentReport->getMetaData().startTime;
        endTime = _compartmentReport->getMetaData().endTime;
    }

    /* Bring the timestamp back into the simulation window currently present
       in the report. This should only affect use cases with sliding simulation
       windows, e.g. with simulation streaming. */

    timestamp = std::max(startTime, std::min(endTime, timestamp));
    auto frame = view->load(timestamp).get();
    if (!frame.data)
        LBWARN << "Could not read simulation frame for timestamp " << timestamp
               << std::endl;
    return frame;
}

void SimulationDataMapper::_getSimulationOffsets(
    const Neuron& neuron, short* relativeOffsets, const size_t absoluteOffset,
    const uint16_t* sections, const float* positions, const size_t length) const
{
    assert(_compartmentReportView);
    size_t index = neuron.getSimulationBufferIndex();

    const auto& mapping = _compartmentReportView->getMapping();
    if (mapping.getOffsets().size() <= index)
    {
        LBWARN << "Missing offset for cell " << neuron.getGID()
               << " in compartment report mapping. Bad cell target?"
               << std::endl;
        return;
    }

    const auto& morphology = *neuron.getMorphology();
    const auto& offsets = mapping.getOffsets()[index];
    const auto& counts = mapping.getCompartmentCounts()[index];

    const auto* section = sections;
    const auto* position = positions;
    size_t numSections = offsets.size();
    /* Assigning compartment offsets to the relative positions and
       sections given. */
    const auto axon =
        morphology.getSectionIDs({brain::neuron::SectionType::axon});
    uint16_t lastAxon = 0; /* No axon is assumed to have section id 0 */
    if (!axon.empty() && numSections != 1)
    {
        if (axon.size() > 1 && counts[axon[1]] != 0)
            lastAxon = axon[1];
        else if (counts[axon[0]] != 0)
            lastAxon = axon[0];
    }
    const auto& sectionTypes = morphology.getSectionTypes();
    for (size_t i = 0; i < length; ++position, ++section, ++i)
    {
        uint16_t compartments = *section < numSections ? counts[*section] : 0;
        if (compartments)
        {
            /* Computing the relative offset of this compartment */
            const uint16_t compartment =
                std::min(compartments - 1,
                         (int)floor(compartments * *position));
            size_t offset = offsets[*section];
            if (offset + compartment - absoluteOffset > 32767)
            {
                LBTHROW(std::runtime_error(
                    "Compartment offset overflow in neuron " +
                    lexical_cast(neuron.getGID())));
            }
            relativeOffsets[i] = offset + compartment - absoluteOffset;
        }
        else
        {
            if (sectionTypes[*section] == brain::neuron::SectionType::axon)
            {
                /* This is an unreported axon section.
                   Assigning the offset of the first compartment depending on
                   the presence of spike data. */
                if (!_spikeReport)
                {
                    if (offsets[lastAxon] - absoluteOffset > 32767)
                    {
                        LBTHROW(std::runtime_error(
                            "Compartment offset overflow in neuron " +
                            lexical_cast(neuron.getGID())));
                    }
                    relativeOffsets[i] = offsets[lastAxon] - absoluteOffset +
                                         counts[lastAxon] - 1;
                }
            }
        }
    }
}

void SimulationDataMapper::_getAxonalDelays(const Neuron& neuron, short* delays,
                                            const uint16_t* sections,
                                            const float* positions,
                                            const size_t length,
                                            const bool assignDelayToSoma) const
{
    assert(neuron.getMorphology());
    const auto& morphology = *neuron.getMorphology();

    const auto* section = sections;
    const auto* position = positions;

    const auto& sectionTypes = morphology.getSectionTypes();

    /* Assigning delays to the relative positions given as long as they
       are still zero and the section is an axon. */
    for (size_t i = 0; i < length; ++position, ++section, ++i)
    {
        if (delays[i] != 0)
            continue;

        if (sectionTypes[*section] == brain::neuron::SectionType::axon)
        {
            /* This is an unreported axon section for which spike data
               will be used if available. */

            /* Computing delays and using negative number for them to
               distingish them from the offsets. */
            const float p = std::max(std::min(*position, 1.0f), 0.0f);
            const auto& s = morphology.getSection(*section);
            const float distance = s.getDistanceToSoma() + s.getLength() * p;
            const float delay = distance / _actionPotentialPropagationSpeed;
            if (delay * 1024 > 32767)
            {
                LBTHROW(
                    std::runtime_error("Axonal delay time overflow in neuron " +
                                       lexical_cast(neuron.getGID())));
            }
            /* The delays are represented as negative numbers decremented
               by a constant 1. */
            delays[i] = -short(delay * 1024) - 1;
        }
        else if (assignDelayToSoma && *section == 0) /* Assumes soma is sec 0 */
        {
            /* This means delay 0. */
            delays[i] = -1;
        }
    }
}
}
}
}
