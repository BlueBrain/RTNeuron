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
#ifndef RTNEURON_SIMULATIONRENDERBUFFER_H
#define RTNEURON_SIMULATIONRENDERBUFFER_H

#include "coreTypes.h"

#include <brain/types.h>

#include <osg/ref_ptr>

namespace osg
{
class StateSet;
class TextureBuffer;
class Uniform;
}

namespace bbp
{
namespace rtneuron
{
namespace core
{
//! Class to manage a Texture Buffer Object containing simulation data.
/*! This class creates and updates the TBOs used to store the membrane
  simulation data and spike data of a given framestamp in GPU memory.

  \todo Use dash instead of a double buffer (including state attributes) to
  support time multiplexing.
*/
class SimulationRenderBuffer
{
public:
    /*--- Public constructors/destructor ---*/

    SimulationRenderBuffer();

    ~SimulationRenderBuffer();

    /*--- Public member functions ---*/

    /**
       Updates the back buffer texture for compartmental simulation.
       @param frame The simulation data.
       @param subtarget If not empty, this target
     */
    void update(const brion::Frame& frame);

    /**
       Updates the back buffer texture with the spike times for each neuron
       at a given timestamp.
       @param timestamp The requested timesamp
       @param spikes The spike data
       @param gidsToIndices The GID -> buffer index table to use.
     */
    void update(double timestamp, const brain::Spikes& spikes,
                const GIDToIndexMap& gidsToIndices);

    /**
       Sets the back textures used for simulation display as front textures
       in the state set stored in setTargetStateSet.
       Time tiemstamp uniform is also updated.
     */
    void swapBuffers();

    /**
       Prepares a StateSet to use it to hold the simulation buffer textures
       and GLSL uniforms and stores it to add the texture attributes when
       needed.
     */
    void setTargetStateSet(osg::StateSet* stateSet);

    /**
       @return The timestamp of the back buffer or quiet NaN if invalid.
     */
    double getBackBufferTimestamp() const { return _timestamp; }
    /**
       @return The timestamp of the front buffer or quiet NaN if invalid.
     */
    double getFrontBufferTimestamp() const { return _frontTimestamp; }
private:
    /*--- Private member attributes ---*/

    osg::ref_ptr<osg::TextureBuffer> _textures[2];
    osg::ref_ptr<osg::TextureBuffer> _spikeTextures[2];
    int _current;
    double _timestamp;      /* Available timestamp in the back buffers */
    double _frontTimestamp; /* A different variable is used because it's not
                               possible to recover NaN from an osg::Uniform to
                               detect the uninitialized state. */

    brion::Frame _frames[2];
    osg::ref_ptr<osg::StateSet> _stateSet;

    osg::ref_ptr<osg::Uniform> _compartmentDataBufferUniform;
    osg::ref_ptr<osg::Uniform> _spikeDataBufferUniform;
    osg::ref_ptr<osg::Uniform> _timestampUniform;
};
}
}
}
#endif
