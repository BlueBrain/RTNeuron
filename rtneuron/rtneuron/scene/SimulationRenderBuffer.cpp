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

#include "SimulationRenderBuffer.h"

#include "config/constants.h"
#include "data/Neuron.h"

#include <osg/Image>
#include <osg/TextureBuffer>
#include <osg/Uniform>

#include <iterator>
#include <limits>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Constructors/destructor
*/
SimulationRenderBuffer::SimulationRenderBuffer()
    : _current(1)
    , _timestamp(std::numeric_limits<double>::quiet_NaN())
    , _frontTimestamp(std::numeric_limits<double>::quiet_NaN())
{
    for (int i = 0; i < 2; ++i)
    {
        _textures[i] = new osg::TextureBuffer();
        _textures[i]->setInternalFormat(GL_LUMINANCE32F_ARB);
        _spikeTextures[i] = new osg::TextureBuffer();
        _textures[i]->setInternalFormat(GL_R32F);
    }
}

SimulationRenderBuffer::~SimulationRenderBuffer()
{
}

/*
  Member functions
*/
void SimulationRenderBuffer::update(const brion::Frame& frame)
{
    const auto& values = *frame.data;
    assert(!values.empty());

    /* Updating compartment data buffer */
    int i = 1 - _current;
    osg::Image* image = _textures[i]->getImage();
    if (image == 0)
    {
        image = new osg::Image();
        _textures[i]->setImage(image);
    }
    _frames[i] = frame;
    image->setImage(values.size(), 1, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE,
                    GL_FLOAT, (unsigned char*)values.data(),
                    osg::Image::NO_DELETE);
    /* The SimulationDataMapper ensures that the timestamp used for spikes
       and report data matches */
    _timestamp = frame.timestamp;
    image->dirty();
    _textures[i]->setImage(image);
}

void SimulationRenderBuffer::update(const double timestamp,
                                    const brain::Spikes& spikes,
                                    const GIDToIndexMap& gidsToIndices)

{
    osg::Image* image = _spikeTextures[1 - _current]->getImage();
    if (image == 0)
    {
        image = new osg::Image();
        _spikeTextures[1 - _current]->setImage(image);
    }
    /* Resizing image if necessary */
    size_t size = gidsToIndices.size();
    if (image->s() != (int)size * 5)
    {
        image->setImage(size * 5, 1, 1, GL_R32F, GL_RED, GL_FLOAT,
                        (unsigned char*)new float[size * 5],
                        osg::Image::USE_NEW_DELETE);
        for (size_t j = 0; j < size * 5; ++j)
            ((float*)image->data())[j] =
                std::numeric_limits<float>::quiet_NaN();
    }

    /* Updating the buffer of spike times.
       The SimulationDataMapper ensures that the timestamp used for spikes
       and report data match. */
    _timestamp = timestamp;
    float* spikeTimes = (float*)image->data();
    bool dirty = false;
    for (const auto& spike : spikes)
    {
        auto indexIter = gidsToIndices.find(spike.second);

        if (indexIter == gidsToIndices.end())
            continue;

        const float time = spike.first;
        const size_t index = indexIter->second * 5;

        assert((int)index < image->s());
        /* Shifting all previous spikes times for this neuron to the right.
           This keeps the relevant spike times sorted highest to lowest.
           This is what the shader expects regardless of playback going
           forward of backwards. */
        for (int k = 4; k > 0; --k)
            spikeTimes[index + k] = spikeTimes[index + k - 1];
        spikeTimes[index] = time;
        dirty = true;
    }
    if (dirty)
        image->dirty();
}

void SimulationRenderBuffer::swapBuffers()
{
    /** Add a critical section with update? */
    _current = 1 - _current;
    assert(_textures[_current]->getImage() != 0 ||
           _spikeTextures[_current]->getImage() != 0);
    _frontTimestamp = _timestamp;

    if (_stateSet)
    {
        if (_textures[_current].get() != 0)
            _stateSet->setTextureAttribute(COMPARTMENT_DATA_TEXTURE_NUMBER,
                                           _textures[_current].get());
        if (_spikeTextures[_current].get() != 0)
            _stateSet->setTextureAttribute(SPIKE_DATA_TEXTURE_NUMBER,
                                           _spikeTextures[_current].get());
        _timestampUniform->set((float)_timestamp);
    }
}

void SimulationRenderBuffer::setTargetStateSet(osg::StateSet* stateSet)
{
    if (_stateSet == stateSet)
        return;
    _stateSet = stateSet;

    if (!_compartmentDataBufferUniform.valid())
    {
        _compartmentDataBufferUniform =
            new osg::Uniform("compartmentsBuffer",
                             COMPARTMENT_DATA_TEXTURE_NUMBER);
        _spikeDataBufferUniform =
            new osg::Uniform("spikesBuffer", SPIKE_DATA_TEXTURE_NUMBER);
        _timestampUniform = new osg::Uniform(osg::Uniform::FLOAT, "timestamp");
    }
    stateSet->addUniform(_compartmentDataBufferUniform.get());
    stateSet->addUniform(_spikeDataBufferUniform.get());
    stateSet->addUniform(_timestampUniform.get());
}
}
}
}
