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

#include "Channel.h"

#include <eq/compositor.h>
#include <eq/fabric/subPixel.h>
#include <eq/util/accum.h>
#include <eq/util/objectManager.h>
#include <eq/window.h>

#include <osg/GL>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
namespace Compositor
{
namespace
{
struct Blending
{
    Blending()
        : _enabled(false)
    {
    }
    ~Blending()
    {
        if (_enabled)
            glDisable(GL_BLEND);
    }

    void check()
    {
        if (!_enabled)
        {
            _enabled = true;
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        }
    }

private:
    bool _enabled;
};

eq::util::Accum* _obtainAccum(eq::Channel* channel)
{
    const eq::PixelViewport& pvp = channel->getPixelViewport();

    LBASSERT(pvp.isValid());

#if EQ_VERSION_LT(1, 7, 0)
    eq::Window::ObjectManager& objects = *channel->getObjectManager();
#else
    eq::util::ObjectManager& objects = channel->getObjectManager();
#endif
    eq::util::Accum* accum = objects.getEqAccum(channel);
    if (!accum)
    {
        accum = objects.newEqAccum(channel);
        if (!accum->init(pvp, channel->getWindow()->getColorFormat()))
        {
            LBERROR << "Accumulation initialization failed." << std::endl;
        }
    }
    else
        accum->resize(pvp.w, pvp.h);

    accum->clear();
    return accum;
}
}

unsigned int assembleFrames(
    const eq::Frames& frames, eq::util::Accum* accum, Channel* channel,
    const boost::function<int(const eq::Frame&)>& framePosition)
{
    if (frames.empty())
        return 0;

    if (eq::Compositor::isSubPixelDecomposition(frames))
    {
        if (!accum)
        {
            accum = _obtainAccum(channel);
            accum->clear();
            const eq::SubPixel& subpixel =
                frames.back()->getFrameData()->getContext().subPixel;
            accum->setTotalSteps(subpixel.size);
        }

        unsigned int count = 0;
        eq::Frames framesLeft = frames;
        while (!framesLeft.empty())
        {
            eq::Frames subset = eq::Compositor::extractOneSubPixel(framesLeft);
            const unsigned int subCount =
                assembleFrames(subset, accum, channel, framePosition);
            assert(subCount < 2);

            if (subCount > 0)
                accum->accum();
            count += subCount;
        }
        if (count > 0)
            accum->display();
        return count;
    }

    /* RAII style handling for GL_BLEND */
    Blending blending;

    eq::Frames framesReady(frames.size(), 0);
    unsigned int nextFrame = 0; /* Only used if sorting is required. */

    /* Setting up the wait handler for waiting on all input frames at the same
       time. Optimized assembly that processes the frames as soon as they are
       available. Frame ordering is preserved during the process. */
    eq::Compositor::WaitHandle* waitHandle =
        eq::Compositor::startWaitFrames(frames, channel);

    unsigned int count = 0;
    for (size_t i = 0; i != frames.size(); ++i)
    {
        eq::Frame* frame = 0;
        {
            const eq::ChannelStatistics stats(
                eq::Statistic::CHANNEL_FRAME_WAIT_READY, channel);
            frame = eq::Compositor::waitFrame(waitHandle);
        }
        if (!frame || frame->getImages().empty())
            /* Not sure if skipping the frame without doing anything
               else is a good idea. */
            continue;

        count = 1;

        if (framePosition.empty())
        {
            /* No need for sorting, assemble directly */
            eq::Compositor::assembleFrame(frame, channel);
        }
        else
        {
            const int position = framePosition(*frame);

            /* Enable alpha blending for spatial DB modes */
            blending.check();

            /* During initialization it can happen that framesReady has
               fewer slots than the final number of participating nodes.
               This can cause the slot index to be larger than the
               vector. */
            if (framesReady.size() <= (unsigned int)position)
                framesReady.resize(position + 1);
            if (framesReady[position] != 0)
                std::cerr << "Warning: collision in compositing order"
                          << std::endl;

            framesReady[position] = frame;

            /* Composing all frames which are ready */
            for (;
                 nextFrame < framesReady.size() && framesReady[nextFrame] != 0;
                 ++nextFrame)
            {
                eq::Frame* f = framesReady[nextFrame];
                if (f->getImages().empty())
                    continue;
                try
                {
                    eq::Compositor::assembleFrame(f, channel);
                }
                catch (const co::Exception& e)
                {
                    LBWARN << e.what() << std::endl;
                }
            }
        }
    }

    /* Just in case there were gaps in the frame ranges, the vector of ready
       frames is reiterated. */
    for (; nextFrame < framesReady.size(); ++nextFrame)
    {
        eq::Frame* frame = framesReady[nextFrame];
        if (frame == 0 || frame->getImages().empty())
            continue;
        try
        {
            eq::Compositor::assembleFrame(frame, channel);
        }
        catch (const co::Exception& e)
        {
            LBWARN << e.what() << std::endl;
        }
    }

    return count;
}
}
}
}
}
