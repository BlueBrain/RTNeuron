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

#ifndef RTNEURON_OSGEQ_CHANNEL_H
#define RTNEURON_OSGEQ_CHANNEL_H

#include "Compositor.h"
#include "Pipe.h"
#include "Window.h"
#include "render/DepthOfField.h"

#include <eq/eq.h>

#include <boost/scoped_ptr.hpp>
#include <fstream>
#include <osg/ref_ptr>

namespace osg
{
class Camera;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
class MultiFragmentFunctors;

//! A eq::Channel mapped to a osg::Camera
class Channel : public eq::Channel
{
    friend class Window;

public:
    /* Public constructors/destructor */

    Channel(eq::Window* parent);

    virtual ~Channel();

    /* Public member functions */

    osg::Camera* getCamera() { return _camera.get(); }
    const osg::Camera* getCamera() const { return _camera.get(); }
    bool useOrtho() const override;

protected:
    /*--- Protected member functions ---*/

    bool configInit(const eq::uint128_t& initDataID) override;

    bool configExit() override;

    void frameClear(const eq::uint128_t& frameID) override;

    /** All 3D rendering should happen here */
    void frameDraw(const eq::uint128_t& value) override;

    void frameAssemble(const eq::uint128_t&, const eq::Frames&) override;

    void frameReadback(const eq::uint128_t&, const eq::Frames&) override;

    void frameStart(const eq::uint128_t& frameID,
                    const uint32_t frameNumber) override;

    void frameViewStart(const eq::uint128_t& frameID) override;

    void frameFinish(const eq::uint128_t& frameID,
                     const uint32_t frameNumber) override;

    /** All 2D rendering should happen here (after draw and compositing) */
    void frameViewFinish(const eq::uint128_t& frameID) override;

    eq::Vector2f getJitter() const override;

private:
    /* Private member attributes */

    LB_TS_VAR(_thread);
    mutable std::mutex _lock;

    osg::ref_ptr<osgViewer::View> _view;

    osg::ref_ptr<osg::Camera> _camera;

    std::unique_ptr<core::DepthOfField> _depthOfField;

    osg::ref_ptr<osg::Node> _bareScene;

    osg::ref_ptr<osg::Uniform> _lightSource;

    osg::ref_ptr<osg::Viewport> _minimumViewport;

#ifndef NDEBUG
    osg::ref_ptr<osg::Geometry> _border;
#endif

    /* Used during frame assembly */
    eq::Frame _frame;
    osg::Vec4 _clearColor;
    /* Draw range of the last draw/assemble operation on this channel. */
    eq::Range _drawRange;
    /* Original draw range for source channels */
    eq::Range _sourceDrawRange;

    eq::uint128_t _frameCounterRef;

    eq::util::Texture* _texture;

    struct Accum
    {
        Accum();
        ~Accum();

        std::unique_ptr<eq::util::Accum> buffer;
        int remainingSteps;
        uint32_t stepsDone;
        bool transfer;
    };

    eq::PixelViewport _currentDestinationPVP;
    Accum _accum[eq::NUM_EYES];

    std::unique_ptr<MultiFragmentFunctors> _multiFragmentFunctors;

    std::fstream _log;

    /*--- Private member functions ---*/

    /**
        \brief Sets up the scene data for the osgViewer::View used in this
        channel.

        Returns true if drawing should proceed (there exists and scene and
        it was already setup) and false otherwise.
    */
    bool _setupChannelScene(const uint32_t frameNumber);

    void _setupPostRenderEffects();

    void _updateClearColor();

    void _postDraw();

    void _drawViewportBorder();

    void _updateRegionOfInterest();

    void _multiFragmentCompositing(const eq::Frames& inFrames);

    void _singleFragmentCompositing(const eq::Frames& frames,
                                    const FramePositionFunctor& framePosition);

    eq::Vector2i _getJitterStep() const;

    void _initJitter();

    void _clearAccumBuffer();

    bool _initAccum();

    bool _isAccumDone() const;

    osg::ref_ptr<osg::Image> _grabFrame();

    void _grabFrameSendEvent();

    void _writeFrame(const eq::uint128_t& frameID);

    void _writeFrame(const std::string& fileName);
};

//! A wrapper around a Channel* as an osg::Referenced
/*! This is used to make it possible pass a Channel* as user data to
  osg::Object::setUserData */
struct ChannelRef : public osg::Referenced
{
    ChannelRef(Channel* channel_)
        : channel(channel_)
    {
    }
    Channel* operator->() { return channel; }
    Channel& operator*() { return *channel; }
    const Channel* operator->() const { return channel; }
    const Channel& operator*() const { return *channel; }
    Channel* const channel;
};
}
}
}
#endif
