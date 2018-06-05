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

#ifndef RTNEURON_OSGEQ_CONFIG_H
#define RTNEURON_OSGEQ_CONFIG_H

#include "FrameData.h"
#include "types.h"

#include "../../AttributeMap.h"

#include <memory>

#include <eq/admin/types.h>
#include <eq/eq.h>
#include <eq/version.h>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
//! The Equalizer config object of RTNeuron.
/**
    This class is responsible of doing the cluster initialization of some
    objects such as the initialization data, FrameData and ClusterObjectManager

    The Config also collects the GUI events recevied and pushes them to the
    event queue stored by the active View to which they can be attributed.
    This events are processed later one inside handleEvents calling
    View::processEvents. The return value of View::processEvents can set
    to true the return value of Channel::needsRedraw. Most events can be
    masked to not cause a redraw.
    To provide fine control of the frames produced, redraw events are masked.
    Client::resume needs to be called to unmask them.
*/
class Config : public eq::Config
{
public:
    /*--- Public constructors/destructor ---*/

    /**
       @param parent
       @param attributes The attribute map for InitData
     */
    Config(eq::ServerPtr parent);

    virtual ~Config();

    /*--- Public member functions ---*/

    /**
       Inits the configuration.
       This function is only called from the application node.
     */
    virtual bool init(const eq::uint128_t& initID);

    virtual bool exit();

    bool isDone() const { return _done; }
    void setDone();

    virtual uint32_t startFrame(const eq::uint128_t& frameID);

    void setInitData(const InitDataPtr& initData);

    const InitData& getInitData() const { return *_initData; }
    /**
       Master copy of the frame data.

       Only modified by the application node and distributed to all nodes.
       This data mustn't be read from the rendering code because if the
       application node is also a rendering node, the values retreived from
       this may be inconsistent with the values used by the rest of the
       nodes.
       Use always Node::getFrameData instead, which is guaranteed to be
       in synch at the frame level.
     */
    FrameData& getFrameData() { return _frameData; }
    const FrameData& getFrameData() const { return _frameData; }
    Client* getClient();

    /**
       Map per-config data to the local node process
    */
    bool mapDistributedObjects(const eq::uint128_t& initDataID);

    void unmapDistributedObjects();

    bool handleEvent(eq::EventICommand command) override;
    bool handleEvent(eq::EventType type, const eq::KeyEvent&) override;
    bool handleEvent(eq::EventType type, const eq::PointerEvent&) override;
    bool handleEvent(eq::EventType type, const eq::Event&) override;

    /**
       Overriden to update views after events have been processed.

       This call updates the flag for redraw.
    */
    void handleEvents() override;

    bool needsRedraw() const;

    bool isRedrawMasked() const { return _maskRedraw; }
    /**
       Sets the active layout
    */
    void useLayout(const std::string& name);

    eq::admin::Config* getAdminConfig();

private:
    /*--- Private member attributes ---*/

    LB_TS_VAR(_clientThread);

    InitDataPtr _initData;

    bool _done;

    FrameData _frameData;
    osg::Timer _timer; /* Only meaningful in the application node */

#ifdef RTNEURON_USE_VRPN
    std::unique_ptr<Tracker> _tracker;
#endif

    /* JH: This plain pointers may be unsafe if dynamic view configuration is
       possible but a View is not a Referenced object. I need to understand
       better the lifetime cycle of these objects. */
    View* _lastFocusedView;
    eq::fabric::RenderContext _lastFocusedContext;

    bool _needsRedraw;
    bool _needsRedrawUnmaskable;
    bool _maskRedraw;
    int32_t _remainingAAFrames;

    eq::admin::ClientPtr _adminClient;
    eq::admin::ServerPtr _adminServer;

    /*--- Private member functions ---*/

    void _pushFrameEventToViews();

    void _handleUserInputEvent(const eq::EventICommand&);

    /** Function for debugging observer position */
    void _setHeadMatrix(const eq::Matrix4f& matrix);
    const eq::Matrix4f& _getHeadMatrix() const;

    void _switchLayout(int32_t increment);
};
}
}
}
#endif
