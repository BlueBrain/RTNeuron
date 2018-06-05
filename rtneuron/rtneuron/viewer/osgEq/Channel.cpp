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
#include "Client.h"
#include "Compositor.h"
#include "Config.h"
#include "ConfigEvent.h"
#include "InitData.h"
#if defined OSG_GL3_AVAILABLE and defined USE_CUDA
#include "MultiFragmentFunctors.h"
#endif
#include "Node.h"
#include "NodeViewer.h"
#include "Scene.h"
#include "View.h"

#include "config/Globals.h"
#include "config/constants.h"
#include "ui/Pointer.h"
#include "util/log.h"

#include <osg/CullSettings>
#include <osg/ValueObject>
#include <osg/Version>
#include <osgDB/WriteFile>
#include <osgUtil/SceneView>
#include <osgViewer/Renderer>
#include <osgViewer/View>
#include <osgViewer/ViewerEventHandlers>
#if OSG_VERSION_GREATER_OR_EQUAL(3, 5, 0)
#include <osg/ContextData>
#endif

#include <lunchbox/clock.h>
#include <lunchbox/rng.h>

#include <eq/compositor.h>
#include <eq/eq.h>

#include <boost/shared_array.hpp>

#include <algorithm>
#include <sstream>
#include <thread>

#include <osg/io_utils>
//#define DEBUG_ROI

#ifdef RTNEURON_USE_LIBJPEGTURBO
#include <turbojpeg.h>
#endif

namespace osgUtil
{
class RenderLeaf;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
#if !defined OSG_GL3_AVAILABLE || !defined USE_CUDA
class MultiFragmentFunctors
{
};
#endif

/*
   Helper functions and classes
*/
namespace
{
const bool INVARIANT_IDLE_AA = ::getenv("RTNEURON_INVARIANT_IDLE_AA") != 0;
const bool SET_MINIMUM_VIEWPORT =
    ::getenv("RTNEURON_SET_MINIMUM_ALPHA_BLENDING_VIEWPORT") != 0;

ScenePtr _getScene(Channel& channel)
{
    unsigned int sceneID = static_cast<View*>(channel.getView())->getSceneID();
    if (sceneID == UNDEFINED_SCENE_ID)
        return ScenePtr();

    Node* node = static_cast<Node*>(channel.getPipe()->getNode());
    return node->findScene(sceneID);
}

const FrameData& _getFrameData(const Channel& channel)
{
    const Pipe* pipe = static_cast<const Pipe*>(channel.getPipe());
    return pipe->getFrameData();
}

int _getFramePosition(const eq::Frame& frame)
{
    return frame.getFrameData()->getContext().range.start;
}

bool _is2DCompositing(const eq::Frames& frames)
{
    for (const auto& frame : frames)
    {
        if (frame->getFrameData()->getContext().range != eq::Range::ALL)
            return false;
    }
    return true;
}

int _findCompositingPosition(const Channel& channel, const ScenePtr& scene,
                             const eq::Range& range)
{
    if (!scene)
        /* Unknown compositing order */
        return 0;

    const osg::Matrix view(channel.getHeadTransform().data());
    const osg::Matrix& model =
        dynamic_cast<const View*>(channel.getView())->getModelMatrix();
    return scene->computeCompositingPosition(model * view, range);
}

#ifndef NDEBUG
osg::StateSet* getOrCreate2DStateSet()
{
    static osg::ref_ptr<osg::StateSet> s_2DState;

    if (s_2DState.valid())
        return s_2DState;

    s_2DState = new osg::StateSet();

    osg::Program* program = new osg::Program();
    osg::Shader* vert = new osg::Shader(osg::Shader::VERTEX);
#ifdef OSG_GL3_AVAILABLE
    vert->setShaderSource(
        "#version 410\n"
        "in vec4 osg_Vertex;\n"
        "in vec4 osg_Color;\n"
        "{\n"
        "   color = osg_Color;\n"
        "   gl_Position = osg_Vertex;\n"
        "}");
#else
    vert->setShaderSource(
        "varying vec4 color;\n"
        "void main()"
        "{\n"
        "   color = gl_Color;\n"
        "   gl_Position = gl_Vertex;\n"
        "}");
#endif
    program->addShader(vert);
    osg::Shader* frag = new osg::Shader(osg::Shader::FRAGMENT);
#ifdef OSG_GL3_AVAILABLE
    frag->setShaderSource(
        "#version 410\n"
        "in vec4 inColor;\n"
        "out vec4 outColor;\n"
        "void main() { outcolor = inColor; }");
#else
    frag->setShaderSource(
        "varying vec4 color;\n"
        "void main() { gl_FragColor = color; }");
#endif
    program->addShader(frag);

    s_2DState->setAttributeAndModes(program);

    return s_2DState;
}
#endif

/*
  A custom SceneView which doesn't override the color mask applied to
  the camera.
  This class doesn't care about stereo rendering. That's Equalizer's
  responsibility.
*/
class SceneView : public osgUtil::SceneView
{
public:
    void cull()
    {
        _dynamicObjectCount = 0;

        if (_camera->getNodeMask() == 0)
            return;

        osg::State* state = _renderInfo.getState();
        state->setFrameStamp(_frameStamp);
        if (_displaySettings.valid())
            state->setDisplaySettings(_displaySettings);

        _renderInfo.setView(_camera->getView());

        // update the active uniforms
        updateUniforms();

        LBASSERT(_renderInfo.getState());

        if (!_localStateSet)
            _localStateSet = new osg::StateSet;

        if (!_cullVisitor)
        {
            OSG_INFO << "Warning: no valid osgUtil::SceneView:: attached,"
                        " creating a default CullVisitor automatically."
                     << std::endl;
            _cullVisitor = osgUtil::CullVisitor::create();
        }
        if (!_stateGraph)
        {
            OSG_INFO << "Warning: no valid osgUtil::SceneView:: attached,"
                        " creating a global default StateGraph automatically."
                     << std::endl;
            _stateGraph = new osgUtil::StateGraph;
        }
        if (!_renderStage)
        {
            OSG_INFO
                << "Warning: no valid osgUtil::SceneView::_renderStage"
                   " attached, creating a default RenderStage automatically."
                << std::endl;
            _renderStage = new osgUtil::RenderStage;
        }

        _cullVisitor->setTraversalMask(_cullMask);
        bool computeNearFar =
            cullStage(getProjectionMatrix(), getViewMatrix(), _cullVisitor,
                      _stateGraph, _renderStage, getViewport());

        if (computeNearFar)
        {
            osgUtil::CullVisitor::value_type zNear =
                _cullVisitor->getCalculatedNearPlane();
            osgUtil::CullVisitor::value_type zFar =
                _cullVisitor->getCalculatedFarPlane();
            _cullVisitor->clampProjectionMatrix(getProjectionMatrix(), zNear,
                                                zFar);
        }
    }

    virtual void draw()
    {
        if (_camera->getNodeMask() == 0)
            return;

        osg::State* state = _renderInfo.getState();
#if OSG_VERSION_GREATER_OR_EQUAL(3, 5, 0)
        osg::get<osg::ContextData>(state->getContextID())
            ->newFrame(state->getFrameStamp());
#else
        osg::GLBufferObjectManager::getGLBufferObjectManager(
            state->getContextID())
            ->newFrame(state->getFrameStamp());

        osg::Texture::getTextureObjectManager(state->getContextID())
            ->newFrame(state->getFrameStamp());
        osg::GLBufferObjectManager::getGLBufferObjectManager(
            state->getContextID())
            ->newFrame(state->getFrameStamp());
#endif

        if (!_initCalled)
            init();

        /* Note, to support multi-pipe systems the deletion of OpenGL display
           list and texture objects is deferred until the OpenGL context is the
           correct context for when the object were originally created.
           Here we know what context we are in so can flush the appropriate
           caches. */

        if (_requiresFlush)
        {
            double availableTime = 0.005;
            flushDeletedGLObjects(availableTime);
        }

        /* Assume the the draw which is about to happen could generate GL
           objects that need flushing in the next frame. */
        _requiresFlush = _automaticFlush;

        osgUtil::RenderLeaf* previous = NULL;

        _renderStage->setDrawBufferApplyMask(false);
        _renderStage->setReadBufferApplyMask(false);

        _localStateSet->setAttribute(getViewport());

        _renderStage->setColorMask(_camera->getColorMask());

        _renderStage->drawPreRenderStages(_renderInfo, previous);
        _renderStage->draw(_renderInfo, previous);

        /* Reapply the default OGL state. */
        state->popAllStateSets();
        state->apply();

        if (state->getCheckForGLErrors() != osg::State::NEVER_CHECK_GL_ERRORS &&
            state->checkGLErrors("end of SceneView::draw()"))
        {
            // go into debug mode of OGL error in a fine grained way to help
            // track down OpenGL errors.
            state->setCheckForGLErrors(osg::State::ONCE_PER_ATTRIBUTE);
        }
    }
};

/*
  A custom Renderer which uses the scene view from above
*/
class Renderer : public osgViewer::Renderer
{
public:
    Renderer(osg::Camera* camera)
        : osgViewer::Renderer(camera)
    {
        /* Replacing the original osgUtil::SceneView with our custom one */
        _sceneView[0] = new SceneView();
        _sceneView[1] = new SceneView();

        _sceneView[0]->setFrameStamp(new osg::FrameStamp());
        _sceneView[1]->setFrameStamp(new osg::FrameStamp());

        osgViewer::View* view =
            static_cast<osgViewer::View*>(_camera->getView());
        bool automaticFlush =
            view->getViewerBase()->getIncrementalCompileOperation() == 0;

        osg::StateSet* global_stateset = 0;
        osg::StateSet* secondary_stateset = 0;
        global_stateset = camera->getOrCreateStateSet();
        _sceneView[0]->setAutomaticFlush(automaticFlush);
        _sceneView[0]->setGlobalStateSet(global_stateset);
        _sceneView[0]->setSecondaryStateSet(secondary_stateset);
        _sceneView[1]->setAutomaticFlush(automaticFlush);
        _sceneView[1]->setGlobalStateSet(global_stateset);
        _sceneView[1]->setSecondaryStateSet(secondary_stateset);

        unsigned int sceneViewOptions = osgUtil::SceneView::HEADLIGHT;
        _sceneView[0]->setDefaults(sceneViewOptions);
        _sceneView[1]->setDefaults(sceneViewOptions);

        osg::DisplaySettings* ds =
            _camera->getDisplaySettings()
                ? _camera->getDisplaySettings()
                : (view->getDisplaySettings()
                       ? view->getDisplaySettings()
                       : osg::DisplaySettings::instance().get());

        _sceneView[0]->setDisplaySettings(ds);
        _sceneView[1]->setDisplaySettings(ds);

        _sceneView[0]->setCamera(_camera.get(), false);
        _sceneView[1]->setCamera(_camera.get(), false);

        osg::ref_ptr<osgUtil::CullVisitor::Identifier> identifier =
            _sceneView[0]->getCullVisitor()->getIdentifier();
        _sceneView[1]->getCullVisitor()->setIdentifier(identifier);

        /* No stereo setup on purpose. */
        _availableQueue._queue.clear();
        _availableQueue.add(_sceneView[0]);
        _availableQueue.add(_sceneView[1]);
    }
};

#define LOG_CHANNEL \
    core::Globals::profileChannels() && _log << std::this_thread::get_id()

class Profiler
{
public:
    Profiler(const std::string& operation, const size_t frame,
             std::ostream& log)
        : _log(log)
    {
        if (core::Globals::profileChannels())
        {
            _operation = operation;
            _frame = frame;
        }
    }

    ~Profiler()
    {
        if (core::Globals::profileChannels())
        {
            LOG_CHANNEL << ' ' << _frame << ' ' << _operation << ' '
                        << _clock.getTimef() << std::endl;
        }
    }

private:
    lunchbox::Clock _clock;
    std::string _operation;
    size_t _frame;
    std::ostream& _log;
};
}

/*
   Nested classes
*/

Channel::Accum::Accum()
    : remainingSteps(0)
    , stepsDone(0)
    , transfer(false)
{
}

Channel::Accum::~Accum()
{
}

/*
  Constructor/destructor
*/
Channel::Channel(eq::Window* parent)
    : eq::Channel(parent)
    , _texture(0)
{
}

Channel::~Channel()
{
}

/*
  Member fuctions
*/
bool Channel::configInit(const eq::uint128_t& initDataID)
{
    if (!eq::Channel::configInit(initDataID))
        return false;

    Window* window = static_cast<Window*>(getWindow());

    /* Setting up the view and camera */
    _view = new osgViewer::View;
    _view->setUserData(new ChannelRef(this));
    static_cast<Node*>(getNode())->addView(_view);

    _camera = _view->getCamera();
    osg::GraphicsContext* context = window->embedded();
    context->getState()->initializeExtensionProcs();
    _camera->setGraphicsContext(context);
    _camera->setAllowEventFocus(true);
    /* No clear operation will be performed by OSG itself */
    _camera->setClearMask(0);
    /** \todo Move this out of here. Add it to the view? */
    osg::CullSettings cullSettings;
    using core::CULL_AND_DRAW_MASK;
    cullSettings.setCullMask(CULL_AND_DRAW_MASK);
    _camera->setCullSettings(cullSettings);

    /* Replacing the default SceneView with our custom one. The renderer
       has to be replaced for that purpose. */
    _camera->setRenderer(new Renderer(_camera));

    /* Setting up lighting */
    _view->setLightingMode(osg::View::NO_LIGHT);
    _lightSource = new osg::Uniform("lightSource", osg::Vec3(0.f, 0.f, 0.f));
    _camera->getOrCreateStateSet()->addUniform(_lightSource.get());

    /* Setup for readback frame */
    eq::FrameDataPtr frameData = new eq::FrameData;
    frameData->setBuffers(eq::Frame::Buffer::color);
    _frame.setFrameData(frameData);

    /* Logging file */
    if (core::Globals::profileChannels())
    {
        std::string name = getName();
        std::replace(name.begin(), name.end(), ' ', '_');
        _log.open(name + ".txt", std::ios_base::out);
        assert(_log);
    }

    Config* config = static_cast<Config*>(getConfig());
    const InitData& initData = config->getInitData();
    if (getView() && initData.shareContext &&
        /* Don't create the texture for snapshot channels. */
        getName() != "aux channel")
    {
        _texture =
            getObjectManager().newEqTexture(this, GL_TEXTURE_RECTANGLE_ARB);
    }
    return true;
}

bool Channel::configExit()
{
    if (_texture)
        getObjectManager().deleteEqTexture(this);

    _bareScene = 0;

    eq::FrameDataPtr frameData = _frame.getFrameData();
    frameData->resetPlugins();
    frameData->flush();
    _frame.clear();

    static_cast<Node*>(getNode())->removeView(_view);
    _camera = 0;
    _view = 0;

    for (size_t i = 0; i < eq::NUM_EYES; ++i)
    {
        Accum& accum = _accum[i];
        if (accum.buffer)
            accum.buffer->exit();
    }

#if defined OSG_GL3_AVAILABLE and defined USE_CUDA
    _multiFragmentFunctors.reset();
#endif

    if (core::Globals::profileChannels())
        _log.close();

    return eq::Channel::configExit();
}

bool Channel::useOrtho() const
{
    const View* view = dynamic_cast<const View*>(getView());
    return view->useOrtho();
}

void Channel::frameClear(const eq::uint128_t&)
{
    _updateClearColor();

    ScenePtr scene = _getScene(*this);

    /* Setting the clear color differently for DB spatial decompositions
       and the rest. */
    if (_sourceDrawRange == eq::Range::ALL ||
        (scene && scene->getDBCompositingTechnique() != Scene::ORDER_DEPENDENT))
    {
        glClearColor(_clearColor[0], _clearColor[1], _clearColor[2],
                     _clearColor[3]);
    }
    else
    {
        glClearColor(0.f, 0.f, 0.f, 0.f);
    }

    resetRegions();

    applyBuffer();
    glEnable(GL_SCISSOR_TEST);
    applyViewport();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Channel::frameDraw(const eq::uint128_t& value)
{
    LB_TS_THREAD(_thread);

    Profiler profile("frameDraw", value.low(), _log);

    _initJitter();
    if (_isAccumDone())
        return;

    Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    accum.stepsDone =
        std::max(accum.stepsDone, getSubPixel().size * getPeriod());
    /* With idle AA, we need to accumulate at least the background even when
       the scene is not ready, otherwise the displayed buffer is corrupted. */
    accum.transfer = true;

    /* Setting frustum */
    eq::Frustumf frustum = getFrustum();
    const eq::Vector2f jitter = getJitter();
    frustum.jitter(jitter);

    /* Fixing near/far planes if the scene range is not [0, 1] and
       the DB decomposition needs per pixel sorting. */
    ScenePtr scene = _getScene(*this);

    if (scene && getRange() != eq::Range::ALL &&
        scene->getDBCompositingTechnique() != Scene::ORDER_DEPENDENT)
    {
        frustum.nearPlane() = 0.1f;
        frustum.farPlane() = 100000.f;
    }

    const osg::Matrix projection(
        useOrtho() ? frustum.computeOrthoMatrix().data()
                   : frustum.computePerspectiveMatrix().data());
    _camera->setProjectionMatrix(projection);
    /* Correcting the view matrix from the camera manipulator
       with the head position.
       Equalizer stores the matrices column-major with right product
       notation. OSG uses row-major storage with left product notation.
       That means that the actual flat storage is the same for both. */

    osg::Matrix view(getHeadTransform().data());
    osg::Matrix model = dynamic_cast<View*>(getView())->getModelMatrix();
    {
        /* Mutual exclusion with the processEvent function. */
        /** \todo Review if needed */
        lunchbox::ScopedWrite _scope(_lock);
        _camera->setViewMatrix(model * view);
    }

    /* Camera data values */
    _camera->setUserValue("eye", (unsigned int)getEye());
    SceneDecoratorPtr decorator = static_cast<View*>(getView())->getDecorator();
    if (decorator)
    {
        decorator->updateCamera(_camera);
        decorator->update();
    }

    /* Color mask */
    eq::ColorMask mask = getDrawBufferMask();
    _camera->setColorMask(mask.red, mask.green, mask.blue, true);

    /* Only a single point light is supported for the moment. */
    /** \todo Configurable light source position */
    osg::Vec4 light = osg::Vec4(0.f, 0.f, 0.f, 1.f) * view;
    _lightSource->set(osg::Vec3(light[0], light[1], light[2]));

    /* Setting viewports. */
    const eq::PixelViewport& viewport = getPixelViewport();

    _camera->setViewport(viewport.x, viewport.y, viewport.w, viewport.h);

    if (SET_MINIMUM_VIEWPORT)
    {
        /* The purpose of this code is to avoid frame stuterring in the first
           frames of dynamic 2D modes. However for very large viewports it
           may be counterproductive. */
        const eq::PixelViewport pvp = getWindow()->getPixelViewport();
        _camera->setUserValue("max_viewport_hint", osg::Vec2d(pvp.w, pvp.h));
    }

    if (!_setupChannelScene(value.low()))
        /* Nothing to draw yet */
        return;

    _setupPostRenderEffects();

    /* Rendering this camera. Note: this call cannot overlap the rendering
       of any other camera from the same window than this channel. */
    Node* node = static_cast<Node*>(getPipe()->getNode());
    NodeViewer* viewer = node->getViewer();
    viewer->renderTraversal(_camera);

    _postDraw();

    _drawRange = getRange();
}

void Channel::frameAssemble(const eq::uint128_t& frameID,
                            const eq::Frames& inFrames)
{
    if (!core::Globals::isCompositingEnabled() || _isAccumDone())
        return;

    Profiler profiler("frameAssemble", frameID.low(), _log);

    ScenePtr scene = _getScene(*this);

    if (_is2DCompositing(inFrames))
    {
        _singleFragmentCompositing(inFrames, FramePositionFunctor());
        return;
    }

    switch (scene->getDBCompositingTechnique())
    {
    case Scene::ORDER_INDEPENDENT_SINGLE_FRAGMENT:
    {
        _singleFragmentCompositing(inFrames, FramePositionFunctor());
        break;
    }
    case Scene::ORDER_DEPENDENT:
    {
        eq::Frames frames = inFrames;
        const int position = _findCompositingPosition(*this, scene, _drawRange);
        /* The destination channel is also part of a DB compound.
           Starting the readback of this channel and adding it to the
           frame list.
           It's possible to know if this is needed checking if the
           compositing position is 0, but at the current moment the only
           way to render correctly over a non-black background is doing a
           readback and then clearing. */
        _frame.clear();

        /* It's not possible to know which portion needs readback until
           we receive the other frames, furthermore, we also have the
           background issue, so we read the whole viewport. */
        _frame.setOffset(eq::Vector2i(0, 0));
        eq::FrameDataPtr data = _frame.getFrameData();
        data->setPixelViewport(getPixelViewport());
        data->getContext().range = eq::Range(position, position);
#if EQ_VERSION_LT(1, 7, 0)
        eq::Window::ObjectManager* glObjects = getObjectManager();
#else
        eq::util::ObjectManager& glObjects = getObjectManager();
#endif
        _frame.readback(glObjects, getDrawableConfig(),
                        eq::PixelViewports(1, getPixelViewport()),
                        getContext());
        frames.push_back(&_frame);
        /* Clearing the background */
        glClearColor(_clearColor[0], _clearColor[1], _clearColor[2],
                     _clearColor[3]);
        glClear(GL_COLOR_BUFFER_BIT);

        _singleFragmentCompositing(frames, _getFramePosition);
        break;
    }
    case Scene::ORDER_INDEPENDENT_MULTIFRAGMENT:
    {
#if defined OSG_GL3_AVAILABLE and defined USE_CUDA
        _multiFragmentCompositing(inFrames);
        break;
#endif
    }
    }
}

void Channel::frameReadback(const eq::uint128_t& frameID,
                            const eq::Frames& frames)
{
    if (!core::Globals::isReadbackEnabled() || _isAccumDone())
        return;

    Profiler profiler("frameReadback", frameID.low(), _log);

    ScenePtr scene = _getScene(*this);
    if (!_is2DCompositing(frames) && scene)
    {
        switch (scene->getDBCompositingTechnique())
        {
        case Scene::ORDER_DEPENDENT:
            for (auto frame : frames)
            {
                eq::FrameDataPtr data = frame->getFrameData();
                /* Assigning the compositing order to this frame if needed. */
                const int position =
                    _findCompositingPosition(*this, scene,
                                             data->getContext().range);

                /* Currently, the only field where this user data can be
                   transferred is the draw range. */
                data->getContext().range = eq::Range(position, position);
                /* Disabling depth buffer readback */
                frame->disableBuffer(eq::Frame::Buffer::depth);
            }
            break;
        case Scene::ORDER_INDEPENDENT_MULTIFRAGMENT:
            break; // TODO
        default:;
        }
    }
    eq::Channel::frameReadback(frameID, frames);
}

void Channel::frameStart(const eq::uint128_t& frameID,
                         const uint32_t frameNumber)
{
    for (size_t i = 0; i < eq::NUM_EYES; ++i)
        _accum[i].stepsDone = 0;

    eq::Channel::frameStart(frameID, frameNumber);
}

void Channel::frameViewStart(const eq::uint128_t& frameID)
{
    _currentDestinationPVP = getPixelViewport();
    _initJitter();

    eq::Channel::frameViewStart(frameID);
}

void Channel::frameFinish(const eq::uint128_t& frameID,
                          const uint32_t frameNumber)
{
    for (size_t i = 0; i < eq::NUM_EYES; ++i)
    {
        Accum& accum = _accum[i];
        if (accum.remainingSteps > 0)
            accum.remainingSteps -=
                std::min(int(accum.stepsDone), accum.remainingSteps);
    }
    eq::Channel::frameFinish(frameID, frameNumber);
}

void Channel::frameViewFinish(const eq::uint128_t& frameID)
{
    const FrameData& frameData = _getFrameData(*this);

    View* view = static_cast<View*>(getView());

    Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    if (accum.buffer)
    {
        const eq::PixelViewport& pvp = getPixelViewport();
        const bool isResized = accum.buffer->resize(pvp);

        if (isResized)
        {
            accum.buffer->clear();
            accum.remainingSteps = view->getMaxIdleSteps();
            accum.stepsDone = 0;
        }
        else if (frameData.isIdle)
        {
            setupAssemblyState();
            applyViewport();

            if (!_isAccumDone() && accum.transfer)
                accum.buffer->accum();
            accum.buffer->display();

            resetAssemblyState();
        }
    }

    std::string snapshot;
    const bool writeFrames = view->getWriteFrames();
    bool grabFrame = false;
    if (view->getMaxIdleSteps() == 0 || !view->getSnapshotAtIdle() ||
        _isAccumDone())
    {
        snapshot = view->getSnapshotName(getName());
        grabFrame = view->getGrabFrame();
    }

    if (frameData.statistics || !snapshot.empty() || writeFrames || grabFrame)
    {
        applyBuffer();
        applyViewport();
    }

    if (grabFrame)
    {
        _grabFrameSendEvent();
        view->setGrabFrame(false);
    }
    if (writeFrames)
        _writeFrame(frameID);

    if (!snapshot.empty())
    {
        _writeFrame(snapshot);
        view->snapshotDone(*this);
    }

    /* Drawing statistics */
    if (frameData.statistics)
        drawStatistics();

    int32_t remainingSteps = 0;
    if (frameData.isIdle)
    {
        for (size_t i = 0; i < eq::NUM_EYES; ++i)
            remainingSteps = std::max(remainingSteps, _accum[i].remainingSteps);
    }
    else
    {
        remainingSteps = static_cast<View*>(getView())->getMaxIdleSteps();
    }

    /* if _jitterStep == 0 and no user redraw event happened, the app will
       exit FSAA idle mode and block on the next redraw event. */
    eq::Config* config = getConfig();
    config->sendEvent(ConfigEvent::IDLE_AA_LEFT) << remainingSteps;

    eq::Channel::frameViewFinish(frameID);

    if (_texture)
    {
        const eq::PixelViewport& pvp = getPixelViewport();
        _texture->copyFromFrameBuffer(GL_RGBA, pvp);
        config->sendEvent(ConfigEvent::ACTIVE_VIEW_TEXTURE)
            << _texture->getName();
    }
}

void Channel::_setupPostRenderEffects()
{
    osgViewer::Renderer* renderer =
        (osgViewer::Renderer*)_camera->getRenderer();
    const View* eqView = static_cast<View*>(getView());
    if (eqView->useDOF())
    {
        if (!_depthOfField)
        {
            /* Setting the ClearColor before DOF setup */
            _camera->setClearColor(_clearColor);
            _depthOfField.reset(new core::DepthOfField(_camera));
        }
        _depthOfField->setFocalDistance(eqView->getFocalDistance());
        _depthOfField->setFocalRange(eqView->getFocalRange());
    }
    else
    {
        _depthOfField.reset();
        renderer->getSceneView(0)->getRenderStage()->setFrameBufferObject(0);
    }

    renderer->getSceneView(0)->getRenderStage()->setCameraRequiresSetUp(
        eqView->useDOF());
}

bool Channel::_setupChannelScene(const uint32_t frameNumber)
{
    View* view = static_cast<View*>(getView());

    ScenePtr scene = _getScene(*this);
    const eq::Range& range = getRange();
    osg::ref_ptr<osg::Node> scenegraph =
        scene ? scene->getOrCreateSubSceneNode(range) : 0;

    if (scene)
        scene->updateCameraData(range, _camera);

    bool newScene = scenegraph != _bareScene;

    osgViewer::View* osgView =
        static_cast<osgViewer::View*>(_camera->getView());

    if (!scenegraph)
    {
        osgView->setSceneData(0);
        _bareScene = 0;
        return !newScene;
    }

    const bool isDB = range != eq::Range::ALL;
    const auto compositingTechnique = scene->getDBCompositingTechnique();

    if (newScene)
    {
        /* The subscene is created with one frame delay */
        getConfig()->sendEvent(ConfigEvent::REQUEST_REDRAW);

        _sourceDrawRange = range;
        _bareScene = scenegraph;

        /** \bug This works for now, but if the decorator is changed the
            channel won't notice it */
        SceneDecoratorPtr decorator =
            static_cast<View*>(getView())->getDecorator();
        assert(decorator);
        osg::Group* sceneData = decorator->decorateScenegraph(scenegraph);
        osgView->setSceneData(sceneData);

        /* If the range is not 0, 1 we are using a DB decomposition mode.
           Setting the camera to not adjust near and far planes if compositing
           needs to do per pixel sorting. */
        if (isDB && compositingTechnique != Scene::ORDER_DEPENDENT)
        {
            _camera->setComputeNearFarMode(
                osg::Camera::DO_NOT_COMPUTE_NEAR_FAR);
            for (int i = 0; i < 2; ++i)
            {
                osgUtil::SceneView* sv =
                    static_cast<osgViewer::Renderer*>(_camera->getRenderer())
                        ->getSceneView(i);
                osgUtil::CullVisitor* cullVisitor = sv->getCullVisitor();
                cullVisitor->setComputeNearFarMode(
                    osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
            }
        }
    }

#if defined OSG_GL3_AVAILABLE and defined USE_CUDA
    if (isDB && compositingTechnique == Scene::ORDER_INDEPENDENT_MULTIFRAGMENT)
    {
        osg::GraphicsContext* context = _camera->getGraphicsContext();
        if (!_multiFragmentFunctors)
            _multiFragmentFunctors.reset(new MultiFragmentFunctors(context));
        _multiFragmentFunctors->setup(*scene);
    }
    else
        _multiFragmentFunctors.reset();
#endif

    if (_depthOfField)
        _depthOfField->update();

    /* Updating the dynamic elements of the subscene range for the current
       frame (e.g. simulation data). */
    scene->channelSync(range, frameNumber, _getFrameData(*this));

    /* Adding the pointer geometry if needed.
       This code is outside the if block above because the pointer can be
       added to the view at any moment. If added after the scene, setPointer
       may not be called in time.
       Although it's possible to force to always set the pointer first, the
       code below shouldn't cause any noticeable overhead. */

    /** \bug The pointer geometry is not removed once added */
    PointerPtr pointer = view->getPointer();
    osg::Node* pointerGeom = pointer ? pointer->getGeometry() : 0;
    osg::Group* sceneData = osgView->getSceneData()->asGroup();
    assert(sceneData);
    /* This code assumes that the top root node has very few children */
    if (pointerGeom && !sceneData->asGroup()->containsNode(pointerGeom))
        sceneData->addChild(pointerGeom);

    return !newScene;
}

osg::Vec2 transform(const osg::Vec3& o, const osg::Matrix& m,
                    const osg::Matrix& p, osg::Viewport& vp)
{
    osg::Vec4 q = osg::Vec4(o.x(), o.y(), o.z(), 1) * m * p;
    q /= q.w();
    return osg::Vec2((q.x() + 1.f) * 0.5f * vp.width(),
                     (q.y() + 1.f) * 0.5f * vp.height());
}

void Channel::_updateClearColor()
{
    osg::Vec4 color;
    static bool s_taintChannels = ::getenv("EQ_TAINT_CHANNELS");
    if (s_taintChannels)
    {
        const eq::Vector3ub c = getUniqueColor();
        color[0] = c.r() / 255.f;
        color[1] = c.g() / 255.f;
        color[2] = c.b() / 255.f;
    }
    else
    {
        color = static_cast<View*>(getView())->getClearColor();
    }
    _clearColor = color;
}

void Channel::_postDraw()
{
/* Temporary code, used to process and display the fragments without
   network communication. */
#if defined OSG_GL3_AVAILABLE and defined USE_CUDA
    if (_multiFragmentFunctors)
    {
        const auto viewports = PixelViewports{getPixelViewport()};
        auto parts = _multiFragmentFunctors->extractFrameParts(viewports);

        const osg::Vec4& background =
            static_cast<View&>(*getNativeView()).getClearColor();
        for (auto& future : parts)
        {
            const auto part = future.get();
            const eq::PixelViewport& viewport = part.viewport;
            charPtr buffer =
                _multiFragmentFunctors->mergeAndBlend({part}, background);

            eq::PixelData data;
            data.pixels = buffer.get();
            data.internalFormat = GL_RGBA;
            data.externalFormat = GL_RGBA;
            data.pixelSize = 4;
            data.pvp = viewport;

            eq::Image image;
            image.setInternalFormat(eq::Frame::Buffer::color, GL_RGBA);
            image.setStorageType(eq::Frame::TYPE_MEMORY);
            image.setPixelViewport(viewport);
            image.setPixelData(eq::Frame::Buffer::color, data);

            eq::ImageOp imageOp;
            imageOp.image = &image;
            eq::Compositor::setupAssemblyState(viewport, glewGetContext());
            eq::Compositor::assembleImage2D(imageOp, this);
            eq::Compositor::resetAssemblyState();
        }
        _multiFragmentFunctors->cleanup();
    }
#endif

#ifndef NDEBUG
    const FrameData& frameData = _getFrameData(*this);
    if (frameData.drawViewportBorder)
        _drawViewportBorder();
#endif

    if (static_cast<View*>(getView())->useROI())
        _updateRegionOfInterest();
}

void Channel::_drawViewportBorder()
{
#ifndef NDEBUG
    /* This code is a bit verbose but it is compatible with both OpenGL 2
       and OpenGL > 3. */

    /* Creating the geometry object if needed. */
    if (_border.get() == 0)
    {
        osg::StateSet* stateSet = getOrCreate2DStateSet();
        _border = new osg::Geometry();
        _border->setStateSet(stateSet);
        _border->setUseDisplayList(false);
        _border->setUseVertexBufferObjects(false);

        _border->setVertexArray(new osg::Vec2Array(4));

        const eq::Vector3ub c = getUniqueColor();
        osg::Vec4Array* color = new osg::Vec4Array();
        color->push_back(
            osg::Vec4(c.r() / 255.f, c.g() / 255.f, c.b() / 255.f, 1.f));
        _border->setColorArray(color);
        _border->setColorBinding(osg::Geometry::BIND_OVERALL);

        _border->addPrimitiveSet(new osg::DrawArrays(GL_LINE_LOOP, 0, 4));
    }

    /* Updating the vertices. */
    const eq::PixelViewport& vp = getPixelViewport();
    applyViewport();

    osg::Vec2Array& vertices =
        *static_cast<osg::Vec2Array*>(_border->getVertexArray());
    float dx = 1.f / vp.w, dy = 1.f / vp.h;
    vertices[0] = osg::Vec2(-1.f + dx, -1.f + dy);
    vertices[1] = osg::Vec2(1.f - dx, -1.f + dy);
    vertices[2] = osg::Vec2(1.f - dx, 1.f - dy);
    vertices[3] = osg::Vec2(-1.f + dx, 1.f - dy);

    /* Rendering the rectangle. */
    /* This is the same state what is used by osgViewer::Renderer to
       initialize the osgUtil::SceneView object that does the rendering */
    osg::GraphicsContext* context = _camera->getGraphicsContext();
    assert(context);
    osg::State& state = *context->getState();
    state.pushStateSet(_border->getStateSet());
    state.apply();
    osg::RenderInfo renderInfo(&state, _camera->getView());
    _border->draw(renderInfo);

    state.popStateSet();
#endif
}

void Channel::_multiFragmentCompositing(const eq::Frames& inFrames)
{
    (void)inFrames;
}

void Channel::_singleFragmentCompositing(
    const eq::Frames& frames, const FramePositionFunctor& framePosition)
{
    applyBuffer();
    applyViewport();
    setupAssemblyState();

    Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    accum.transfer = true;
    _clearAccumBuffer();

    if (getPixelViewport() != _currentDestinationPVP)
    {
        if (accum.buffer && !accum.buffer->usesFBO())
        {
            LBWARN << "Current viewport different from view viewport, "
                   << "idle anti-aliasing not implemented." << std::endl;
            accum.remainingSteps = 0;
        }

        Compositor::assembleFrames(frames, 0, this, framePosition);

        resetAssemblyState();
        return;
    }

    for (const auto frame : frames)
    {
        const eq::SubPixel& subPixel =
            frame->getFrameData()->getContext().subPixel;

        if (subPixel != eq::SubPixel::ALL)
            accum.transfer = false;

        accum.stepsDone =
            std::max(accum.stepsDone,
                     subPixel.size *
                         frame->getFrameData()->getContext().period);
    }

    Compositor::assembleFrames(frames, accum.buffer.get(), this, framePosition);

    resetAssemblyState();

    _drawRange = getRange();
}

void Channel::_updateRegionOfInterest()
{
#if !defined NDEBUG or defined DEBUG_ROI
    const eq::PixelViewport& vp = getPixelViewport();
    applyViewport();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, vp.w, 0.f, vp.h, -1.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
#endif

    ScenePtr scene = _getScene(*this);
    if (!scene)
    {
        declareRegion(eq::PixelViewport());
        return;
    }
    const osg::Vec4 region = scene->getRegionOfInterest(*this);

    if (region == osg::Vec4(0.f, 0.f, 0.f, 0.f))
        declareRegion(eq::PixelViewport());
    else
        declareRegion(eq::PixelViewport(region[0], region[1],
                                        region[2] - region[0],
                                        region[3] - region[1]));

#if !defined NDEBUG or defined DEBUG_ROI
    const eq::PixelViewport r = getRegion();

    eq::Vector4f rect(r.x + 0.5f, r.y + 0.5f, r.getXEnd() - 0.5f,
                      r.getYEnd() - 0.5f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(rect[0], rect[1], -0.99f);
    glVertex3f(rect[2], rect[1], -0.99f);
    glVertex3f(rect[2], rect[3], -0.99f);
    glVertex3f(rect[0], rect[3], -0.99f);
    glEnd();
#endif
}

eq::Vector2f Channel::getJitter() const
{
    const FrameData& frameData = _getFrameData(*this);
    const Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];

    if (!frameData.isIdle || accum.remainingSteps <= 0)
        return eq::Channel::getJitter();

    const View* view = static_cast<const View*>(getView());
    assert(view);
    if (view->getMaxIdleSteps() == 0)
        return eq::Vector2f();

    const eq::Vector2i jitterStep = _getJitterStep();
    if (jitterStep == eq::Vector2i())
        return eq::Vector2f();

    const eq::PixelViewport& pvp = getPixelViewport();
    const float pvp_w = float(pvp.w);
    const float pvp_h = float(pvp.h);
    const float frustum_w = float((getFrustum().getWidth()));
    const float frustum_h = float((getFrustum().getHeight()));

    const float pixel_w = frustum_w / pvp_w;
    const float pixel_h = frustum_h / pvp_h;

    const float sampleSize = sqrt(view->getMaxIdleSteps());
    const float subpixel_w = pixel_w / sampleSize;
    const float subpixel_h = pixel_h / sampleSize;

    // Sample value randomly computed within the subpixel
    lunchbox::RNG rng;
    const eq::Pixel& pixel = getPixel();
    const float fraction_x = INVARIANT_IDLE_AA ? 0.5 : rng.get<float>();
    const float fraction_y = INVARIANT_IDLE_AA ? 0.5 : rng.get<float>();

    const float i =
        (fraction_x * subpixel_w + float(jitterStep.x()) * subpixel_w) /
        float(pixel.w);
    const float j =
        (fraction_y * subpixel_h + float(jitterStep.y()) * subpixel_h) /
        float(pixel.h);

    return eq::Vector2f(i, j);
}

void Channel::_initJitter()
{
    View* view = static_cast<View*>(getView());
    if (!view)
        return;

    const FrameData& frameData = _getFrameData(*this);
    if (frameData.isIdle)
        return;

    _clearAccumBuffer();
    Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    /* Preparing for next accumulation */
    accum.remainingSteps = view->getMaxIdleSteps();
}

eq::Vector2i Channel::_getJitterStep() const
{
    static const uint32_t _primes[100] = {
        739,  743,  751,  757,  761,  769,  773,  787,  797,  809,  811,  821,
        823,  827,  829,  839,  853,  857,  859,  863,  877,  881,  883,  887,
        907,  911,  919,  929,  937,  941,  947,  953,  967,  971,  977,  983,
        991,  997,  1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
        1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
        1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231,
        1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307,
        1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429,
        1433, 1439, 1447, 1451};

    const eq::SubPixel& subPixel = getSubPixel();
    const uint32_t channelID = subPixel.index;
    if (getView() == 0)
        return eq::Vector2i();

    const View* view = static_cast<const View*>(getView());
    const uint32_t totalSteps = view->getMaxIdleSteps();
    if (totalSteps <= 1)
        return eq::Vector2i();

    const Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    const uint32_t subset = totalSteps / getSubPixel().size;
    const uint32_t index =
        (accum.remainingSteps * _primes[channelID % 100]) % subset +
        (channelID * subset);
    const uint32_t sampleSize = uint32_t(round(sqrt(totalSteps)));
    const int dx = index % sampleSize;
    const int dy = index / sampleSize;

    return eq::Vector2i(dx, dy);
}

void Channel::_clearAccumBuffer()
{
    if (!_initAccum())
        return;

    const FrameData& frameData = _getFrameData(*this);
    if (frameData.isIdle)
        return;

    Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    if (accum.buffer)
        accum.buffer->clear();
}

bool Channel::_initAccum()
{
    /* Only destination channels require an accumulation buffer */
    View* view = static_cast<View*>(getNativeView());
    if (!view)
        return true;

    const eq::Eye eye = getEye();
    Accum& accum = _accum[lunchbox::getIndexOfLastBit(eye)];

    if (accum.buffer)
        return true;

    if (accum.remainingSteps == -1) /* Initialization failed last time */
        return false;

    /* Check for unsupported cases */
    if (!eq::util::Accum::usesFBO(glewGetContext()))
    {
        for (size_t i = 0; i < eq::NUM_EYES; ++i)
        {
            if (_accum[i].buffer)
            {
                LBWARN << "glAccum-based accumulation does not support "
                       << "stereo, disabling idle anti-aliasing." << std::endl;
                for (size_t j = 0; j < eq::NUM_EYES; ++j)
                {
                    _accum[j].buffer.reset();
                    _accum[j].remainingSteps = -1;
                }

                view->setMaxIdleSteps(0);
                return false;
            }
        }
    }

    /* Set up accumulation buffer */
    accum.buffer.reset(new eq::util::Accum(glewGetContext()));
    const eq::PixelViewport& pvp = getPixelViewport();
    LBASSERT(pvp.isValid());

    if (!accum.buffer->init(pvp, getWindow()->getColorFormat()) ||
        accum.buffer->getMaxSteps() < view->getMaxIdleSteps())
    {
        LBWARN << "Accumulation buffer initialization failed, "
               << "idle AA not available." << std::endl;
        accum.buffer.reset();
        accum.remainingSteps = -1;
        return false;
    }

    /* Doing first clear here just in case the first frame is already idle. */
    accum.buffer->clear();
    view->setMaxIdleSteps(
        std::min(view->getMaxIdleSteps(), accum.buffer->getMaxSteps()));

    return true;
}

bool Channel::_isAccumDone() const
{
    const FrameData& frameData = _getFrameData(*this);
    if (!frameData.isIdle)
        return false;

    const eq::SubPixel& subpixel = getSubPixel();
    const Accum& accum = _accum[lunchbox::getIndexOfLastBit(getEye())];
    return int(subpixel.index) >= accum.remainingSteps;
}

void Channel::_grabFrameSendEvent()
{
    unsigned long jpegSize = 0;
    View& view = static_cast<View&>(*getNativeView());
#ifdef RTNEURON_USE_LIBJPEGTURBO
    osg::ref_ptr<osg::Image> image = _grabFrame();
    const osg::Vec4& background = view.getClearColor();

    unsigned char* srcBuf = image->data();
    const int32_t colorComponents =
        osg::Image::computeNumComponents(image->getPixelFormat());
    const int32_t pitch = image->s() * colorComponents;
    const int32_t pixelFormat = background.a() == 1.0 ? TJPF_RGB : TJPF_RGBA;
    const int32_t jpegSubsamp = TJSAMP_444;
    const int32_t jpegQual = 100;
    const int32_t flags = TJXOP_ROT180;

    tjhandle handle = tjInitCompress();
    unsigned char* jpegBuf = 0;
    const int32_t success =
        tjCompress2(handle, srcBuf, image->s(), pitch, image->t(), pixelFormat,
                    &jpegBuf, &jpegSize, jpegSubsamp, jpegQual, flags);
    if (success != 0)
    {
        LBERROR << "libjpeg-turbo image conversion failure" << std::endl;
        getConfig()->sendEvent(ConfigEvent::GRAB_FRAME) << view.getID()
                                                        << uint64_t(jpegSize);
        return;
    }

    getConfig()->sendEvent(ConfigEvent::GRAB_FRAME)
        << view.getID() << uint64_t(jpegSize)
        << co::Array<const uint8_t>(jpegBuf, jpegSize);
    tjFree(jpegBuf);
    tjDestroy(handle);
#else
    LBWARN << "Not linked against libjpeg-turbo for GRAB_FRAME" << std::endl;
    getConfig()->sendEvent(ConfigEvent::GRAB_FRAME) << view.getID() << jpegSize;
#endif
}

void Channel::_writeFrame(const eq::uint128_t& frameID)
{
    LBLOG(core::LOG_FRAME_RECORDING) << " writing frame " << frameID
                                     << std::endl;

    View& view = static_cast<View&>(*getNativeView());

    if (view.resetFrameCounter())
        _frameCounterRef = frameID;

    std::stringstream fileName;
    /** Upper 64 bits ignored */
    const size_t frameNumber = frameID.low() - _frameCounterRef.low();
    /* Replacing spaces by dashes in the channel name */
    const std::string prefix = view.getFilePrefix();
    if (prefix.empty())
    {
        std::string name = view.getName();
        if (!name.empty())
        {
            for (size_t i = 0; i < name.size(); ++i)
                if (name[i] == ' ')
                    name[i] = '-';
        }
        else
        {
            name = core::DEFAULT_FRAME_FILE_NAME_PREFIX;
        }
        fileName << name << '_';
    }
    else
        fileName << prefix << '_';

    fileName << std::setfill('0') << std::setw(6) << frameNumber << '.'
             << view.getFileFormat();
    _writeFrame(fileName.str());
}

osg::ref_ptr<osg::Image> Channel::_grabFrame()
{
    /* The buffer and viewport must have been already applied */
    osg::ref_ptr<osg::Image> image = new osg::Image();

    const eq::PixelViewport& pvp = getPixelViewport();
    const osg::Vec4& background =
        static_cast<View&>(*getNativeView()).getClearColor();

    image->readPixels(pvp.x, pvp.y, pvp.w, pvp.h,
                      background.a() == 1.0 ? GL_RGB : GL_RGBA,
                      GL_UNSIGNED_BYTE);
    if (background.a() != 1.0)
    {
        uint32_t* data = (uint32_t*)image->data();

#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < pvp.w * pvp.h; ++i)
        {
            uint32_t& pixel = data[i];
            const uint32_t alpha = pixel >> 24;
            if (alpha != 0)
            {
                const uint32_t blue = (pixel >> 16) & 0xff;
                const uint32_t green = (pixel >> 8) & 0xff;
                const uint32_t red = pixel & 0xff;
                pixel =
                    ((alpha << 24) | (((255 * blue) / alpha) << 16) |
                     (((255 * green) / alpha) << 8) | ((255 * red) / alpha));
            }
            else
                pixel = 0;
        }
    }

    return image;
}

void Channel::_writeFrame(const std::string& fileName)
{
    osg::ref_ptr<osg::Image> image = _grabFrame();
    osgDB::writeImageFile(*image, fileName);
}
}
}
}
