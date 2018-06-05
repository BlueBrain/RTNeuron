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

#ifndef RTNEURON_OSGEQ_VIEW_H
#define RTNEURON_OSGEQ_VIEW_H

#include "../../types.h"

#include "types.h"

#include <eq/view.h>

#include <lunchbox/thread.h>

#include <osgGA/EventQueue>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>

#include <osg/Version>
#if OSG_VERSION_LESS_THAN(2, 9, 0)
#include <osgGA/MatrixManipulator>
#define CameraManipulator MatrixManipulator
#else
#include <osgGA/CameraManipulator>
#endif

#include <boost/signals2/signal.hpp>
#include <mutex>

namespace osg
{
class Node;
}

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
typedef osg::ref_ptr<osgGA::CameraManipulator> OSGCameraManipulatorPtr;

/**
   This class extends eq::View with the a camera manipulator.

   The Config pushes keyboard/mouse events (converted to
   osgGA::GUIEventAdapter events) objects into the event queue from
   handleEvent.
   At the start of the frame, the views are requested to process the
   events and output a model matrix to use for the view.

   All methods in this class are thread-safe.

   \todo What to do with request continuous update and request redraw?
*/
class View : public eq::View, private osgGA::GUIActionAdapter
{
public:
    /*--- Public declarations ---*/

    typedef void DirtySignalSignature();
    typedef boost::signals2::signal<DirtySignalSignature> DirtySignal;

    /*--- Public constructors/destructor ---*/

    View(eq::Layout* parent);

    ~View();

    /*--- Public member functions ---*/

    /**
       Copies scene and render attributes from another view.

       Attributes copied are:
       - Scene ID
       - Model view matrix
       - Stereo properties (mode and model units)
       - Decoration properties (background color, scene decorator, ...)
       - Other rendering related properties: ROI, idle-AA attributes

       The projection is not copied because it requires additional information
       to preserve the destination aspect ratio properly.
    */
    void copyAttributes(const View& other);

    void pushEvent(EventAdapter* event);

    void pushFrameEvent(const double timestamp);

    void addEventHandler(const EventHandlerPtr& eventHandler);

    /**
       Process the events on this view and returns whether it needs to
       be redrawn or not.

       To be executed only in the application node after the config events
       have been handled inside osgEq::Config.
    */
    bool processEvents();

    /**
       Makes View::update to return true in the next call
    */
    void dirty();

    /**
       Triggers a re-rendering of this view.
    */
    virtual void requestRedraw();

    /**
       Returns the world/local to camera coordinates transformation

       This matrix is independent of the observer matrix. This way the
       model can be moved around while the projection surface can be
       tracked independently.
    */
    const osg::Matrixd& getModelMatrix() const;

    /**
       Sets the world/local to camera coordinates transformation
    */
    void setModelMatrix(const osg::Matrixd& matrix,
                        const bool emitDirty = true);

    void setCameraManipulator(const OSGCameraManipulatorPtr& manipulator);
    OSGCameraManipulatorPtr getCameraManipulator();

    /**
       Sets the projection matrix type to use: orthographic or perspective.
       @param ortho True for orthographic projections, false for perspective.
     */
    void setUseOrtho(bool ortho);

    /**
       @return True if the projection matrix is set to orthographic, false
               otherwise.
    */
    bool isOrtho() const;

    /**
       Sets a wall projection using orthographic projection.
     */
    void setOrthoWall(const eq::Wall& wall);

    /**
       Sets a wall projection using perspective projection.
     */
    void setPerspectiveWall(const eq::Wall& wall);

    /**
       Sets a perspective projection.

       Despite this is not a good idea this call hides
       eq::Frustum::setProjection on purpose in order to set the projection
       type of perspective and request a single redraw.
     */
    void setProjection(const eq::Projection& projection);

    /**
       @return True for orthographic projections, false for perspective.
     */
    bool useOrtho() const { return _useOrtho; }
    /**
       Sets a 3D pointer to render and update by this view.

       Pointer will be rendered on all rendering clients whereas pointer
       interactions will be processed in the application node.
     */
    void setPointer(const PointerPtr& pointer);

    PointerPtr getPointer();

    /**
       Sets the home position of the camera manipulator and resets it
       to that position.
    */
    void setHomePosition(const osg::Vec3d& eye, const osg::Vec3d& center,
                         const osg::Vec3d& up);

    void setSceneID(unsigned int id);

    unsigned int getSceneID() const { return _sceneID; }
    /**
       Changes the clear color and sets the dirty bit of this attribute.

       To be called only from the application node.
     */
    void setClearColor(const osg::Vec4& color);

    const osg::Vec4& getClearColor() const;

    void useDOF(bool use);
    bool useDOF() const { return _useDOF; }
    void setFocalDistance(float value);
    float getFocalDistance() const { return _focalDistance; }
    void setFocalRange(float value);
    float getFocalRange() const { return _focalRange; }
    /**
       To be called only from the application node.
    */
    void setDecorator(const SceneDecoratorPtr& decorator);

    SceneDecoratorPtr getDecorator() { return _decorator; }
    bool useROI() const { return _useROI; }
    /**
       Enable use of regions of interest for framebuffer readback.

       ROI is disabled by default
    */
    void useROI(const bool use);

    void setMaxIdleSteps(const unsigned int steps);

    unsigned int getMaxIdleSteps() const { return _maxIdleSteps; }
    /**
       Sets whether the channel must wait until idle antialias is done when
       snapshot is requested before capturing the rendered image.
     */
    void setSnapshotAtIdle(const bool enable);

    bool getSnapshotAtIdle() const { return _snapshotAtIdle; }
    void setWriteFrames(const bool enable);

    bool getWriteFrames() const;

    void setGrabFrame(const bool enable);

    bool getGrabFrame() const;

    void grabFrame();

    /**
       Returns true if the grab frames bit was dirtied and set to true
       during last serialization and toggles the internal state to return
       false at next call.
     */
    bool resetFrameCounter();

    /**
       Stores the filename to be returned by getSnapshotName next time it
       is called.

       This methods is indented to be called only from the application node.

       @param fileName
       @param waitForCompletion If true, wait until the done signal is
       committed by the pipe instance and deserialized by this view.

       @bug When idle-AA is enabled and the snaphost considers it, frames
       are not being trigerred until the snapshot is done.
     */
    void snapshot(const std::string& fileName, const bool waitForCompletion);

    /**
       Called from Channel to determine whether it has to get a snapshot
       and the filename to use.

       @return empty string if no snapshot is to be performed. Otherwise the
               filename to use.
     */
    std::string getSnapshotName(const std::string& channelName) const;

    /**
       To be called from a destination channel when it has written a snapshot
       to a file.

       This method resets the snapshot name to the empty string if all the
       destination channels from a given pipe have completed the snapshot.
     */
    void snapshotDone(const eq::Channel& channel);

    void setFilePrefix(const std::string& prefix);

    std::string getFilePrefix() const;

    /**
       Image file format extension without dot to use for frame recording.
    */
    void setFileFormat(const std::string& extension);

    std::string getFileFormat() const;

    /** Overriden to provide mutual exclusion with set methods and
        to reset the snapshot attribute. */
    virtual co::uint128_t commit(const uint32_t incarnation = CO_COMMIT_NEXT);

    /*--- Public signals ---*/

    /** Emitted when the modelview matrix is updated in the master instance of
        this view.

        This signal is not intended to be used in slave intances. */
    DirtySignal modelviewDirty;

private:
    /*--- Private declarations ---*/

    class Proxy : public co::Serializable
    {
        friend class View;

    public:
        Proxy(View& view)
            : _view(view)
        {
        }

        enum DirtyBits
        {
            DIRTY_SCENE_ID = co::Serializable::DIRTY_CUSTOM << 0,
            DIRTY_MODELMATRIX = co::Serializable::DIRTY_CUSTOM << 1,
            DIRTY_CLEARCOLOR = co::Serializable::DIRTY_CUSTOM << 2,
            DIRTY_USE_ROI = co::Serializable::DIRTY_CUSTOM << 3,
            DIRTY_IDLE = co::Serializable::DIRTY_CUSTOM << 4,
            DIRTY_DECORATOR_OBJECT = co::Serializable::DIRTY_CUSTOM << 5,
            DIRTY_DECORATOR = co::Serializable::DIRTY_CUSTOM << 6,
            DIRTY_POINTER = co::Serializable::DIRTY_CUSTOM << 7,
            DIRTY_WRITE_FRAMES = co::Serializable::DIRTY_CUSTOM << 8,
            DIRTY_GRAB_FRAME = co::Serializable::DIRTY_CUSTOM << 9,
            DIRTY_SNAPSHOT = co::Serializable::DIRTY_CUSTOM << 10,
            DIRTY_FILE_PREFIX = co::Serializable::DIRTY_CUSTOM << 11,
            DIRTY_FILE_FORMAT = co::Serializable::DIRTY_CUSTOM << 12,
            DIRTY_SNAPSHOT_DONE = co::Serializable::DIRTY_CUSTOM << 13,
            DIRTY_ORTHO = co::Serializable::DIRTY_CUSTOM << 14,
            DIRTY_USE_DOF = co::Serializable::DIRTY_CUSTOM << 15,
            DIRTY_FOCAL_DISTANCE = co::Serializable::DIRTY_CUSTOM << 16,
            DIRTY_FOCAL_RANGE = co::Serializable::DIRTY_CUSTOM << 17,
            DIRTY_ALL_BITS =
                (DIRTY_SCENE_ID | DIRTY_MODELMATRIX | DIRTY_CLEARCOLOR |
                 DIRTY_USE_ROI | DIRTY_IDLE | DIRTY_POINTER | DIRTY_DECORATOR |
                 DIRTY_DECORATOR_OBJECT | DIRTY_WRITE_FRAMES |
                 DIRTY_GRAB_FRAME | DIRTY_SNAPSHOT | DIRTY_FILE_PREFIX |
                 DIRTY_FILE_FORMAT | DIRTY_SNAPSHOT_DONE | DIRTY_ORTHO |
                 DIRTY_USE_DOF | DIRTY_FOCAL_DISTANCE | DIRTY_FOCAL_RANGE)
        };

    protected:
        virtual void serialize(co::DataOStream& os, const uint64_t bits);
        virtual void deserialize(co::DataIStream& is, const uint64_t bits);
        virtual void notifyNewVersion() { sync(); }
    private:
        View& _view;
    };

    /*--- Private member attributes ---*/

    LB_TS_VAR(_thread);
    Proxy _proxy;
    mutable std::mutex _lock;

    /* Attributes that are only used in the application node */
    osg::ref_ptr<osgGA::EventQueue> _eventQueue;
    OSGCameraManipulatorPtr _manipulator;
    typedef std::vector<EventHandlerPtr> EventHandlers;
    EventHandlers _eventHandlers;
    bool _resetManipulator;
    PointerPtr _pointer;
    bool _requiresContinuousUpdate;

    bool _useOrtho;

    /* Distributed attributes */
    unsigned int _sceneID;
    osg::Matrixd _matrix;
    osg::Vec4 _clearColor;
    bool _useDOF;
    float _focalDistance;
    float _focalRange;
    bool _useROI;

    unsigned int _maxIdleSteps;
    bool _snapshotAtIdle;

    bool _writeFrames;
    bool _grabFrame;
    bool _resetFrameCounter;
    std::string _filePrefix;
    std::string _fileFormat;

    struct SnapshotHelper;
    SnapshotHelper* _snapshotHelper;

    // The base class should also be valid, but boost serialization crashes
    // if we try to serialize the base pointer instead of the derived.
    SceneDecoratorPtr _decorator;

    /* This variable is used to block updates from clients to the master
       decorator which can trigger the creation of a new scene decorator
       (thus disconnecting this view from its rtneuron::View counterpart).
       In principle this shouldn't be happening but it does, I need to
       learn more about how Views are handled by Equalizer to understand
       why. */
    bool _decoratorProtected;

    /*--- Private member functions ---*/

    virtual void notifyFrustumChanged();

    virtual void requestContinuousUpdate(bool needed = true)
    {
        LB_TS_THREAD(_thread);
        _requiresContinuousUpdate = needed;
    }

    virtual void requestWarpPointer(float /* x */, float /* y */) {}
    void _requestRedraw();

    void _onPointerDirty();

    void _snapshotDoneOnPipe();

    void _onDecoratorDirty();

    void _handleEvent(EventAdapter* event, const EventHandlers& handlers,
                      const OSGCameraManipulatorPtr& manipulator);

    void _updatePointer(PointerPtr pointer,
                        const OSGCameraManipulatorPtr& manipulator);

    void _replaceDecorator(const SceneDecoratorPtr& decorator);
};
}
}
}
#endif
