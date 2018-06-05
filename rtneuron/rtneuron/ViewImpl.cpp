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

#include "ViewImpl.h"

#include "Camera.h"
#include "CameraManipulator.h"
#include "Scene.h"
#include "render/ViewStyle.h"
#include "ui/PickEventHandler.h"
#include "ui/Pointer.h"
#include "util/attributeMapHelpers.h"
#include "util/vec_to_vec.h"
#include "viewer/osgEq/Scene.h"
#include "viewer/osgEq/View.h"

#include <eq/observer.h>
#include <eq/view.h>

#include <iostream>

namespace bbp
{
namespace rtneuron
{
namespace
{
void _computeHomePosition(const Scene& scene, osg::Vec3& eye, osg::Vec3& center)
{
    /* This is a heuristic that ignores the neuron morphologies
       because for computing the exact bounding box of the neurons
       we also need the actual display mode, which can be changed by
       the user any time. Despite it's a coarse approximation, this
       solution keeps the result consistent and easier to obtain. */

    osg::BoundingSphere sphere = scene.getSomasBoundingSphere();
    if (!sphere.valid())
        sphere.expandBy(scene.getSynapsesBoundingSphere());
    sphere.expandBy(scene.getExtraModelsBoundingSphere());
    const float fov = 45.0f; // TODO BBPRTN-268
    center = sphere.center();
    const float distance =
        (sphere.radius() + 200) / tan(fov / 180 * osg::PI * 0.5);
    eye = center + osg::Vec3(0, 0, distance);
}
}

// View::_Impl ----------------------------------------------------------------

/*
  Constructor
*/

View::_Impl::_Impl(osgEq::View* view, const AttributeMap& attributes)
    : _view(view)
    , _camera(new Camera(view))
    , _autoComputeHomePosition(true)
    , _autoAdjustModelScale(true)
    , _modelScale(1)
    , _stereoCorrection(1)
    , _style(new core::ViewStyle(attributes))
{
    PickEventHandler* pickEventHandler = new PickEventHandler;
    EventHandlerPtr pickEventHandlerPtr(pickEventHandler);
    pickEventHandler->pick.connect(
        boost::bind(&_Impl::_onPointerPick, this, _1, _2));
    view->addEventHandler(pickEventHandlerPtr);

    AttributeMap& internal = getAttributes();
    blockAttributeMapSignals();
    /* Setting default values for attributes so this default values can be
       checked from the public interface. */
    internal.set("background", 0.0, 0.0, 0.0, 0.0);
    internal.set("output_file_prefix", "");
    internal.set("output_file_format", "png");
    internal.set("snapshot_at_idle", true);
    internal.set("use_roi", true);
    /* We must use an integer because that's how the parameter is extracted
       (Python always passes integers). */
    internal.set("idle_AA_steps", (int)_view->getMaxIdleSteps());
    internal.set("auto_compute_home_position", true);
    internal.set("model_scale", 1);
    /* Stereo attributes */
    internal.set("auto_adjust_model_scale", true);
    internal.set("stereo_correction", 1);
    internal.set("stereo", view->getMode() == eq::View::MODE_STEREO);
    /* This is unknown at this point because the observer is not ready. */
    internal.set("zero_parallax_distance", 1.0);

    AttributeMapPtr depthField(new AttributeMap());
    internal.set("depth_of_field", depthField);
    depthField->set("enabled", _view->useDOF());
    depthField->set("focal_distance", _view->getFocalDistance());
    depthField->set("focal_range", _view->getFocalRange());

    unblockAttributeMapSignals();
    /* Now we want the attribute changed signals to be delivered.
       View style attributes will try to be applied again, but it doesn't
       matter because they have the same values. Filtering them out doesn't
       seem necessary and would be verbose. */
    internal.merge(attributes);

    /* To merge the attributes from the view style (this will include all
       the attributes taking default values) we want to block signals
       again as the attributes changes don't need to be propagated. */
    blockAttributeMapSignals();
    internal.merge(_style->getAttributes());
    unblockAttributeMapSignals();

    view->setDecorator(_style);
}

View::_Impl::~_Impl()
{
    if (_scene)
        _scene->dirty.disconnect(boost::bind(&_Impl::onSceneDirty, this, _1));
    const PointerPtr pointer = _view->getPointer();
    if (pointer)
    {
        pointer->pick.disconnect(
            boost::bind(&_Impl::_onPointerPick, this, _1, _2));
        pointer->transformCoord.disconnect(
            boost::bind(&_Impl::transform, this, _1));
        pointer->stereoCorrection.disconnect(
            boost::bind(&_Impl::_multiplyStereoCorrection, this, _1));
    }
    _camera->invalidate();
}

/*
  Member functions
*/

void View::_Impl::setScene(const ScenePtr& scene)
{
    if (_scene)
        _scene->dirty.disconnect(boost::bind(&_Impl::onSceneDirty, this, _1));

    if (scene)
    {
        _scene = scene;
        _view->setSceneID(scene->getSceneProxy()->getID());

        _scene->dirty.connect(boost::bind(&_Impl::onSceneDirty, this, _1));
    }
    else
        _scene.reset();

    _checkSceneAndViewCompatibility();

    onSceneDirty(true);
}

void View::_Impl::setCameraManipulator(const CameraManipulatorPtr& manipulator)
{
    _manipulator = manipulator;
    if (manipulator)
        /* osgEq::View::setCameraManipulator dirties the model matrix */
        _view->setCameraManipulator(manipulator->osgManipulator());
    else
        _view->setCameraManipulator(0);
    _view->dirty();
}

CameraManipulatorPtr View::_Impl::getCameraManipulator() const
{
    return _manipulator;
}

void View::_Impl::computeHomePosition()
{
    osg::Vec3 center, eye;
    _computeHomePosition(*_scene, eye, center);
    _view->setHomePosition(eye, center, osg::Vec3(0, 0, 1));
    _view->dirty();
}

void View::_Impl::transform(osg::Vec3& vec)
{
    /** \bug This code is multiplying vec by the identify matrix as
        the camera manipulator matrix and the model matrix are reciprocal.
        See osgEq/View.cpp code. */
    vec =
        vec * osg::Matrix::inverse(_view->getCameraManipulator()->getMatrix() *
                                   _view->getModelMatrix());
}

void View::_Impl::setPointer(const PointerPtr& pointer)
{
    const PointerPtr current = _view->getPointer();
    if (current)
    {
        current->pick.disconnect(
            boost::bind(&_Impl::_onPointerPick, this, _1, _2));
        current->transformCoord.disconnect(
            boost::bind(&_Impl::transform, this, _1));
        current->stereoCorrection.disconnect(
            boost::bind(&_Impl::_multiplyStereoCorrection, this, _1));
    }
    pointer->pick.connect(boost::bind(&_Impl::_onPointerPick, this, _1, _2));
    pointer->transformCoord.connect(boost::bind(&_Impl::transform, this, _1));
    pointer->stereoCorrection.connect(
        boost::bind(&_Impl::_multiplyStereoCorrection, this, _1));

    _view->setPointer(pointer);

    if (_autoAdjustModelScale)
        _adjustPointer();
}

PointerPtr View::_Impl::getPointer()
{
    return _view->getPointer();
}

void View::_Impl::record(const bool enable)
{
    _view->setWriteFrames(enable);
}

void View::_Impl::grabFrame()
{
    _view->grabFrame();
}

void View::_Impl::snapshot(const std::string& fileName,
                           const bool waitForCompletion)
{
    _view->snapshot(fileName, waitForCompletion);
}

void View::_Impl::onAttributeChangingImpl(
    const AttributeMap& attributes, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    if (name == "background")
    {
        osg::Vec4 color;
        core::AttributeMapHelpers::getColor(parameters, color);
    }
    else if (name == "idle_AA_steps")
    {
        bool bad = false;
        try
        {
            const unsigned int steps = parameters;
            bad = steps > 256;
        }
        catch (...)
        {
            const int steps = parameters;
            bad = steps < 0 || steps > 256;
        }
        if (bad)
            throw std::runtime_error("Invalid number of accumulation steps");
    }
    else if (name == "use_roi")
    {
        (void)(bool) parameters;
    }
    else if (name == "output_file_prefix")
    {
        /* (std::string)parameters doesn't compile */
        std::string dummy LB_UNUSED = parameters;
    }
    else if (name == "output_file_format")
    {
        /* (std::string)parameters doesn't compile */
        std::string dummy LB_UNUSED = parameters;
    }
    else if (name == "snapshot_at_idle")
    {
        (void)(bool) parameters;
    }
    else if (name == "zero_parallax_distance")
    {
        (void)(double) parameters;
    }
    else if (name == "stereo_correction")
    {
        (void)(double) parameters;
    }
    else if (name == "stereo")
    {
        (void)(bool) parameters;
    }
    else if (name == "model_scale")
    {
        (void)(double) parameters;
    }
    else if (name == "auto_adjust_model_scale")
    {
        (void)(bool) parameters;
    }
    else if (name == "auto_compute_home_position")
    {
        (void)(bool) parameters;
    }
    else if (name == "depth_of_field.enabled")
    {
        (void)(bool) parameters;
    }
    else if (name.substr(0, 15) == "depth_of_field.")
    {
        (void)(double) parameters;
    }
    else
        _style->validateAttributeChange(attributes, name, parameters);
}

void View::_Impl::onAttributeChangedImpl(const AttributeMap& attributes,
                                         const std::string& name)
{
    if (name == "background")
    {
        using namespace core::AttributeMapHelpers;
        osg::Vec4 color;
        getRequiredColor(attributes, "background", color);
        _view->setClearColor(color);
        _view->dirty();
    }
    else if (name == "use_roi")
    {
        _view->useROI(attributes("use_roi"));
    }
    else if (name == "output_file_prefix")
    {
        _view->setFilePrefix(attributes("output_file_prefix"));
    }
    else if (name == "output_file_format")
    {
        _view->setFileFormat(attributes("output_file_format"));
    }
    else if (name == "idle_AA_steps")
    {
        try
        {
            _view->setMaxIdleSteps(attributes("idle_AA_steps"));
        }
        catch (...)
        {
            _view->setMaxIdleSteps((int)attributes("idle_AA_steps"));
        }
    }
    else if (name == "snapshot_at_idle")
    {
        _view->setSnapshotAtIdle(attributes("snapshot_at_idle"));
    }
    else if (name == "zero_parallax_distance")
    {
        _setZeroParallaxDistance(attributes("zero_parallax_distance"));
    }
    else if (name == "stereo_correction")
    {
        _setStereoCorrection(attributes("stereo_correction"));
    }
    else if (name == "stereo")
    {
        _view->changeMode(attributes("stereo") ? eq::View::MODE_STEREO
                                               : eq::View::MODE_MONO);
        _view->dirty();
        _view->requestRedraw();
    }
    else if (name == "model_scale")
    {
        _modelScale = (double)attributes("model_scale");
        _view->setModelUnit(_modelScale * _stereoCorrection);
        _view->dirty();
        _view->requestRedraw();
    }
    else if (name == "auto_adjust_model_scale")
    {
        _autoAdjustModelScale = attributes("auto_adjust_model_scale");
    }
    else if (name == "auto_compute_home_position")
    {
        _autoComputeHomePosition = attributes("auto_compute_home_position");
    }
    else if (name == "depth_of_field.enabled")
    {
        /* Amend attribute errors before anything else. */
        _checkSceneAndViewCompatibility();

        _view->useDOF(attributes(name));
    }
    else if (name == "depth_of_field.focal_distance")
    {
        _view->setFocalDistance((double)attributes(name));
    }
    else if (name == "depth_of_field.focal_range")
    {
        _view->setFocalRange((double)attributes(name));
    }
    else
    {
        /* For colormap related attributes we only propagate the change to
           the style if the color maps doesn't exist yet there. Otherwise,
           we will be replacing the ColorMap with itself. Since this code
           is triggerred from the ColorMap::dirty slot, the attribute copy
           seems to confuse the rewiring of dirty signals and the AttributeMap
           from SceneStyle ends up disconnected from the dirty signal. (I think
           this has to do once more with the disconnection "optimization" in
           boost::signals). */
        bool copyIt = true;
        try
        {
            if (name.substr(0, 10) == "colormaps.")
            {
                ColorMapPtr current = _style->getAttributes()(name);
                ColorMapPtr incoming = attributes(name);
                copyIt = current.get() != incoming.get();
            }
        }
        catch (...)
        {
        }

        if (copyIt)
            _style->getAttributes().copy(attributes, name);
        _view->dirty();
    }
}

void View::_Impl::onSceneDirty(bool recomputeHome)
{
    if (_scene && recomputeHome)
    {
        if (_autoComputeHomePosition || _autoAdjustModelScale)
        {
            osg::Vec3 center, eye;
            _computeHomePosition(*_scene, eye, center);

            if (_autoComputeHomePosition)
                _view->setHomePosition(eye, center, osg::Vec3(0, 0, 1));

            if (_autoAdjustModelScale)
            {
                _modelScale = eye[2] - center[2];
                blockAttributeMapSignals();
                getAttributes().set("model_scale", _modelScale);
                unblockAttributeMapSignals();
                _view->setModelUnit(_modelScale * _stereoCorrection);
                _adjustPointer();
            }
        }
    }

    _checkSceneAndViewCompatibility();

    _view->dirty();
    _view->requestRedraw(); // This shouldn't be needed
}

void View::_Impl::_onPointerPick(const osg::Vec3& position,
                                 const osg::Vec3& direction)
{
    if (_scene)
        _scene->pick(core::vec_to_vec(position), core::vec_to_vec(direction));
}

void View::_Impl::_adjustPointer()
{
    PointerPtr pointer = getPointer();
    if (pointer)
    {
        const float factor = 20.f;
        pointer->setRadius(pointer->getRadius() * _stereoCorrection *
                           _modelScale / factor);
        pointer->setDistance(pointer->getDistance() * _stereoCorrection *
                             _modelScale / factor);
    }
}

void View::_Impl::_setStereoCorrection(double value)
{
    _stereoCorrection = value;
    if (getPointer())
        getPointer()->adjust(value, false);
    /* This should trigger rendering, but it doesn't */
    _view->setModelUnit(_stereoCorrection * _modelScale);
    _view->requestRedraw();
}

void View::_Impl::_multiplyStereoCorrection(double factor)
{
    _stereoCorrection *= factor;
    if (getPointer())
        getPointer()->adjust(factor, true);
    /* This should trigger rendering, but it doesn't */
    _view->setModelUnit(_stereoCorrection * _modelScale);
    _view->requestRedraw();
}

void View::_Impl::_setZeroParallaxDistance(double distance)
{
    eq::Observer* observer = _view->getObserver();
    if (observer == 0)
    {
        std::cerr << "Warning: Untracked view, "
                     "can't change zero parallax distance"
                  << std::endl;
        return;
    }
    /** \todo Fixme for HMDs */
    observer->setFocusMode(eq::fabric::FOCUSMODE_RELATIVE_TO_ORIGIN);
    observer->setFocusDistance(distance);
    _view->requestRedraw();
}

void View::_Impl::_checkSceneAndViewCompatibility()
{
    if (!_scene)
        return;

    /* Checking that alpha blending and depth of field are not both enabled.
       If that's the case, disabling depth of field. */
    AttributeMapPtr dof;
    getAttributes().get("depth_of_field", dof);
    AttributeMapPtr alphaBlending =
        _scene->getAttributes()("alpha_blending", AttributeMapPtr());
    if ((*dof)("enabled", false) && alphaBlending && !alphaBlending->empty())
    {
        LBWARN << "Depth of field and transparency cannot be combined. "
               << "Disabling depth of field" << std::endl;
        dof->set("enabled", false);
        _view->useDOF(false);
    }
}
}
}
