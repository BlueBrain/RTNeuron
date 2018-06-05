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

#ifndef RTNEURON_API_VIEW_IMPL_H
#define RTNEURON_API_VIEW_IMPL_H

#include "View.h"
#include "detail/Configurable.h"

#include "coreTypes.h"
#include "viewer/osgEq/types.h"

#include <osg/Vec3>

namespace osg
{
class Node;
class Camera;
class RenderInfo;
}

namespace bbp
{
namespace rtneuron
{
/*
  View::_Impl
*/
class View::_Impl : public detail::Configurable
{
    friend class View;

public:
    /*--- Public declarations ---*/

    class UpdateCallback;

    /*--- Public constructors/destructor ---*/

    /**
       \todo Is it possible to use any smart pointer for the view?
     */
    _Impl(osgEq::View* view, const AttributeMap& style);

    ~_Impl();

    /*--- Public Member functions ---*/

    void setScene(const ScenePtr& scene);

    void setCameraManipulator(const CameraManipulatorPtr& manipulator);

    CameraManipulatorPtr getCameraManipulator() const;

    void computeHomePosition();

    void transform(osg::Vec3& vec);

    void setPointer(const PointerPtr& pointer);

    PointerPtr getPointer();

    void record(bool enable);

    void grabFrame();

    void snapshot(const std::string& fileName, const bool waitForCompletion);

    osgEq::View* getEqView() { return _view; }
    const osgEq::View* getEqView() const { return _view; }
protected:
    /*--- Slots ---*/

    void onAttributeChangedImpl(const AttributeMap& attributes,
                                const std::string& name);

    void onAttributeChangingImpl(
        const AttributeMap& attributes, const std::string& name,
        const AttributeMap::AttributeProxy& parameters);

    void onSceneDirty(bool recomputeHome);

private:
    /*--- Private member attributes ---*/
    osgEq::View* const _view;

    CameraPtr _camera;
    ScenePtr _scene;

    bool _autoComputeHomePosition;

    bool _autoAdjustModelScale;
    float _modelScale;
    float _stereoCorrection;

    CameraManipulatorPtr _manipulator;

    core::ViewStylePtr _style;

    /*--- Private Member functions ---*/
    //! Try picking on the current scene.
    void _onPointerPick(const osg::Vec3& position, const osg::Vec3& direction);

    void _adjustPointer();

    /** @sa View::getAttributes stereo_correction */
    void _setStereoCorrection(double value);

    void _multiplyStereoCorrection(double factor);

    /** @sa View::getAttributes zero_parallax_distance */
    void _setZeroParallaxDistance(double distance);

    void _checkSceneAndViewCompatibility();
};
}
}
#endif
