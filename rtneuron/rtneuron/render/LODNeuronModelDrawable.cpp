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

#include "LODNeuronModelDrawable.h"

#include "CameraData.h"
#include "ViewStyleData.h"
#include "data/Neuron.h"
#include "scene/SphericalSomaModel.h"

#include <osgUtil/CullVisitor>

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
inline osgUtil::CullVisitor::value_type _distance(const osg::Vec3& coord,
                                                  const osg::Matrix& matrix)
{
    typedef osgUtil::CullVisitor::value_type value_type;
    return -((value_type)coord[0] * (value_type)matrix(0, 2) +
             (value_type)coord[1] * (value_type)matrix(1, 2) +
             (value_type)coord[2] * (value_type)matrix(2, 2) + matrix(3, 2));
}

void _boundProjection(osgUtil::CullVisitor* visitor,
                      osg::RenderInfo* renderInfo,
                      const osg::BoundingBox& bound, float& x, float& X,
                      float& y, float& Y)
{
    /** \todo Factor out this code since it's also used in
        osgEq::Scene to compute the ROI. */
    x = y = std::numeric_limits<float>::max();
    X = Y = -std::numeric_limits<float>::max();
    const osg::Camera* camera = renderInfo->getCurrentCamera();
    const osg::Viewport& vp = *camera->getViewport();
    double nearPlane, dummy;
    camera->getProjectionMatrixAsFrustum(dummy, dummy, dummy, dummy, nearPlane,
                                         dummy);
    const osg::Matrix mvp =
        *visitor->getModelViewMatrix() * *visitor->getProjectionMatrix();
    osg::Vec4 corners[8];
    for (unsigned int i = 0; i < 8; ++i)
    {
        const osg::Vec3& c = bound.corner(i);
        corners[i] = osg::Vec4(c.x(), c.y(), c.z(), 1) * mvp;
    }

    for (unsigned int i = 0; i < 8; ++i)
    {
        osg::Vec4 corner = corners[i];
        if (corner.w() > 0)
        {
            /* A point in front of the camera */
            corner /= corner.w();
            const osg::Vec2 screenCorner((corner.x() + 1) * 0.5 * vp.width(),
                                         (corner.y() + 1) * 0.5 * vp.height());
            x = std::min(x, screenCorner.x());
            y = std::min(y, screenCorner.y());
            X = std::max(X, screenCorner.x());
            Y = std::max(Y, screenCorner.y());
        }
        else
        {
            /* This point is behind the camera, clipping the bounding
               box */
            for (unsigned int j = 0; j < 3; ++j)
            {
                const osg::Vec4& neighbour = corners[i ^ (1 << j)];
                if (neighbour.w() > 0)
                {
                    /* Moving the corner to make w a small positive */
                    osg::Vec4 corner2 =
                        corner +
                        (neighbour - corner) *
                            (std::min(float(nearPlane), neighbour.w()) -
                             corner.w()) /
                            (neighbour.w() - corner.w());
                    corner2 /= corner2.w();
                    const osg::Vec2 screenCorner((corner2.x() + 1) * 0.5 *
                                                     vp.width(),
                                                 (corner2.y() + 1) * 0.5 *
                                                     vp.height());
                    x = std::min(x, screenCorner.x());
                    y = std::min(y, screenCorner.y());
                    X = std::max(X, screenCorner.x());
                    Y = std::max(Y, screenCorner.y());
                } /* else the side is completely clipped */
            }
        }
    }
}
}
/*
   Helper classes
*/
class LODNeuronModel::Drawable::CullCallback
    : public osg::Drawable::CullCallback
{
public:
    virtual bool cull(osg::NodeVisitor* nv, osg::Drawable* drawable,
                      osg::RenderInfo* renderInfo) const
    {
        osgUtil::CullVisitor* visitor = static_cast<osgUtil::CullVisitor*>(nv);
        unsigned long frameNumber = visitor->getFrameStamp()->getFrameNumber();

        Drawable* lodDrawable = dynamic_cast<Drawable*>(drawable);
        if (!lodDrawable)
        {
            /* Removing this useless callback from the drawable */
            drawable->setCullCallback(0);
            return false;
        }

        /* Discarding the drawable if the model object has been deleted. */
        if (!lodDrawable->_model)
            return true;

        /* Discarding this drawable and all its sub-drawables if the display
           mode doesn't include the whole neuron */
        const Neuron* neuron = lodDrawable->_model->getNeuron();
        if (!neuron)
            return true;

        const RepresentationMode mode =
            neuron->getRepresentationMode(frameNumber);
        switch (mode)
        {
        case RepresentationMode::WHOLE_NEURON:
        case RepresentationMode::NO_AXON:
            _cullLevelsOfDetail(lodDrawable, nv, renderInfo);
            break;
        case RepresentationMode::SOMA:
            /* The requested display mode for this neuron is "only soma".
               Making the SphericalSomaModel visible. */
            lodDrawable->_setSphericalSomaVisibility(visitor, true);
            break;
        case RepresentationMode::SEGMENT_SKELETON:
        case RepresentationMode::NO_DISPLAY:
            /* In these cases we need to make the spherical somas invisible */
            lodDrawable->_setSphericalSomaVisibility(visitor, false);
            break;
        default:;
        }
        return true;
    }

private:
    void _cullLevelsOfDetail(Drawable* drawable, osg::NodeVisitor* nodeVisitor,
                             osg::RenderInfo* renderInfo) const
    {
        osgUtil::CullVisitor* visitor =
            static_cast<osgUtil::CullVisitor*>(nodeVisitor);

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
        const osg::BoundingBox& bound = drawable->getBoundingBox();
#else
        const osg::BoundingBox& bound = drawable->getBound();
#endif
        float x, X, y, Y;
        _boundProjection(visitor, renderInfo, bound, x, X, y, Y);

        CameraData* data =
            CameraData::getOrCreateCameraData(visitor->getCurrentCamera());
        const float levelOfDetailBias =
            data->viewStyle ? data->viewStyle->getLevelOfDetailBias() : 1.f;
        const float pixelSize = (X - x) * (Y - y) * levelOfDetailBias;
        const float thresholdCorrection = 1 / 2000.f;
        const float threshold =
            std::min(1.f, (pixelSize / bound.radius()) * thresholdCorrection);

        /* Pushing the state of the parent drawable. */
        osg::StateSet* stateset = drawable->getStateSet();
        if (stateset)
            visitor->pushStateSet(stateset);

        /* Adding the drawables into the cull visitor directly from
           here if they are in the range of selection and returning true. */
        for (const auto& lod : drawable->_model->_models)
        {
            const bool inRange =
                (threshold > lod.min && threshold <= lod.max) ||
                (lod.min == 0 && threshold == 0);

            SphericalSomaModel* soma =
                dynamic_cast<SphericalSomaModel*>(lod.model.get());
            if (soma)
                soma->setVisibility(data->getCircuitSceneID(), inRange);
            else if (inRange)
            {
                osg::Drawable* lodDrawable = lod.model->getDrawable();
                osg::Drawable::CullCallback* callback =
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
                    dynamic_cast<osg::Drawable::CullCallback*>(
                        lodDrawable->getCullCallback());
#else
                    lodDrawable->getCullCallback();
#endif
                if (!callback ||
                    !callback->cull(nodeVisitor, lodDrawable, renderInfo))
                {
                    _addDrawable(visitor, lodDrawable);
                }
            }
        }

        if (stateset)
            visitor->popStateSet();
    }

    void _addDrawable(osgUtil::CullVisitor* cv, osg::Drawable* drawable) const
    {
/* This code is taken from osgUtil::CullVisitor::apply */
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
        const osg::BoundingBox& bb = drawable->getBoundingBox();
#else
        const osg::BoundingBox& bb = drawable->getBound();
#endif
        osg::RefMatrix& matrix = *cv->getModelViewMatrix();
        if (cv->getComputeNearFarMode() && bb.valid())
        {
            if (!cv->updateCalculatedNearFar(matrix, *drawable, false))
                return;
        }

        /* need to track how push/pops there are, so we can unravel the
           stack correctly. */
        unsigned int numPopStateSetRequired = 0;

        /* push the geoset's state on the geostate stack. */
        osg::StateSet* stateset = drawable->getStateSet();
        if (stateset)
        {
            ++numPopStateSetRequired;
            cv->pushStateSet(stateset);
        }

        osg::CullingSet& cs = cv->getCurrentCullingSet();
        if (!cs.getStateFrustumList().empty())
        {
            osg::CullingSet::StateFrustumList& sfl = cs.getStateFrustumList();
            for (osg::CullingSet::StateFrustumList::iterator i = sfl.begin();
                 i != sfl.end(); ++i)
            {
                if (i->second.contains(bb))
                {
                    ++numPopStateSetRequired;
                    cv->pushStateSet(i->first.get());
                }
            }
        }

        const float depth = bb.valid() ? _distance(bb.center(), matrix) : 0.0f;

        if (!osg::isNaN(depth))
            cv->addDrawableAndDepth(drawable, &matrix, depth);

        for (unsigned int i = 0; i < numPopStateSetRequired; ++i)
            cv->popStateSet();
    }
};
static osg::ref_ptr<LODNeuronModel::Drawable::CullCallback> s_cullCallback(
    new LODNeuronModel::Drawable::CullCallback());

class LODNeuronModel::Drawable::UpdateCallback
    : public osg::Drawable::UpdateCallback
{
public:
    virtual void update(osg::NodeVisitor* nv, osg::Drawable* drawable_)
    {
        Drawable* drawable = dynamic_cast<Drawable*>(drawable_);
        if (drawable && drawable->_model)
        {
            for (auto& lod : drawable->_model->_models)
            {
                osg::Drawable* lodDrawable = lod.model->getDrawable();
                if (!lodDrawable)
                    continue;

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
                osg::Drawable::UpdateCallback* callback =
                    dynamic_cast<osg::Drawable::UpdateCallback*>(
                        lodDrawable->getCullCallback());
                if (callback)
                    callback->update(nv, lodDrawable);

#else
                if (lodDrawable->getUpdateCallback())
                    lodDrawable->getUpdateCallback()->update(nv, lodDrawable);
#endif
            }
        }
    }
};
static osg::ref_ptr<LODNeuronModel::Drawable::UpdateCallback> s_updateCallback(
    new LODNeuronModel::Drawable::UpdateCallback());

/*
  Constructors
*/
LODNeuronModel::Drawable::Drawable()
{
    setUseDisplayList(false);
    setCullCallback(s_cullCallback);
    setUpdateCallback(s_updateCallback);
}

LODNeuronModel::Drawable::Drawable(const LODNeuronModel::Drawable& drawable,
                                   const osg::CopyOp& op)
    : osg::Drawable(drawable, op)
{
    setUseDisplayList(false);
    setCullCallback(s_cullCallback);
    setUpdateCallback(s_updateCallback);
}

/*
  Member functions
*/
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
osg::BoundingBox LODNeuronModel::Drawable::computeBoundingBox() const
#else
osg::BoundingBox LODNeuronModel::Drawable::computeBound() const
#endif
{
    return getInitialBound();
}

void LODNeuronModel::Drawable::releaseGLObjects(osg::State* state) const
{
    if (!_model)
        return;

    for (auto& lod : _model->_models)
    {
        osg::Drawable* drawable = lod.model->getDrawable();
        if (drawable)
            drawable->releaseGLObjects(state);
    }
    /* Calling the base implementation Just in case there's anything
       attached to this drawable */
    osg::Drawable::releaseGLObjects(state);
}

void LODNeuronModel::Drawable::_setSphericalSomaVisibility(
    osgUtil::CullVisitor* visitor, const bool visibility)
{
    for (auto& lod : _model->_models)
    {
        SphericalSomaModel* soma =
            dynamic_cast<SphericalSomaModel*>(lod.model.get());
        if (!soma)
            continue;
        osg::Referenced* data = visitor->getCurrentCamera()->getUserData();
        assert(dynamic_cast<CameraData*>(data));
        CameraData* cameraData = static_cast<CameraData*>(data);
        soma->setVisibility(cameraData->getCircuitSceneID(), visibility);
        return;
    }
}
}
}
}
