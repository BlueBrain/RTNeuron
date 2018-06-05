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

#include <osg/io_utils>

#include "Channel.h"
#include "Scene.h"

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
void null_deleter(const void*)
{
}

// Scene ----------------------------------------------------------------------

osg::Vec4 Scene::getRegionOfInterest(const Channel& channel) const
{
    /** \bug This code doesn't work with orthographic projections */
    const osg::Camera* camera = channel.getCamera();
    const osg::Viewport& vp = *camera->getViewport();
    float x, X, y, Y;
    x = y = std::numeric_limits<float>::max();
    X = Y = -std::numeric_limits<float>::max();

    osg::Vec4 eye = osg::Vec4(0, 0, 0, 1) * camera->getInverseViewMatrix();
    osg::BoundingBox bounding = getSceneBoundingBox(channel.getRange());
    if (bounding.contains(osg::Vec3(eye.x(), eye.y(), eye.z())))
        return osg::Vec4(0, 0, vp.width(), vp.height());

    bool valid = false;
    double nearPlane, _;
    camera->getProjectionMatrixAsFrustum(_, _, _, _, nearPlane, _);
    osg::Matrix mvp = camera->getViewMatrix() * camera->getProjectionMatrix();
    osg::Vec4 corners[8];
    for (int i = 0; i < 8; ++i)
    {
        osg::Vec3 c = bounding.corner(i);
        corners[i] = osg::Vec4(c.x(), c.y(), c.z(), 1) * mvp;
    }

    for (int i = 0; i < 8; ++i)
    {
        osg::Vec4 corner = corners[i];
        if (corner.w() > 0)
        {
            /* A point in front of the camera */
            valid = true;
            corner /= corner.w();
            osg::Vec2 screenCorner((corner.x() + 1) * 0.5 * vp.width(),
                                   (corner.y() + 1) * 0.5 * vp.height());
            x = std::min(x, screenCorner.x());
            y = std::min(y, screenCorner.y());
            X = std::max(X, screenCorner.x());
            Y = std::max(Y, screenCorner.y());
        }
        else
        {
            /* This point is behind the camera, this means that the bounding
               box needs to be clipped in this corner. The loop will shift
               the corner towards each of the visible neighbours until the
               corner is at the near plane. The transformed corner will
               be used to update the ROI. */
            for (int j = 0; j < 3; ++j)
            {
                osg::Vec4 neighbour = corners[i ^ (1 << j)];
                if (neighbour.w() > 0)
                {
                    valid = true;
                    /* Moving the corner to make w a small positive */
                    osg::Vec4 corner2 =
                        corner +
                        (neighbour - corner) *
                            (std::min(float(nearPlane), neighbour.w()) -
                             corner.w()) /
                            (neighbour.w() - corner.w());
                    corner2 /= corner2.w();
                    osg::Vec2 screenCorner((corner2.x() + 1) * 0.5 * vp.width(),
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
    if (!valid)
        return osg::Vec4(0, 0, 0, 0);

#if !defined NDEBUG or defined DEBUG_ROI
#ifndef OSG_GL3_AVAILABLE
    glBegin(GL_LINES);
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int neighbourIndex = i ^ (1 << j);
            osg::Vec4 corner = corners[i];
            osg::Vec4 neighbour = corners[neighbourIndex];

            if (neighbour.w() <= 0 || (corner.w() > 0 && i > neighbourIndex))
                /* Nothing to draw in this case */
                continue;

            if (corner.w() <= 0)
            {
                assert(neighbour.w() > 0);
                /* Moving the corner to make w a small positive */
                corner +=
                    (neighbour - corner) *
                    (std::min(float(nearPlane), neighbour.w()) - corner.w()) /
                    (neighbour.w() - corner.w());
            }
            corner /= corner.w();
            neighbour /= neighbour.w();
            osg::Vec2 a((corner.x() + 1) * 0.5 * vp.width(),
                        (corner.y() + 1) * 0.5 * vp.height());
            osg::Vec2 b((neighbour.x() + 1) * 0.5 * vp.width(),
                        (neighbour.y() + 1) * 0.5 * vp.height());
            glVertex2f(a.x(), a.y());
            glVertex2f(b.x(), b.y());
        }
    }
    glEnd();
#endif
#endif

    return osg::Vec4(std::max(x, float(vp.x())), std::max(y, float(vp.y())),
                     std::min(X, float(vp.x() + vp.width())),
                     std::min(Y, float(vp.y() + vp.height())));
}
}
}
}
