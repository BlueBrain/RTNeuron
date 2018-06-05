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

#ifndef RTNEURON_MODELVIEWINVERSEUPDATE_H
#define RTNEURON_MODELVIEWINVERSEUPDATE_H

#include <osg/NodeCallback>
#include <osg/Uniform>
#include <osgUtil/CullVisitor>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*! \brief Hold a osg_ModelViewMatrixInverse GLSL uniform to update it
  during the cull pass.

  This class is used by the GL3 implementation of some shaders. It is
  directly attached to the node objects created by LODNeuronMode and
  DetailedNeuronModel when they add their geodes into the scene.
*/
class ModelViewInverseUpdate : public osg::NodeCallback
{
    /* Construtors */
public:
    ModelViewInverseUpdate(osg::Uniform* uniform = 0)
        : _uniform(uniform)
    {
        if (uniform == 0)
            _uniform = new osg::Uniform(osg::Uniform::FLOAT_MAT4,
                                        "osg_ModelViewMatrixInverse");
    }

    /* Member functions */
public:
    osg::Uniform* getUniform() { return _uniform.get(); }
    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
    {
        osgUtil::CullVisitor* cv = static_cast<osgUtil::CullVisitor*>(nv);
        _uniform->set(osg::Matrix::inverse(*cv->getModelViewMatrix()));
        traverse(node, nv);
    }

    /* Member attributes */
protected:
    osg::ref_ptr<osg::Uniform> _uniform;
};
}
}
}
#endif
