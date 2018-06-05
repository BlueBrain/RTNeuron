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

#ifndef RTNEURON_TEXT_H
#define RTNEURON_TEXT_H

#include <osgText/Text>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   This is an extension of osgText::Text to support correct rendering
   under Equalizer 2D tiled decomposition.
*/
class Text : public osgText::Text
{
public:
    Text()
        : osgText::Text()
    {
    }
    Text(const Text& text,
         const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY)
        : osgText::Text(text, copyop)
    {
    }

    virtual osg::Object* cloneType() const { return new Text(); }
    virtual osg::Object* clone(const osg::CopyOp& copyop) const
    {
        return new Text(*this, copyop);
    }
    virtual bool isSameKindAs(const osg::Object* obj) const
    {
        return dynamic_cast<const Text*>(obj) != 0;
    }
    virtual const char* className() const { return "Text"; }
    virtual const char* libraryName() const { return "rtneuron"; }
protected:
    virtual void computePositions(unsigned int contextID) const;
};
}
}
}
#endif
