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

#ifndef RTNEURON_OSGEQ_PIPE_H
#define RTNEURON_OSGEQ_PIPE_H

#include "FrameData.h"

#include <eq/pipe.h>

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
//! Inherits from eq::Pipe without extending or overriding.
class Pipe : public eq::Pipe
{
public:
    /*--- Public constructors/destructor */

    Pipe(eq::Node* parent);

    virtual ~Pipe();

    /*--- Public member functions ---*/

    /**
       Pipe-local read-only instance of the frame data.
    */
    const FrameData& getFrameData() const { return _frameData; }
private:
    /*--- Private member functions, called by Equalizer ---*/

    bool configInit(const eq::uint128_t& initDataID) final;

    bool configExit() final;

    void frameStart(const eq::uint128_t& frameID,
                    const uint32_t frameNumber) final;

    /**
       Select Qt as window system if GUI is requested
     */
    eq::WindowSystem selectWindowSystem() const final;

    /*--- Private member attributes ---*/

    FrameData _frameData;
};
}
}
}
#endif
