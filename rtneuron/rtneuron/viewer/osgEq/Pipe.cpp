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

#include "Pipe.h"

#include "Application.h"
#include "Client.h"
#include "Config.h"
#include "InitData.h"

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
/*
   Constructor
*/
Pipe::Pipe(eq::Node* parent)
    : eq::Pipe(parent)
{
}

/*
  Destructor
*/
Pipe::~Pipe()
{
}

/*
  Member functions
*/
bool Pipe::configInit(const eq::uint128_t& initDataID)
{
    if (!eq::Pipe::configInit(initDataID))
        return false;

    Config* config = static_cast<Config*>(getConfig());
    const InitData& initData = config->getInitData();
    if (!config->mapObject(&_frameData, initData.frameDataID))
        return false;

    return true;
}

bool Pipe::configExit()
{
    getConfig()->unmapObject(&_frameData);

    return eq::Pipe::configExit();
}

void Pipe::frameStart(const eq::uint128_t& frameID, const uint32_t frameNumber)
{
    _frameData.sync(frameID);
    eq::Pipe::frameStart(frameID, frameNumber);
}

eq::WindowSystem Pipe::selectWindowSystem() const
{
/* Use Qt windows for sharing the texture and the event handling when we
   have the GUI widget */
#ifdef EQUALIZER_USE_QT5WIDGETS
    const Config* config = static_cast<const Config*>(getConfig());
    const InitData& initData = config->getInitData();
    if (initData.shareContext && isWindowSystemAvailable("Qt"))
        return eq::WindowSystem("Qt");
#endif
    return eq::Pipe::selectWindowSystem();
}
}
}
}
