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

#include "InitData.h"
#include "ConfigEvent.h"

#include <co/dataIStream.h>
#include <co/dataOStream.h>

#include <lunchbox/bitOperation.h>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
// InitData --------------------------------------------------------------------

/*
  Constructor
*/
InitData::InitData()
    : isClient(false)
    , shareContext(0)
{
}

/*
  Member functions
*/

void InitData::getInstanceData(co::DataOStream& out)
{
    out << frameDataID << shareContext;

    _getInstanceDataImpl(out);
}

void InitData::applyInstanceData(co::DataIStream& in)
{
    in >> frameDataID >> shareContext;

    _applyInstanceDataImpl(in);
}
}
}
}
