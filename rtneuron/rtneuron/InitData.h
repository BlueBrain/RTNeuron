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

#ifndef RTNEURON_INITDATA_H
#define RTNEURON_INITDATA_H

#include "viewer/osgEq/InitData.h"

#include "types.h"

#include <lunchbox/uint128_t.h>

#include <map>

namespace bbp
{
namespace rtneuron
{
class InitData : public osgEq::InitData
{
public:
    /*--- Public declarations ---*/

    struct ProfilingFlags
    {
        ProfilingFlags()
            : enable(false)
            , gpuCullCPUTime(false)
            , gpuCullGPUTime(false)
            , skeletonCullCPUTime(false)
            , alphablending(false)
        {
        }

        bool enable; // This means that at least basic profiling must be done
        bool gpuCullCPUTime;
        bool gpuCullGPUTime;
        bool skeletonCullCPUTime;
        bool alphablending;
        std::string file;
    };

    /*--- Public attributes ---*/

    ProfilingFlags profilingFlags;

    typedef std::map<unsigned int, lunchbox::uint128_t> SceneDistributableUUIDs;
    SceneDistributableUUIDs sceneUUIDs;

    /*--- Public constructors/destructor ---*/

    InitData(const AttributeMap& attributes);

protected:
    /*--- Protected member functions ---*/

    virtual void _getInstanceDataImpl(co::DataOStream& out);

    virtual void _applyInstanceDataImpl(co::DataIStream& in);
};
}
}
#endif
