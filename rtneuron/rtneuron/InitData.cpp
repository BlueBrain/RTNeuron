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
#include "AttributeMap.h"

#include <co/co.h>

#include <boost/algorithm/string.hpp>

namespace bbp
{
namespace rtneuron
{
InitData::InitData(const AttributeMap& attributes)
{
    const std::string dummy = attributes("tracker", "");
    trackerDeviceName = dummy;

    const std::string flags = attributes("profiling_flags", "");
    if (!flags.empty())
    {
        std::vector<std::string> tokens;
        boost::split(tokens, flags, boost::is_any_of(","));
        for (std::vector<std::string>::const_iterator i = tokens.begin();
             i != tokens.end(); ++i)
        {
            /* Some of these profiling modes only make sense under certain
               circumstances but we don't have enough information to decide
               whether they make sense or not from here. */
            if (*i == "cull-gpu")
            {
                profilingFlags.gpuCullGPUTime = true;
                profilingFlags.gpuCullCPUTime = true;
                profilingFlags.skeletonCullCPUTime = true;
            }
            if (*i == "cull")
            {
                profilingFlags.gpuCullCPUTime = true;
                profilingFlags.skeletonCullCPUTime = true;
            }
            if (i->substr(0, 5) == "file=")
                profilingFlags.file = i->substr(5);
        }
    }
}

void InitData::_getInstanceDataImpl(co::DataOStream& out)
{
    const ProfilingFlags& f = profilingFlags;
    out << f.enable;
    if (f.enable)
        out << f.gpuCullCPUTime << f.gpuCullGPUTime << f.skeletonCullCPUTime
            << f.alphablending << f.file;

    out << sceneUUIDs.size();
    for (SceneDistributableUUIDs::const_iterator i = sceneUUIDs.begin();
         i != sceneUUIDs.end(); ++i)
        out << i->first << i->second;
}

void InitData::_applyInstanceDataImpl(co::DataIStream& in)
{
    ProfilingFlags& f = profilingFlags;
    in >> f.enable;
    if (f.enable)
        in >> f.gpuCullCPUTime >> f.gpuCullGPUTime >> f.skeletonCullCPUTime >>
            f.alphablending >> f.file;

    size_t size;
    in >> size;
    for (size_t i = 0; i != size; ++i)
    {
        unsigned int id;
        lunchbox::UUID uuid;
        in >> id >> uuid;
        sceneUUIDs[id] = uuid;
    }
}
}
}
