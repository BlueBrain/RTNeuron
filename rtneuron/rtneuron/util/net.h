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

#ifndef RTNEURON_UTIL_NET_H
#define RTNEURON_UTIL_NET_H

#include <co/types.h>

#include <vector>

namespace bbp
{
namespace rtneuron
{
namespace core
{
typedef std::vector<co::ConstConnectionDescriptionPtr> ConnectionDescriptions;

/** Given a set of node connection decriptors this function establishes
    connectionsPerPeer connections to each of the given nodes.

    @param descriptions The list of connection descriptions. This parameter must
           be the same in each participating node, that is, same descriptions
           and in the same order. The localhost must also be in the list at the
           position indicated by localIndex. The description there will be
           used to open a temporary server to receive the incoming connections
           from peers.
    @param localIndex Position of the localhost connection description.
    @param connsPerPeer Number of connections to stablish with each peer.
    @return All the established connections connections. The connections in the
            range [index * connectionsPerPeer, (index + 1) * connectionsPerPeer]
            are left empty as they map to the localhost.
*/
co::Connections connectPeers(const ConnectionDescriptions& descriptions,
                             size_t index, size_t connsPerPeer = 1);

/** The same as above but with a prexisting server for input connection.

    @param descriptions See above. In this case the list of descriptions must
           still the local host, although it will be ignored.
    @param index See above.
    @param listener A listening connection in an interface an port reachable by
           the other peers.
    @param connsPerPeer See above.
    @return See above.
*/
co::Connections connectPeers(const ConnectionDescriptions& descriptions,
                             co::Connection& listener, size_t index,
                             size_t connsPerPeer = 1);
}
}
}
#endif
