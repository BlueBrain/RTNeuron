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

#include "net.h"

#include <co/buffer.h>
#include <co/connection.h>
#include <co/connectionDescription.h>

#include <chrono>
#include <thread>

namespace bbp
{
namespace rtneuron
{
namespace core
{
co::Connections connectPeers(const ConnectionDescriptions& descriptions,
                             const size_t localIndex, const size_t connsPerPeer)
{
    auto listener = co::Connection::create(
        new co::ConnectionDescription(*descriptions[localIndex]));
    if (!listener->listen())
    {
        std::stringstream error;
        error << "Could not start server listening on "
              << descriptions[localIndex];
        throw std::runtime_error(error.str());
    }

    return connectPeers(descriptions, *listener, localIndex, connsPerPeer);
}

co::Connections connectPeers(const ConnectionDescriptions& descriptions,
                             co::Connection& listener, const size_t localIndex,
                             const size_t connsPerPeer)
{
    co::Connections connections;

    /* Establishing the connections the easiest possible way.
       This wouldn´t scale if the number of nodes is high, but doing it
       optimally is damn hard (I even think there's a connection between
       this problem and the prime numbers). */
    assert(listener.isListening());

    co::BufferPtr buffer(new co::Buffer);
    buffer->reserve(sizeof(uint32_t) * 2);

    for (size_t index = 0; index != descriptions.size(); ++index)
    {
        if (index == localIndex)
        {
            /* This is the local node, it's this time to be the server to
               accept the connections. */
            const size_t newConnections =
                (descriptions.size() - 1) * connsPerPeer - connections.size();

            /* A gap of connsPerPeer empty connections is left of the list
               corresponding to the connections at this node's index. */
            connections.resize(connections.size() + newConnections +
                               connsPerPeer);
            for (size_t i = 0; i != newConnections; ++i)
            {
                listener.acceptNB();
                co::ConnectionPtr peer = listener.acceptSync();
                assert(peer);
                /* Finding the position of this peer in the list. We could do
                   this by finding the peer hostname in the list of connection
                   descriptions, however that doesn't work in multprocessing
                   configurations because hostnames appear repeated. Instead,
                   we expect the connecting peer to send its node index and
                   connection number. */
                buffer->setSize(0);
                peer->recvNB(buffer, sizeof(uint32_t) * 2);
                peer->recvSync(buffer);
                const uint32_t* indices =
                    reinterpret_cast<uint32_t*>(buffer->getData());
                const size_t position = indices[0] * connsPerPeer + indices[1];
                if (position >= connections.size())
                    throw std::runtime_error("Bad peer node indices received");
                assert(descriptions[indices[0]]->hostname ==
                       peer->getDescription()->hostname);
                assert(!connections[position]);
                connections[position] = peer;
            }
        }
        else if (connections.size() != descriptions.size() * connsPerPeer)
        {
            const auto& connDesc = descriptions[index];
            for (size_t i = 0; i != connsPerPeer; ++i)
            {
                auto connection = co::Connection::create(
                    new co::ConnectionDescription(*connDesc));

                for (size_t retries = 0; !connection->connect() && retries < 5;
                     ++retries)
                    std::this_thread::sleep_for(std::chrono::seconds(1));

                /* Writing the connection indices so the server peer can
                   properly identify which is the connecting node even in
                   multiprocessing configurations. */
                std::vector<uint32_t> info{uint32_t(localIndex), uint32_t(i)};
                connection->send(info.data(), sizeof(uint32_t) * 2);

                if (!connection->isConnected())
                    throw std::runtime_error(
                        listener.getDescription()->hostname +
                        ": Could not connect peer: " + connDesc->hostname);
                connections.push_back(connection);
            }
        }
    }
    return connections;
}
}
}
}
