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
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"

#include "co/co.h"

#include <lunchbox/test.h>

// uint128_t used in archive causes this warning:
// negative integral constant converted to unsigned type
#pragma warning(disable : 4308)

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

template <typename T>
class Object : public co::Object
{
public:
    Object()
        : _value()
    {
    }
    explicit Object(T value)
        : _value(value)
    {
    }

    T getValue() const { return _value; }
protected:
    virtual void getInstanceData(co::DataOStream& os)
    {
        bbp::rtneuron::net::DataOStreamArchive archive(os);
        archive << _value;
    }

    virtual void applyInstanceData(co::DataIStream& is)
    {
        bbp::rtneuron::net::DataIStreamArchive archive(is);
        archive >> _value;
    }

private:
    T _value;
};

template <typename T>
void testObjectSerialization(co::LocalNodePtr server, co::LocalNodePtr client,
                             const T& value)
{
    Object<T> object(value);
    TEST(client->registerObject(&object));

    Object<T> remoteObject;
    TEST(server->mapObject(&remoteObject, object.getID()));

    TEST(object.getValue() == remoteObject.getValue());

    server->unmapObject(&remoteObject);
    client->deregisterObject(&object);
}

int main(int argc, char** argv)
{
    TEST(co::init(argc, argv));

    co::ConnectionDescriptionPtr connDesc = new co::ConnectionDescription;
    connDesc->type = co::CONNECTIONTYPE_TCPIP;
    connDesc->setHostname("localhost");

    co::LocalNodePtr server = new co::LocalNode;
    server->addConnectionDescription(connDesc);
    TEST(server->listen());

    co::NodePtr serverProxy = new co::Node;
    serverProxy->addConnectionDescription(connDesc);

    connDesc = new co::ConnectionDescription;
    connDesc->type = co::CONNECTIONTYPE_TCPIP;
    connDesc->setHostname("localhost");

    co::LocalNodePtr client = new co::LocalNode;
    client->addConnectionDescription(connDesc);
    TEST(client->listen());
    TEST(client->connect(serverProxy));

    testObjectSerialization(server, client, 42);
    testObjectSerialization(server, client, 5.f);
    testObjectSerialization(server, client, false);
    testObjectSerialization(server, client, std::string("blablub"));
    testObjectSerialization(server, client, co::uint128_t(12345, 54321));
    testObjectSerialization(server, client, std::vector<int>(9));

    TEST(client->disconnect(serverProxy));
    TEST(client->close());
    TEST(server->close());

    serverProxy->printHolders(std::cerr);
    TESTINFO(serverProxy->getRefCount() == 1, serverProxy->getRefCount());
    TESTINFO(client->getRefCount() == 1, client->getRefCount());
    TESTINFO(server->getRefCount() == 1, server->getRefCount());

    serverProxy = 0;
    client = 0;
    server = 0;

    TEST(co::exit());

    return EXIT_SUCCESS;
}
