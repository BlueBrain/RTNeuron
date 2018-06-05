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

#include "Testing.h"

#include <rtneuron/RTNeuron.h>
#include <rtneuron/View.h>
#include <rtneuron/net/CameraBroker.h>

#include "rtneuron/viewer/osgEq/View.h"

#include <lexis/lexis.h>
#include <zeroeq/publisher.h>
#include <zeroeq/subscriber.h>

#include <eq/config.h>
#include <eq/init.h>
#include <eq/layout.h>
#include <eq/nodeFactory.h>
#include <eq/server.h>

#include <lunchbox/sleep.h>

#define BOOST_TEST_MODULE
#include <boost/test/unit_test.hpp>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <sys/types.h>
#include <unistd.h>
#endif

using namespace bbp;
using namespace bbp::rtneuron;

namespace
{
const uint32_t STARTUP_DELAY = 50;
const uint32_t WRITE_DELAY = 20;
}

struct EqInit : public eq::NodeFactory
{
    EqInit()
    {
        using boost::unit_test::framework::master_test_suite;
        eq::init(master_test_suite().argc, master_test_suite().argv, this);
    }
    ~EqInit() { eq::exit(); }
};

struct Fixture : EqInit
{
    Fixture()
        : config(eq::ServerPtr(new eq::Server()))
        , layout(&config)
        , view(&layout)
        , camera(Testing::createCamera(&view))
        , position(1, 2, 3)
        , orientation(0, 1, 0, M_PI * 0.5)
    {
        const vmml::Matrix4d matrix(
            vmml::Quaterniond(orientation.w() * M_PI / 180,
                              orientation.get_sub_vector<3, 0>()),
            position);
        viewMatrix = matrix.inverse();
    }

    eq::Config config;
    eq::Layout layout;
    osgEq::View view;
    CameraPtr camera;
    vmml::Matrix4d viewMatrix;
    Vector3d position;
    Orientation orientation;
};

BOOST_FIXTURE_TEST_SUITE(camera_broker, Fixture)

BOOST_AUTO_TEST_CASE(constructors)
{
    net::CameraBroker broker1(camera);
    net::CameraBroker broker2("session", camera);
    net::CameraBroker broker3(zeroeq::URI("*:0"), zeroeq::URI("localhost:1234"),
                              camera);
}

BOOST_AUTO_TEST_CASE(receive)
{
    zeroeq::Publisher publisher(zeroeq::URI("*:0"));
    net::CameraBroker broker(zeroeq::URI("*:0"), publisher.getURI(), camera);
    lunchbox::sleep(STARTUP_DELAY);

    std::vector<double> values(viewMatrix.data(), viewMatrix.data() + 16);
    // lexis::render::LookOut is in meters
    values[12] /= 1000000.f;
    values[13] /= 1000000.f;
    values[14] /= 1000000.f;
    Vector3f finalPosition;
    Orientation finalOrientation;

    for (size_t retries = 0; retries < 10; ++retries)
    {
        publisher.publish(lexis::render::LookOut(values));
        lunchbox::sleep(WRITE_DELAY);

        camera->getView(finalPosition, finalOrientation);
        if ((finalPosition - position).length() < 1e-6 &&
            (finalOrientation - orientation).length() < 1e-6)
            break;
    }

    BOOST_CHECK_MESSAGE((finalPosition - position).length() < 1e-6,
                        finalPosition);
    BOOST_CHECK_MESSAGE((finalOrientation - orientation).length() < 1e-6,
                        finalOrientation);
}

struct Subscriber
{
    Subscriber(const zeroeq::URI& uri)
        : received(false)
        , subscriber(uri)
    {
        subscriber.subscribe(lexis::render::LookOut::ZEROBUF_TYPE_IDENTIFIER(),
                             [&](const void* data, const size_t size) {
                                 _onUpdate(
                                     lexis::render::LookOut::create(data,
                                                                    size));
                             });
    }

    void _onUpdate(lexis::render::ConstLookOutPtr event)
    {
        matrix = event->getMatrixVector();
        received = true;
    }

    bool received;
    std::vector<double> matrix;
    zeroeq::Subscriber subscriber;
};

BOOST_AUTO_TEST_CASE(publish)
{
    net::CameraBroker broker(zeroeq::URI("*:0"), zeroeq::URI("localhost:1234"),
                             camera);
    Subscriber subscriber(broker.getURI());
    lunchbox::sleep(STARTUP_DELAY);

    for (size_t retries = 0; !subscriber.received && retries < 10; ++retries)
    {
        camera->setView(position, orientation);
        subscriber.subscriber.receive(WRITE_DELAY);
    }
    BOOST_REQUIRE(subscriber.received);

    std::vector<float> original(viewMatrix.data(), viewMatrix.data() + 16);
    // lexis::render::LookOut is in meters
    original[12] /= 1000000.f;
    original[13] /= 1000000.f;
    original[14] /= 1000000.f;

    for (size_t i = 0; i != 16; ++i)
        BOOST_CHECK_CLOSE(subscriber.matrix[i], original[i], 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
