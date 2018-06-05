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

#include <rtneuron/RTNeuron.h>
#include <rtneuron/Scene.h>
#include <rtneuron/net/SceneEventBroker.h>

#include <BBP/Cell_Target.h>

#include <zeroeq/publisher.h>
#include <zeroeq/subscriber.h>

#include <lunchbox/sleep.h>

#define BOOST_TEST_MODULE
#include <boost/foreach.hpp>
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

void _makeTargetAndIds(Cell_Target& target, uint32_ts& ids,
                       const uint32_t first)
{
    target.clear();
    ids.clear();
    for (uint32_t i = first; i < 100; i += 7)
    {
        target.insert(i);
        ids.push_back(i);
    }
}
}

struct Fixture
{
    Fixture()
        : rtneuron(new RTNeuron(0, 0))
        , scene(rtneuron->createScene(AttributeMap()))
    {
    }

    void cellsSelected(const Cell_Target& target) { selected = target; }
    void cellSetBinaryOp(const Cell_Target& first_, const Cell_Target& second_,
                         const lexis::data::CellSetBinaryOpType operation_)
    {
        first = first_;
        second = second_;
        operation = operation_;
    }

    void onToggleRequest(const lexis::data::ConstToggleIDRequestPtr& event)
    {
        requested.clear();
        BOOST_FOREACH (const uint32_t gid, event->getIdsVector())
            requested.insert(gid);
    }

    RTNeuronPtr rtneuron;
    ScenePtr scene;

    Cell_Target selected;

    Cell_Target first;
    Cell_Target second;
    lexis::data::CellSetBinaryOpType operation;

    Cell_Target requested;
};

BOOST_FIXTURE_TEST_SUITE(camera_broker, Fixture)

BOOST_AUTO_TEST_CASE(constructors)
{
    net::SceneEventBroker broker1();
    net::SceneEventBroker broker2("session");
    net::SceneEventBroker broker3(zeroeq::URI("*:0"),
                                  zeroeq::URI("localhost:1234"));
}

BOOST_AUTO_TEST_CASE(receive_no_track)
{
    zeroeq::Publisher publisher(zeroeq::URI("*:0"));
    net::SceneEventBroker broker(zeroeq::URI("*:0"), publisher.getURI());

    broker.cellsSelected.connect(
        boost::bind(&Fixture::cellsSelected, this, _1));
    broker.cellSetBinaryOp.connect(
        boost::bind(&Fixture::cellSetBinaryOp, this, _1, _2, _3));

    lunchbox::sleep(STARTUP_DELAY);

    Cell_Target target1, target2;
    uint32_ts ids1, ids2;
    _makeTargetAndIds(target1, ids1, 1);
    _makeTargetAndIds(target2, ids2, 2);

    publisher.publish(lexis::data::SelectedIDs(ids1));
    lunchbox::sleep(WRITE_DELAY);
    BOOST_CHECK_EQUAL(selected, target1);

    publisher.publish(lexis::data::CellSetBinaryOp(
        ids1, ids2, lexis::data::CellSetBinaryOpType::Projections));
    lunchbox::sleep(WRITE_DELAY);
    BOOST_CHECK_EQUAL(first, target1);
    BOOST_CHECK_EQUAL(second, target2);
    BOOST_CHECK_EQUAL(operation, lexis::data::CellSetBinaryOpType::Projections);
}

BOOST_AUTO_TEST_CASE(publish)
{
    net::SceneEventBroker broker(zeroeq::URI("*:0"),
                                 zeroeq::URI("localhost:1234"));
    zeroeq::Subscriber subscriber(broker.getURI());
    broker.trackScene(scene);
    subscriber.subscribe(
        lexis::data::ToggleIDRequest::ZEROBUF_TYPE_IDENTIFIER(),
        [&](const void* data, const size_t size) {
            onToggleRequest(lexis::data::ToggleIDRequest::create(data, size));
        });
    lunchbox::sleep(STARTUP_DELAY);

    Cell_Target target;
    target.insert(10);
    scene->cellSelected(10, 0, 0);
    subscriber.receive(WRITE_DELAY);
    BOOST_CHECK_EQUAL(requested, target);

    target.insert(20);
    scene->cellSetSelected(target);
    subscriber.receive(WRITE_DELAY);
    BOOST_CHECK_EQUAL(requested, target);
}

BOOST_AUTO_TEST_CASE(inter_brokers_communication)
{
    // This test could be performed with only two brokers if there was
    // a way to change the application instance ID used by one of the brokers

    net::SceneEventBroker requester(zeroeq::URI("*:0"),
                                    zeroeq::URI("localhost:1234"));
    net::SceneEventBroker tracker(zeroeq::URI("*:0"), requester.getURI());
    net::SceneEventBroker receiver(zeroeq::URI("*:0"), tracker.getURI());

    lunchbox::sleep(STARTUP_DELAY);

    tracker.cellsSelected.connect(
        boost::bind(&Fixture::cellsSelected, this, _1));

    Cell_Target target;
    uint32_ts ids;
    _makeTargetAndIds(target, ids, 1);

    requester.sendToggleRequest(target);
    lunchbox::sleep(WRITE_DELAY);
    BOOST_CHECK_EQUAL(selected, Cell_Target());

    tracker.setTrackState(true);
    BOOST_CHECK(tracker.getTrackState());

    for (int i = 0; i != 2; ++i)
    {
        requester.sendToggleRequest(target);
        lunchbox::sleep(WRITE_DELAY);
        BOOST_CHECK_EQUAL(selected, target);

        requester.sendToggleRequest(target);
        lunchbox::sleep(WRITE_DELAY);
        BOOST_CHECK_EQUAL(selected, Cell_Target());

        tracker.sendToggleRequest(target);
        lunchbox::sleep(WRITE_DELAY * 2);
        BOOST_CHECK_EQUAL(selected, target);

        tracker.sendToggleRequest(target);
        lunchbox::sleep(WRITE_DELAY * 2);
        BOOST_CHECK_EQUAL(selected, Cell_Target());

        tracker.cellsSelected.disconnect(
            boost::bind(&Fixture::cellsSelected, this, _1));
        receiver.cellsSelected.connect(
            boost::bind(&Fixture::cellsSelected, this, _1));
    }
}

BOOST_AUTO_TEST_SUITE_END()
