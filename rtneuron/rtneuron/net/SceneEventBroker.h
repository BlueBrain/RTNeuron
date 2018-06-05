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

#ifndef RTNEURON_API_NET_SCENEEVENTBROKER_H
#define RTNEURON_API_NET_SCENEEVENTBROKER_H

#include "../types.h"

#include <lexis/lexis.h>
#include <zeroeq/uri.h>

#include <boost/signals2/signal.hpp>

namespace bbp
{
namespace rtneuron
{
namespace net
{
/**
   Subscriber/publisher for scene related events

   This class creates a thread that listens for the following scene events
   - zeroeq.hbp.SelectedIDs (zeroeq::hbp::EVENT_SELECTEDIDS)
   - zeroeq.hbp.ToggleIDRequest (zeroeq::hbp::EVENT_TOGGLEREQUEST)
   - zeroeq.hbp.CellSetBinaryOp (zeroeq::hbp::EVENT_CELLSETBINARYOP)
   and triggers a signal whenever any of them is received.

   @version 2.9
*/
class SceneEventBroker
{
public:
    /*--- Public declarations ---*/

    typedef void CellSetSelectedSignalSignature(const brain::GIDSet&);
    typedef boost::signals2::signal<CellSetSelectedSignalSignature>
        CellSetSelectedSignal;

    typedef void CellSetBinaryOpSignalSignature(
        const brain::GIDSet&, const brain::GIDSet&,
        lexis::data::CellSetBinaryOpType);
    typedef boost::signals2::signal<CellSetBinaryOpSignalSignature>
        CellSetBinaryOpSignal;

    /*--- Public constructors/destructor ---*/

    /**
       Create a broker that will connect using the default ZeroEQ session.
       @version 2.10
    */
    SceneEventBroker();

    /**
       Create a broker using a ZeroEQ session name.
       @version 2.10
    */
    SceneEventBroker(const std::string& session);

    /**
       Create a broker using explicit pub/sub URIs.
       @version 2.10
    */
    SceneEventBroker(const zeroeq::URI& publisher,
                     const zeroeq::URI& subscriber);

    ~SceneEventBroker();
    SceneEventBroker(const SceneEventBroker&) = delete;
    SceneEventBroker& operator=(const SceneEventBroker&) = delete;

    /*--- Public member functions ---*/

    /**
       Track cellSelected and cellSetSelected events and send ZeroEQ toggle
       events when these events occur.
       @version 2.9
    */
    void trackScene(const ScenePtr& scene);

    /**
       Set whether this class retains the state of selected cells.

       If that's the case this class will also listen and react to toggle id
       requests.
       @version 2.9
    */
    void setTrackState(const bool track);

    /** @version 2.9 */
    bool getTrackState() const;

    /** @version 2.9 */
    void sendToggleRequest(const brain::GIDSet& cells);

    /*--- Public signals ---*/

    /* Selection */

    /**
       Signal emitted when a EVENT_SELECTEDIDS is received.
       @version 2.9
    */
    CellSetSelectedSignal cellsSelected;

    /**
       Signal emitted when a EVENT_CELLSETBINARYOP is received.
       @version 2.9
    */
    CellSetBinaryOpSignal cellSetBinaryOp;

    /** @internal For testing purposes only */
    zeroeq::URI getURI() const;

    /*--- Private member variables ---*/
private:
    class _Impl;
    _Impl* _impl;
};
}
}
}
#endif
