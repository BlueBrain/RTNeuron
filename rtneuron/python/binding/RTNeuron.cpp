/* Copyright (c) 2006-2018, Ecole Polytechnique Federale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politecnica de Madrid (UPM)
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

#include "boost_signal_connect_wrapper.h"
#include "docstrings.h"
#include "gil.h"

#include "rtneuron/AttributeMap.h"
#include "rtneuron/RTNeuron.h"
#include "rtneuron/Scene.h"
#include "rtneuron/SimulationPlayer.h"
#include "rtneuron/View.h"
#include "rtneuron/ui/CameraPath.h"

#ifdef override
#undef override
#endif
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/signals2/signal.hpp>

class QOpenGLContext;

using namespace boost::python;
using namespace bbp::rtneuron;

static AttributeMap defaultAttributeMap;

// export_RTNeuron -------------------------------------------------------------

/* We need to build a fake argc and argv from a Python list before invoking
   the constructor RTNeuron. This class is used for that purpose. */
class Args
{
public:
    Args(list args)
    {
        /* Extracting const char * objects from the list is safe as
           long as the RTNeuron class constructor copies the strings (and
           it does). */
        size_t argc = len(args);
        if (argc == 0)
        {
            object sys = import("sys");
            args = extract<list>(sys.attr("argv"));
            /* Only taking the app name in this case, otherwise
               eq::Server::chooseConfig fails:
               https://github.com/Eyescale/Equalizer/issues/270 */
            _args.push_back(extract<std::string>(args[0]));
            _argv.push_back(_args.back().c_str());
        }
        else
        {
            /* Reserving the required size in _args is important to avoid
               pointer  invalidations due to relocations. */
            _args.reserve(argc);
            _argv.reserve(argc);
            for (stl_input_iterator<std::string> i(args), end; i != end; ++i)
            {
                _args.push_back(*i);
                _argv.push_back(&_args.back()[0]);
            }
        }
    }

protected:
    std::vector<const char*> _argv;
    std::vector<std::string> _args;
};

typedef std::vector<ViewPtr> ViewPtrs;

class RTNeuronWrapper : public Args, public RTNeuron
{
public:
    RTNeuronWrapper(list args = list(),
                    const AttributeMap& attributes = AttributeMap())
        : Args(args)
        , RTNeuron(_argv.size(), (char**)&_argv[0], attributes)
    {
        /* This is needed to avoid a crash if the configuration is exited
           by pressing ESC and signals are connected */
        exited.connect(
            boost::bind(&RTNeuronWrapper::_disconnectViewSlots, this));
    }

    ~RTNeuronWrapper()
    {
        /* Disconnecting all signals first, the GIL is needed during the
           first call because it may trigger the destruction of python
           objects. */
        disconnectAllSlots();

        ReleaseGIL release;
        /* This one will take the GIL only when needed avoiding lock inversion
           hazards. */
        _disconnectViewSlots();

        /* Ensuring that the destructor of the wrapped class runs without
           the GIL acquired. Running finalize twice is well defined. */
        finalize();
    }

private:
    void _disconnectViewSlots()
    {
        /* getAllViews requires a lock, do not lock GIL before getting the
           views. This is needed to avoid a locking inversion hazard when
           this function is invoked from the exited signal. */
        const ViewPtrs& views = getAllViews();
        EnsureGIL lock;
        for (const ViewPtr& view : views)
            view->disconnectAllSlots();
    }
};

list RTNeuron_views(RTNeuronWrapper* rtneuron)
{
    ViewPtrs views;
    {
        ReleaseGIL release;
        views = rtneuron->getViews();
    }
    list result;
    for (auto view : views)
        result.append(view);
    return result;
}

list RTNeuron_allViews(RTNeuronWrapper* rtneuron)
{
    ViewPtrs views;
    {
        ReleaseGIL release;
        views = rtneuron->getAllViews();
    }
    list result;
    for (auto view : views)
        result.append(view);
    return result;
}

void RTNeuron_init(RTNeuronWrapper* rtneuron, const std::string& configFileName)
{
    ReleaseGIL release;
    rtneuron->init(configFileName);
}

void RTNeuron_exit(RTNeuronWrapper* rtneuron)
{
    /* Releasing the GIL before calling getAllViews */
    ReleaseGIL release;
    const std::vector<ViewPtr>& views = rtneuron->getAllViews();
    {
        /* Reacquiring the GIL before disconnecting slots */
        EnsureGIL lock;
        for (const ViewPtr& view : views)
            view->disconnectAllSlots();
    }
    rtneuron->exit();
}

void RTNeuron_exitConfig(RTNeuronWrapper* rtneuron)
{
    RTNeuron_exit(rtneuron);
}

void RTNeuron_frame(RTNeuronWrapper* rtneuron)
{
    ReleaseGIL release;
    rtneuron->frame();
}

void RTNeuron_waitFrame(RTNeuronWrapper* rtneuron)
{
    ReleaseGIL release;
    rtneuron->waitFrame();
}

void RTNeuron_waitFrames(RTNeuronWrapper* rtneuron, unsigned int frames)
{
    ReleaseGIL release;
    rtneuron->waitFrames(frames);
}

void RTNeuron_wait(RTNeuronWrapper* rtneuron)
{
    ReleaseGIL release;
    rtneuron->wait();
}

void RTNeuron_waitRecord(RTNeuronWrapper* rtneuron)
{
    ReleaseGIL release;
    rtneuron->waitRecord();
}

void RTNeuron_setShareContext(RTNeuronWrapper* rtneuron, object pyObject)
{
    /* Poor man solution for exchanging QOpenGLContext between C++, Python,
       Boost.Python, SIP and PyQt5:
       http://www.qtcentre.org/archive/index.php/t-29773.html */
    if (pyObject == object())
        rtneuron->setShareContext(0);
    else
    {
        const object sip = import("sip");
        const long int ptr =
            extract<long int>(sip.attr("unwrapinstance")(pyObject));
        QOpenGLContext* context = reinterpret_cast<QOpenGLContext*>(ptr);
        rtneuron->setShareContext(context);
    }
}

AttributeMap& RTNeuron_getAttributes(RTNeuronWrapper* rtneuron)
{
    return rtneuron->getAttributes();
}

SimulationPlayerPtr RTNeuron_player(RTNeuronWrapper* rtneuron)
{
    return rtneuron->getPlayer();
}

object RTNeuron_getActiveViewEventProcessor(RTNeuronWrapper* rtneuron)
{
    /* Poor man solution for exchanging QObject between C++, Python,
       Boost.Python, SIP and PyQt5:
       http://www.qtcentre.org/archive/index.php/t-29773.html */
    const object sip = import("sip");
    const long int ptr =
        reinterpret_cast<long int>(rtneuron->getActiveViewEventProcessor());
    const object pyQt5 = import("PyQt5.QtCore");
    object QObjectType = pyQt5.attr("QObject");
    return sip.attr("wrapinstance")(ptr, QObjectType);
}

void export_RTNeuron()
// clang-format off
{
class_<RecordingParams>
("RecordingParams", DOXY_CLASS(bbp::rtneuron::RecordingParams), init<>())
    .def_readwrite("simulationStart", &RecordingParams::simulationStart,
        DOXY_VAR(bbp::rtneuron::RecordingParams::simulationStart))
    .def_readwrite("simulationEnd", &RecordingParams::simulationEnd,
        DOXY_VAR(bbp::rtneuron::RecordingParams::simulationEnd))
    .def_readwrite("simulationDelta", &RecordingParams::simulationDelta,
        DOXY_VAR(bbp::rtneuron::RecordingParams::simulationDelta))
    .def_readwrite("cameraPathDelta", &RecordingParams::cameraPathDelta,
        DOXY_VAR(bbp::rtneuron::RecordingParams::cameraPathDelta))
    .def_readwrite("cameraPath", &RecordingParams::cameraPath,
        DOXY_VAR(bbp::rtneuron::RecordingParams::cameraPath))
    .def_readwrite("filePrefix", &RecordingParams::filePrefix,
        DOXY_VAR(bbp::rtneuron::RecordingParams::filePrefix))
    .def_readwrite("fileFormat", &RecordingParams::fileFormat,
        DOXY_VAR(bbp::rtneuron::RecordingParams::fileFormat))
    .def_readwrite("stopAtCameraPathEnd", &RecordingParams::stopAtCameraPathEnd,
        DOXY_VAR(bbp::rtneuron::RecordingParams::stopAtCameraPathEnd))
    .def_readwrite("frameCount", &RecordingParams::frameCount,
        DOXY_VAR(bbp::rtneuron::RecordingParams::frameCount))
;

class_<RTNeuronWrapper, std::shared_ptr<RTNeuronWrapper>,
       boost::noncopyable>
rtneuronWrapper("RTNeuron", DOXY_CLASS(bbp::rtneuron::RTNeuron),
 init<optional<list, const AttributeMap &> >(
     (arg("argv") = list(),
      arg("attributes") = boost::ref(defaultAttributeMap)),
     DOXY_FN(bbp::rtneuron::RTNeuron::RTNeuron(list, const AttributeMap&))));

scope rtneuronScope = rtneuronWrapper;

/* Nested classes */
class_<RTNeuron::FrameIssuedSignal, boost::noncopyable>
("__FrameIssuedSignal__", no_init)
    .def("connect", signal_connector<
             RTNeuron::FrameIssuedSignalSignature>::connect)
    .def("disconnect", signal_connector<
             RTNeuron::FrameIssuedSignalSignature>::disconnect)
;

class_<RTNeuron::TextureUpdatedSignal, boost::noncopyable>
("__TextureUpdatedSignal__", no_init)
    .def("connect", signal_connector<
             RTNeuron::TextureUpdatedSignalSignature>::connect)
    .def("disconnect", signal_connector<
             RTNeuron::TextureUpdatedSignalSignature>::disconnect)
;

class_<RTNeuron::IdleSignal, boost::noncopyable>
("__IdleSignal__", no_init)
    .def("connect", signal_connector<
             RTNeuron::IdleSignalSignature>::connect)
    .def("disconnect", signal_connector<
             RTNeuron::IdleSignalSignature>::disconnect)
;

rtneuronWrapper
    .def("init", RTNeuron_init, args("configFile") = "",
         DOXY_FN(bbp::rtneuron::RTNeuron::init))
    .def("exit", RTNeuron_exit, DOXY_FN(bbp::rtneuron::RTNeuron::exit))
    .def("createConfig", &RTNeuron::createConfig, args("configFile") = "",
         DOXY_FN(bbp::rtneuron::RTNeuron::createConfig))
    .def("exitConfig", RTNeuron_exitConfig,
         DOXY_FN(bbp::rtneuron::RTNeuron::exitConfig))
    .def("useLayout", &RTNeuron::useLayout,
         DOXY_FN(bbp::rtneuron::RTNeuron::useLayout))
    .def("createScene", &RTNeuron::createScene,
         args("attributes") = boost::ref(defaultAttributeMap),
         return_value_policy<return_by_value>(),
         DOXY_FN(bbp::rtneuron::RTNeuron::createScene))
    .def("record", &RTNeuron::record,
         DOXY_FN(bbp::rtneuron::RTNeuron::record))
    .def("pause", &RTNeuron::pause,
         DOXY_FN(bbp::rtneuron::RTNeuron::pause))
    .def("resume", &RTNeuron::resume,
         DOXY_FN(bbp::rtneuron::RTNeuron::resume))
    .def("frame", &RTNeuron_frame,
         DOXY_FN(bbp::rtneuron::RTNeuron::frame))
    .def("waitFrame", RTNeuron_waitFrame,
         DOXY_FN(bbp::rtneuron::RTNeuron::waitFrame))
    .def("waitFrames", RTNeuron_waitFrames,
         DOXY_FN(bbp::rtneuron::RTNeuron::waitFrames))
    .def("wait", RTNeuron_wait,
         DOXY_FN(bbp::rtneuron::RTNeuron::wait))
    .def("waitRecord", RTNeuron_waitRecord,
         DOXY_FN(bbp::rtneuron::RTNeuron::waitRecord))
    .def("setShareContext", &RTNeuron_setShareContext,
                       DOXY_FN(bbp::rtneuron::RTNeuron::setShareContext))
    .def("getActiveViewEventProcessor", RTNeuron_getActiveViewEventProcessor,
         DOXY_FN(bbp::rtneuron::RTNeuron::getActiveViewEventProcessor))
    .add_property("attributes", make_function(RTNeuron_getAttributes,
                                              return_internal_reference<>()),
                  DOXY_FN(bbp::rtneuron::RTNeuron::getAttributes))
    .add_property("views", RTNeuron_views,
                  DOXY_FN(bbp::rtneuron::RTNeuron::getViews))
    .add_property("allViews", RTNeuron_allViews,
                  DOXY_FN(bbp::rtneuron::RTNeuron::getAllViews))
    .add_property("player", RTNeuron_player,
                  DOXY_FN(bbp::rtneuron::RTNeuron::getPlayer))
    .def_readonly("frameIssued", &RTNeuron::frameIssued,
                  DOXY_VAR(bbp::rtneuron::RTNeuron::frameIssued))
    .def_readonly("textureUpdated", &RTNeuron::textureUpdated,
                  DOXY_VAR(bbp::rtneuron::RTNeuron::textureUpdated))
    .def_readonly("eventProcessorUpdated",
                  &bbp::rtneuron::RTNeuron::eventProcessorUpdated,
                  DOXY_VAR(bbp::rtneuron::RTNeuron::eventProcessorUpdated))
    .def_readonly("idle", &RTNeuron::idle,
                  DOXY_VAR(bbp::rtneuron::RTNeuron::idle))
    .def_readonly("exited", &RTNeuron::exited,
                  DOXY_VAR(bbp::rtneuron::RTNeuron::exited))
    .add_static_property("versionString", &RTNeuron::getVersionString,
                         DOXY_FN(bbp::rtneuron::RTNeuron::getVersionString))
;
}
