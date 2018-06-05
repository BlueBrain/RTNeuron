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

#ifndef RTNEURON_API_TYPES_H
#define RTNEURON_API_TYPES_H

#include "enums.h"

#include <brain/types.h>

#include <boost/shared_ptr.hpp>
#include <boost/signals2/signal.hpp>

#include <osg/Vec3>
#include <set>
#include <string>

namespace brain
{
class Circuit;
class CompartmentReport;
class SpikeReportReader;
class Synapses;
class Synapse;
using GIDSet = std::set<uint32_t>;
}

namespace bbp
{
namespace rtneuron
{
typedef void RayPickSignalSignature(const osg::Vec3&, const osg::Vec3&);
typedef boost::signals2::signal<RayPickSignalSignature> RayPickSignal;

class AttributeMap;
using AttributeMapPtr = std::shared_ptr<AttributeMap>;

class Camera;
using CameraPtr = std::shared_ptr<Camera>;

class CameraPath;
using CameraPathPtr = std::shared_ptr<CameraPath>;

class CameraManipulator;
using CameraManipulatorPtr = boost::shared_ptr<CameraManipulator>;

class CameraPathManipulator;
using CameraPathManipulatorPtr = boost::shared_ptr<CameraPathManipulator>;

using CircuitPtr = boost::shared_ptr<brain::Circuit>;

class ColorMap;
using ColorMapPtr = std::shared_ptr<ColorMap>;

using CompartmentReportPtr = boost::shared_ptr<brain::CompartmentReport>;
using SpikeReportReaderPtr = boost::shared_ptr<brain::SpikeReportReader>;

class EventHandler;
using EventHandlerPtr = std::shared_ptr<EventHandler>;

using GIDSet = brain::GIDSet;

class InitData;
using InitDataPtr = std::shared_ptr<InitData>;

class RTNeuron;
using RTNeuronPtr = std::shared_ptr<RTNeuron>;

class Pointer;
using PointerPtr = std::shared_ptr<Pointer>;

class Scene;
using ScenePtr = boost::shared_ptr<Scene>;
using SceneWeakPtr = boost::weak_ptr<Scene>;

class SimulationPlayer;
using SimulationPlayerPtr = std::shared_ptr<SimulationPlayer>;

class View;
using ViewPtr = std::shared_ptr<View>;

class Testing; /* Used to access private members from unit tests */

using vmml::Vector2f;
using vmml::Vector3f;
using vmml::Vector4f;
using vmml::Matrix4f;
using Orientation = Vector4f;
using brion::floats;
using brion::uint16_ts;
using brion::uint32_ts;
using brion::Vector3fs;

/**
   A convenience function to provide conversion to/from strings for certain
   enums.

   Template specializations are defined for NeuronModel and RepresentationMode.
*/
template <typename To, typename From>
To lexical_cast(const From&);

/**
   A convenience overload of lexical_cast to support conversion from
   const char *
*/
template <typename To>
To lexical_cast(const char* value)
{
    return lexical_cast<To, std::string>(value);
}

namespace core
{
class SpikeReport;
using SpikeReportPtr = std::shared_ptr<SpikeReport>;
}
}
}

/* OSG types appearing in the public headers that can be forward declared. */
namespace osg
{
class Vec3f;
using Vec3 = Vec3f;

template <typename T>
class BoundingSphereImpl;
using BoundingSphere = BoundingSphereImpl<Vec3f>;
}

#endif
