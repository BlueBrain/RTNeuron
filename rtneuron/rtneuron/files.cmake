## Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
##                           Blue Brain Project and
##                          Universidad Politécnica de Madrid (UPM)
##                          Juan Hernando <juan.hernando@epfl.ch>
##
## This file is part of RTNeuron <https://github.com/BlueBrain/RTNeuron>
##
## This library is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License version 3.0 as published
## by the Free Software Foundation.
##
## This library is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along
## with this library; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

configure_file(config/paths.in.cpp
               ${PROJECT_BINARY_DIR}/config/paths.cpp)

file(GLOB_RECURSE RTNEURON_CORE_HEADERS
  config/[1-9a-zA-z_]*.h
  cuda/[1-9a-zA-z_]*.h
  data/[1-9a-zA-z_]*.h
  net/[1-9a-zA-z_]*.h
  render/[1-9a-zA-z_]*.h
  scene/[1-9a-zA-z_]*.h
  util/[1-9a-zA-z_]*.h
  viewer/[1-9a-zA-z_]*.h
  [1-9a-zA-z_]*.h)

set(RTNEURON_CORE_SOURCES
  ${PROJECT_BINARY_DIR}/config/paths.cpp
  config/Globals.cpp
  data/CircuitCache.cpp
  data/Neuron.cpp
  data/Neurons.cpp
  data/NeuronMesh.cpp
  data/SimulationDataMapper.cpp
  data/SpikeReport.cpp
  data/loaders.cpp
  net/DataIStreamArchive.cpp
  net/DataOStreamArchive.cpp
  render/CameraData.cpp
  render/ColorMap.cpp
  render/ColorMapAtlas.cpp
  render/DepthOfField.cpp
  render/DrawElementsPortions.cpp
  render/LODNeuronModelDrawable.cpp
  render/NeuronColoring.cpp
  render/Noise.cpp
  render/RenderBinManager.cpp
  render/SceneStyle.cpp
  render/Skeleton.cpp
  render/Text.cpp
  render/ViewStyle.cpp
  render/timing.cpp
  scene/CircuitScene.cpp
  scene/CircuitSceneAttributes.cpp
  scene/CollectSkeletonsVisitor.cpp
  scene/DetailedNeuronModel.cpp
  scene/LODNeuronModel.cpp
  scene/GeometryObject.cpp
  scene/ModelObject.cpp
  scene/NeuronObject.cpp
  scene/NeuronModel.cpp
  scene/SceneObject.cpp
  scene/SceneOperations.cpp
  scene/SimulationRenderBuffer.cpp
  scene/SphereSet.cpp
  scene/SphericalSomaModel.cpp
  scene/SubScene.cpp
  scene/SynapseObject.cpp
  scene/UpdateOperation.cpp
  scene/models/Cache.cpp
  scene/models/ConstructionData.cpp
  scene/models/CylinderBasedModel.cpp
  scene/models/MeshModel.cpp
  scene/models/NeuronSkeleton.cpp
  scene/models/TubeletBasedModel.cpp
  scene/models/SkeletonModel.cpp
  scene/models/utils.cpp
  sceneops/NeuronClipping.cpp
  sceneops/NeuronClippingImpl.cpp
  util/MurmurHash3.cpp
  util/Spline.cpp
  util/attributeMapHelpers.cpp
  util/cameraToID.cpp
  util/math.cpp
  util/net.cpp
  util/shapes.cpp
  util/splines/accel.cpp
  util/splines/bsearch.cpp
  util/splines/cspline.cpp
  util/splines/error.cpp
  util/splines/fdiv.cpp
  util/splines/infnan.cpp
  util/splines/interp.cpp
  util/splines/spline.cpp
  util/splines/stream.cpp
  util/splines/tridiag.cpp
  util/splines/view.cpp
  util/transformString.cpp
  util/triangleClassifiers.cpp
  viewer/SnapshotWindow.cpp
  viewer/StereoAnimationPath.cpp
  viewer/StereoAnimationPathManipulator.cpp
  viewer/osgEq/AuxiliaryWindow.cpp
  viewer/osgEq/Channel.cpp
  viewer/osgEq/Client.cpp
  viewer/osgEq/Compositor.cpp
  viewer/osgEq/Config.cpp
  viewer/osgEq/EmbeddedWindow.cpp
  viewer/osgEq/EventAdapter.cpp
  viewer/osgEq/FrameData.cpp
  viewer/osgEq/InitData.cpp
  viewer/osgEq/Node.cpp
  viewer/osgEq/NodeViewer.cpp
  viewer/osgEq/Pipe.cpp
  viewer/osgEq/Scene.cpp
  viewer/osgEq/SceneDecorator.cpp
  viewer/osgEq/View.cpp
  viewer/osgEq/Window.cpp
  AttributeMap.cpp
  Camera.cpp
  CameraImpl.cpp
  ColorMap.cpp
  InitData.cpp
  RTNeuron.cpp
  Scene.cpp
  SceneImpl.cpp
  View.cpp
  ViewImpl.cpp
  types.cpp
  ui/CameraPathManipulator.cpp
  ui/PickEventHandler.cpp
  ui/Pointer.cpp
  ui/TrackballManipulator.cpp
)

if(RTNEURON_WITH_ZEROEQ)
  list(APPEND RTNEURON_CORE_SOURCES
    net/CameraBroker.cpp
    net/SceneEventBroker.cpp
    net/RestInterface.cpp)
endif()

if(OSG_GL3_AVAILABLE AND RTNEURON_USE_CUDA AND CUDA_FOUND)
  list(APPEND RTNEURON_CORE_SOURCES
    viewer/osgEq/MultiFragmentCompositor.cpp
    viewer/osgEq/MultiFragmentFunctors.cpp)
endif()

if(VRPN_FOUND)
  list(APPEND RTNEURON_CORE_SOURCES
    ui/VRPNManipulator.cpp
    viewer/osgEq/Tracker.cpp
    viewer/GyrationMouse.cpp
    viewer/Intersense.cpp
    viewer/SpaceMouse.cpp
    viewer/VRPNMultiDeviceBase.cpp)
  if (WIIUSE_FOUND)
    list(APPEND RTNEURON_CORE_SOURCES
      ui/WiimotePointer.cpp
      viewer/Wiimote.cpp)
  endif()
endif()

set(RTNEURON_CORE_PUBLIC_HEADERS ${RTNEURON_CORE_PUBLIC_HEADERS_RELATIVE})
