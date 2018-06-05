//////////////////////////////////////////////////////////////////////
// RTNeuron
//
// Copyright (c) 2006-2016 Cajal Blue Brain, BBP/EPFL
// All rights reserved. Do not distribute without permission.
//
// Responsible Author: Juan Hernando Vieites (JHV)
// contact: jhernando@fi.upm.es
//////////////////////////////////////////////////////////////////////

#include "FragmentData.h"

#include "rtneuron/rtneuron/viewer/osgEq/MultiFragmentCompositor.h"
#include "rtneuron/rtneuron/viewer/osgEq/ChannelCompositor.h"
#include "rtneuron/rtneuron/cuda/CUDAContext.h"

#include <osg/Image>
#include <osgDB/WriteFile>

#include <co/global.h>
#include <co/localNode.h>
#include <co/connectionDescription.h>

#include <hwsd/netInfo.h>
#include <hwsd/hwsd.h>
#include <hwsd/net/sys/module.h>
#include <hwsd/net/dns_sd/module.h>

#include <lunchbox/fork.h>

#include <boost/filesystem/operations.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace bbp::rtneuron::osgEq;
using bbp::rtneuron::core::ConnectionDescriptions;

/* We want to detect the IB interfaces. */
const hwsd::NetInfo::Type _netType = hwsd::NetInfo::TYPE_INFINIBAND;
/* This can be RDMA o ETHERNET. With ETHERNET we will be doing IPoIB */
co::ConnectionType _coNetType = co::CONNECTIONTYPE_TCPIP;

size_t _nodeIndex;
int _deviceCount;

bool _writeImages = false;

std::vector<std::string> _inputFiles;

using bbp::rtneuron::core::CUDAContext;

/**
   Returns a list of ConnectionDescriptionPtr for all participating nodes.

   @param session hw_sd session name
   @param local Index of the local interface
   @param The list of connection descriptions, guaranteed to be the same order
          in all nodes.
*/
ConnectionDescriptions discoverNodes(const std::string& session, size_t& local)
{
    /* Finding all infiniband end points announced in the hwsd session. */
    const hwsd::FilterPtr filter =
        new hwsd::SessionFilter( session.empty() ? "default" : session ) |
        hwsd::FilterPtr(new hwsd::DuplicateFilter) |
        new hwsd::NetFilter(lunchbox::Strings(), _netType);

    hwsd::net::dns_sd::Module::use();
    hwsd::NetInfos netInfos = hwsd::discoverNetInfos(filter);
    hwsd::net::dns_sd::Module::dispose();

    /* The nodes will be sorted alphabetically based on their hostname. */
    std::sort(netInfos.begin(), netInfos.end(),
              [](const hwsd::NetInfo& left, const hwsd::NetInfo& right)
                  { return left.hostname < right.hostname; });

    ConnectionDescriptions descs;
    for (const auto& info : netInfos)
    {
        if (!info.up)
            continue;

        co::ConnectionDescriptionPtr desc(new co::ConnectionDescription);
        desc->type = _coNetType;
        desc->port = 7777;
        desc->hostname = info.inetAddress;
        if (info.nodeName.empty())
            local = descs.size();
        descs.push_back(desc);
    }

    return descs;
}

void launchRemoteNodes(const ConnectionDescriptions& descs,
                       const size_t localIndex, int argc, char** argv)
{
    namespace fs = boost::filesystem;
    const fs::path executable = fs::canonical(argv[0], ".");

    for (size_t i = 0; i != descs.size(); ++i)
    {
        if (i == localIndex)
            continue;

        auto desc = descs[i];
        std::stringstream command;
        command << "ssh " << desc->hostname
                << " unset CUDA_VISIBLE_DEVICES; " << executable.string()
                << " --client";
        if (_writeImages)
            command << " --out";
        for (int j = 1; j != argc; ++j)
        {
            if (argv[j][0] == '.')
            {
                const fs::path path = fs::canonical(argv[j], ".");
                command << " " << path.string();
            }
            else
                command << " " << argv[j];
        }

        lunchbox::fork(command.str());
    }
}

ConnectionDescriptions init(int argc, char** argv)
{
    std::string session;
    bool isMaster = true;

    for (int i = 1; i != argc; ++i)
    {
        if (!strcmp(argv[i], "--session"))
        {
            if (i == argc - 1)
            {
                std::cerr << "Missing parameter for --session" << std::endl;
                exit(-1);
            }
            else
                session = argv[++i];
        }
        else if (!strcmp(argv[i], "--rdma"))
        {
            _coNetType = co::CONNECTIONTYPE_RDMA;
        }
        else if (!strcmp(argv[i], "--ethernet"))
        {
            _coNetType = co::CONNECTIONTYPE_TCPIP;
        }
        else if (!strcmp(argv[i], "--client"))
        {
            isMaster = false;
        }
        else if (!strcmp(argv[i], "--out"))
        {
            _writeImages = true;
        }
        else
            _inputFiles.push_back(argv[i]);
    }

    /* Detecting available CUDA devices. All nodes are assumed to have the
       same number of devices. */
    const cudaError_t error = cudaGetDeviceCount(&_deviceCount);
    if (error != 0)
    {
        std::cerr << "Error getting device count: " << cudaGetErrorString(error)
                  << std::endl;
        exit(-1);
    }

    ConnectionDescriptions peers = discoverNodes(session, _nodeIndex);
    if (peers.empty())
    {
        std::cerr << "No nodes founds. hw_sd not running?" << std::endl;
        exit(-1);
    }

    const size_t parts = _inputFiles.size();
    if (peers.size() * _deviceCount < parts)
    {
        std::cerr << "Insufficient devices found to process all input files"
                  << std::endl;
        exit(-1);
    }
    /* Removing from the node list those that are not needed. */
    const size_t nodesNeeded = (parts +  _deviceCount - 1) / _deviceCount;
    if (nodesNeeded < peers.size())
        peers.erase(peers.begin() + nodesNeeded, peers.end());

    if (nodesNeeded * _deviceCount != parts)
    {
        std::cerr << "The number of input files must be a multiple of the "
                     "number of devices per node (" << _deviceCount << ")"
                  << std::endl;
        exit(-1);
    }

    /* Launching the other nodes and connecting to them */
    if (isMaster)
        launchRemoteNodes(peers, _nodeIndex, argc, argv);

    return peers;
}

void writeImage(const std::string& filename,
                const char* data, const size_t width,
                const size_t height)
{
    osg::ref_ptr<osg::Image> image(new osg::Image());

    /* Undoing alpha multiplication. */
    uint32_t* out = new uint32_t[width * height];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < width * height; ++i)
    {
        const uint32_t pixel = ((uint32_t*)data)[i];
        const uint32_t alpha = pixel >> 24;
        if (alpha != 0)
        {
            const uint32_t blue = (pixel >> 16) & 0xff;
            const uint32_t green = (pixel >> 8) & 0xff;
            const uint32_t red = pixel & 0xff;
            out[i] = (( alpha << 24 ) |
                     (((255 * blue) / alpha ) << 16 ) |
                     (((255 * green) / alpha )  << 8 ) |
                     ((255 * red) / alpha ));
        }
        else
            out[i] = 0;
    }

    image->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                    (unsigned char*)out, osg::Image::USE_NEW_DELETE);
    osgDB::writeImageFile(*image, filename);
}

void simulateFrame(const DeviceData& partial, ChannelCompositor& compositor)
{
    compositor.setup(partial.counts, partial.heads, partial.rawFragments.get(),
                     partial.items);

    const size_t parts = _inputFiles.size();
    const size_t width = partial.width;
    const size_t height = partial.height;
    const size_t fraction = height / parts;
    PixelViewports viewports;
    for (size_t i = 0; i != parts; ++i)
        viewports.emplace_back(eq::fabric::PixelViewport(0, fraction * i,
                                                         width, fraction));

    const size_t remainder = height % parts;
    for (size_t i = 0; i != remainder; ++i)
    {
        ++viewports[i].h;
        /* i can't be == parts - 1 if remainder is not zero. */
        viewports[i].y += i + 1;
    }
    compositor.extractFrameParts(viewports);
    charPtr output = compositor.compositeFrameParts(osg::Vec4());

    if (_writeImages)
    {
        size_t regionID = compositor.getFrameRegion();
        const auto& viewport = viewports[regionID];
        std::stringstream filename;
        filename << "frame_" << regionID << ".png";
        writeImage(filename.str(), output.get(), viewport.w, viewport.h);
    }
}

void simulateFrames(MultiFragmentCompositor& compositor,
                    const size_t index, const size_t frames)
{
    const int device = index % _deviceCount;

    osg::ref_ptr<CUDAContext> context(new CUDAContext(device));
    bbp::rtneuron::core::ScopedCUDAContext scope(context);

    /* FragmentData doesn't like () initialization in this case */
    DeviceData input(FragmentData{_inputFiles[index]});
    cudaStreamSynchronize(0);

    ChannelCompositorPtr channelCompositor =
        compositor.createChannelCompositor(context.get());

    for (size_t frame = 0; frame != frames; ++frame)
    {
        using namespace std::chrono;
        const auto frameStart = high_resolution_clock::now();
        simulateFrame(input, *channelCompositor);

        if (index == 0)
        {
            const auto frameEnd = high_resolution_clock::now();
            const auto ellapsed =
                duration_cast<microseconds>(frameEnd - frameStart);
            std::cout << "frame " << frame << ' '
                      << ellapsed.count() << " us" << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    const size_t frames = 10;

    ConnectionDescriptions descs = init(argc, argv);

    if (_nodeIndex == 0)
    {
        std::cout << "Processing " << _inputFiles.size() << " files in "
                  << descs.size() << " nodes" << std::endl;
    }

    /* Uploading frame data to the GPU outside the loop as we don't want
       this to be taken into account.  */
    std::vector<std::thread> threads;

    size_t firstDevice = _nodeIndex * _deviceCount;
    size_t lastDevice = firstDevice + _deviceCount;

    MultiFragmentCompositor compositor(descs, _nodeIndex, _deviceCount);

    for (size_t i = firstDevice; i != lastDevice; ++i)
        threads.push_back(
            std::thread(simulateFrames, std::ref(compositor), i, frames));

    using namespace std::chrono;
    const auto startTime = high_resolution_clock::now();

    for (auto& t : threads)
        t.join();

    if (_nodeIndex == 0)
    {
        const auto endTime = high_resolution_clock::now();
        const auto ellapsed = duration<float>(endTime - startTime).count();
        std::cout << "total_time " << ellapsed << " s "
                  << 1/ellapsed * frames << " fps" << std::endl;
    }
}
