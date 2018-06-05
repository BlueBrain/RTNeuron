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

#include "RenderBinManager.h"

#include "AttributeMap.h"
#include "config/constants.h"
#include "types.h"
#include "util/attributeMapHelpers.h"

#include <osg/Version>
#include <osgTransparency/BaseParameters.h>
#include <osgTransparency/DepthPeelingBin.h>
#include <osgTransparency/MultiLayerDepthPeelingBin.h>
#include <osgTransparency/MultiLayerParameters.h>
#ifdef OSG_GL3_AVAILABLE
#include <osgTransparency/FragmentListOITBin.h>
#endif

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
/*
  Helper classes
*/
struct RegisterRenderBinProxy
{
    RegisterRenderBinProxy(const std::string& binName,
                           osgUtil::RenderBin* proto)
    {
        bin = proto;
        osgUtil::RenderBin::addRenderBinPrototype(binName, bin);
    }
    ~RegisterRenderBinProxy()
    {
        osgUtil::RenderBin::removeRenderBinPrototype(bin);
    }
    osg::ref_ptr<osgUtil::RenderBin> bin;
};

typedef std::map<uint32_t, std::weak_ptr<RegisterRenderBinProxy>>
    RegisterRenderBinProxies;
RegisterRenderBinProxies _registeredRenderBins;
OpenThreads::Mutex _registeredBinsMutex;

enum Algorithm
{
    MLDP,
    FLL,
    DP
};

/*
  Helper functions
*/
Algorithm _getAlphaBlendingAlgorithm(const AttributeMap& attributes)
{
    const std::string mode = attributes("mode");

    Algorithm algorithm = MLDP;
    if (mode == "auto")
    {
#ifdef OSG_GL3_AVAILABLE
        algorithm = FLL;
#endif
    }
#ifdef OSG_GL3_AVAILABLE
    else if (mode == "fragment_linked_list")
        algorithm = FLL;
#endif
    else if (mode == "depth_peeling")
        algorithm = DP;
    else if (mode != "multilayer_depth_peeling")
    {
#ifdef OSG_GL3_AVAILABLE
        algorithm = FLL;
#endif
        std::cerr << "Warning: unknown alpha-blending algorithm"
                     " requested: "
                  << mode << ". Falling back to"
#ifdef OSG_GL3_AVAILABLE
                  << " fragment_linked_list."
#else
                  << " multilayer_depth_peeling."
#endif
                  << " Available algorithms are: depth_peeling"
#ifdef OSG_GL3_AVAILABLE
                     ", fragment_linked_list"
#endif
                     " and multilayer_depth_peeling."
                  << std::endl;
    }
    return algorithm;
}
}

class RenderBinManager::Impl
{
public:
    Impl(const AttributeMap& attributes)
    {
        rtneuron::AttributeMapPtr alphaBlending =
            attributes("alpha_blending", AttributeMapPtr());
        if (alphaBlending && !alphaBlending->empty())
        {
            using namespace AttributeMapHelpers;
            const auto partitioning =
                getEnum<DataBasePartitioning>(attributes, "partitioning",
                                              DataBasePartitioning::NONE);
            _createRenderBin(*alphaBlending, partitioning);
        }
    }

    std::shared_ptr<RegisterRenderBinProxy> _registeredRenderBin;

private:
    void _createRenderBin(const AttributeMap& attributes,
                          const DataBasePartitioning partitioning)
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_registeredBinsMutex);

        using namespace osgTransparency;

        std::stringstream keystr;
        keystr << attributes.hash();
        const std::string name = "ab_" + keystr.str();
        auto registeredRenderBin =
            _registeredRenderBins[attributes.hash()].lock();

        if (registeredRenderBin)
        {
            _registeredRenderBin = registeredRenderBin;
            return;
        }

        Algorithm algorithm = _getAlphaBlendingAlgorithm(attributes);

        const std::string error(
            "Unsupported DB partition type for alpha-blending algorithm.");

        osg::ref_ptr<BaseRenderBin> renderBin;
        switch (algorithm)
        {
        case MLDP:
        {
            int slices = attributes("slices", 1);
            MultiLayerDepthPeelingBin::Parameters parameters(slices);
            if (partitioning == DataBasePartitioning::ROUND_ROBIN)
                throw std::runtime_error(error);
            renderBin = new MultiLayerDepthPeelingBin(parameters);
            break;
        }
        case FLL:
        {
#ifdef OSG_GL3_AVAILABLE
            FragmentListOITBin::Parameters parameters;
            double threshold;
            if (attributes.get("alpha_cutoff_threshold", threshold) == 1)
                /* May throw */
                parameters.enableAlphaCutOff(threshold);
            renderBin = new FragmentListOITBin(parameters);
#endif
            break;
        }
        case DP:
        {
            if (partitioning == DataBasePartitioning::ROUND_ROBIN)
                throw std::runtime_error(error);
            renderBin = new DepthPeelingBin();
            break;
        }
        }

        DepthPeelingBin::Parameters& parameters = renderBin->getParameters();
        int value = 0;
        if (attributes.get("cutoff_samples", value) == 1)
            parameters.samplesCutoff = value;
        if (attributes.get("max_passes", value) == 1)
            parameters.maximumPasses = value;
        parameters.reservedTextureUnits = MAX_TEXTURE_UNITS;

        /* This sentence forces the alpha blending render bin
           implementation to be global instead of scene specific. */
        renderBin->setName(name);
        _registeredRenderBin.reset(new RegisterRenderBinProxy(name, renderBin));
        _registeredRenderBins[attributes.hash()] = _registeredRenderBin;
    }
};

RenderBinManager::RenderBinManager(const AttributeMap& attributes)
    : _impl(new Impl(attributes))
{
}

RenderBinManager::~RenderBinManager()
{
}

void RenderBinManager::validateAttributes(const AttributeMap& attributes)
{
    if (attributes.empty())
        return; /* Nothing to validate, this will disable transparency */

    std::string mode = attributes("mode");
    if (mode == "multilayer_depth_peeling")
    {
        try
        {
            /* Testing first for existance */
            AttributeMap::AttributeProxy proxy = attributes("slices");
            /* And then for correctness */
            try
            {
                (void)(int) proxy;
            }
            catch (...)
            {
                (void)(unsigned int) proxy;
            }
        }
        catch (...)
        {
        }
    }
#ifdef OSG_GL3_AVAILABLE
    else if (mode == "fragment_linked_list")
    {
    }
#endif
    else if (mode != "depth_peeling")
    {
    }
    else if (mode != "auto")
    {
        throw std::runtime_error("Unknown alpha blending mode: " + mode);
    }
}

osgTransparency::BaseRenderBin* RenderBinManager::getAlphaBlendedRenderBin()
    const
{
    if (!_impl->_registeredRenderBin)
        return 0;
    return static_cast<osgTransparency::BaseRenderBin*>(
        _impl->_registeredRenderBin->bin.get());
}
}
}
}
