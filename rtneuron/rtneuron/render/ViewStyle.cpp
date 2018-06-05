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

#include "ViewStyle.h"

#include "../ColorMap.h"
#include "ViewStyleData.h"
#include "config/constants.h"
#include "data/loaders.h"
#include "detail/Configurable.h"
#include "net/DataIStreamArchive.h"
#include "net/DataOStreamArchive.h"
#include "render/CameraData.h"
#include "render/ColorMap.h"
#include "render/Noise.h"
#include "util/attributeMapHelpers.h"

#include <osg/Depth>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Program>
#include <osg/StateSet>

#include <boost/foreach.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <mutex>

BOOST_CLASS_EXPORT_IMPLEMENT(bbp::rtneuron::core::ViewStyle)

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const float DEFAULT_CLOD_THRESHOLD = 1500.f;
const float DEFAULT_SPIKE_TAIL_LENGTH = 2.f;
const osg::Vec4 DEFAULT_PROBLE_COLOR(1.f, 1.f, 1.f, 1.f);
const osg::Vec4 DEFAULT_HIGHLIGHT_COLOR(1.f, -.5f, -.5f, 1.f);

float _correctedLODBias(const float bias)
{
    if (bias == 1.f)
        return std::numeric_limits<double>::infinity();
    return 1.f / (1.f - bias) - 1.f;
}

void _defaultColorMapPoints(rtneuron::ColorMap::ColorPoints& points)
{
    points[-80.0f] = osg::Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    points[-77.181205f] = osg::Vec4(0.023529f, 0.023529f, 0.6549020f, 0.05f);
    points[-72.06669f] = osg::Vec4(0.141176f, 0.529412f, 0.9607843f, 0.16f);
    points[-70.2f] = osg::Vec4(0.388235f, 0.345098f, 0.7137255f, 0.22f);
    points[-67.4f] = osg::Vec4(0.960784f, 0.000000f, 0.0196078f, 0.3f);
    points[-61.67785f] = osg::Vec4(0.858824f, 0.674510f, 0.0000000f, 0.4f);
    points[-31.47f] = osg::Vec4(0.964706f, 1.000000f, 0.6313725f, 0.8f);
    points[-10.0f] = osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

void _applyDefaults(AttributeMap& attributes)
{
    attributes.set("lod_bias", DEFAULT_LOD_BIAS);
    attributes.set("clod_threshold", DEFAULT_CLOD_THRESHOLD);
    attributes.set("spike_tail", DEFAULT_SPIKE_TAIL_LENGTH);
    attributes.set("display_simulation", false);
    attributes.set("inflation_factor", 0.0);
    attributes.set("probe_threshold",
                   REPORTED_VARIABLE_DEFAULT_THRESHOLD_VALUE);
    osg::Vec4 c(DEFAULT_PROBLE_COLOR);
    attributes.set("probe_color", c[0], c[1], c[2], c[3]);
    c = DEFAULT_HIGHLIGHT_COLOR;
    attributes.set("highlight_color", c[0], c[1], c[2], c[3]);

    AttributeMapPtr colorMaps(new AttributeMap());
    attributes.set("colormaps", colorMaps);
    /* Color map for compartmental simulation data */
    ColorMapPtr colorMap(new rtneuron::ColorMap());
    std::string colorMapFile = attributes("color_map_filename", "");
    if (!colorMapFile.empty())
        colorMap->load(colorMapFile);
    else
    {
        rtneuron::ColorMap::ColorPoints points;
        _defaultColorMapPoints(points);
        colorMap->setPoints(points);
        colorMap->getImpl().createUniforms(
            COMPARTMENT_COLOR_MAP_UNIFORM_PREFIX,
            COMPARTMENT_COLOR_MAP_TEXTURE_NUMBER);
    }
    colorMaps->set("compartments", colorMap);

    /* Color map for spikes */
    colorMap.reset(new rtneuron::ColorMap());
    rtneuron::ColorMap::ColorPoints points;
    points[0.0] = osg::Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    points[1.0] = osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    colorMap->setPoints(points);
    colorMap->getImpl().createUniforms(SPIKE_COLOR_MAP_UNIFORM_PREFIX,
                                       SPIKE_COLOR_MAP_TEXTURE_NUMBER);
    colorMaps->set("spikes", colorMap);
}
}

/*
  ViewStyle::Impl
*/
class ViewStyle::Impl : public rtneuron::detail::Configurable
{
public:
    /*--- Public declarations ---*/

    class UpdateCallback : public osg::NodeCallback
    {
    public:
        UpdateCallback(Impl* style)
            : _style(style)
        {
        }

        virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
        {
            if (_style == 0)
            {
                /* Deregistering the callback from the node. This may
                   deallocate this object. */
                node->setUpdateCallback(0);
                return;
            }

            _style->update();

            traverse(node, nv);
        }

        virtual void invalidate() { _style = 0; }
    private:
        Impl* _style;
    };

    class AttributeUpdateProxy : public rtneuron::detail::Configurable
    {
    public:
        AttributeUpdateProxy(Impl* style)
            : _style(style)
        {
        }

    private:
        void onAttributeChangedImpl(const AttributeMap& attributes,
                                    const std::string& attribute)
        {
            if (attribute == "spike_tail" && _style->_spikeTail.get() != 0)
            {
                double value = attributes("spike_tail");
                _style->_spikeTail->set((float)value);
            }
            else if (attribute == "probe_threshold")
            {
                double value = attributes("probe_threshold");
                _style->_probeThreshold->set((float)value);
            }
            else if (attribute == "probe_color")
            {
                osg::Vec4 color;
                AttributeMapHelpers::getColor(attributes, "probe_color", color);
                _style->_probeColor->set(color);
            }
            else if (attribute == "display_simulation")
            {
                _style->_useStaticColor->set(!attributes("display_simulation"));
            }
            else if (attribute == "lod_bias" || attribute == "clod_threshold")
            {
                _style->_applyLODAttributes(attributes);
            }
            else if (attribute == "inflation_factor")
            {
                double value = attributes("inflation_factor");
                _style->_inflationFactor->set((float)value);
            }
            else if (attribute == "highlight_color")
            {
                osg::Vec4 color;
                AttributeMapHelpers::getColor(attributes, "highlight_color",
                                              color);
                _style->_highlightColor->set(color);
            }
            else if (attribute == "colormaps")
            {
                const AttributeMapPtr colormaps = attributes("colormaps");
                _style->_updateColorMaps(*colormaps);
            }
        }

        Impl* _style;
    };

    /*--- Public constructor/destructor ---*/

    Impl(ViewStyle& parent, const AttributeMap& attributes)
        : _initPending(true)
        , _parent(parent)
        , _attributeUpdateProxy(this)
    {
        AttributeMap& internal = getAttributes();

        blockAttributeMapSignals();
        _applyDefaults(internal);
        internal.merge(attributes);
        /* color_map_filename doesn't need to be propagated or stored */
        internal.unset("color_map_filename");
        unblockAttributeMapSignals();

        _toSerialize.merge(internal);
    }

    ~Impl()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_callback.valid())
            _callback->invalidate();
    }

    /*--- Public member functions ---*/

    osg::Group* decorateScenegraph(osg::Node* root)
    {
        osg::Group* viewRoot = new osg::Group();
        _callback = new UpdateCallback(this);
        viewRoot->setUpdateCallback(_callback);
        viewRoot->addChild(root);

        viewRoot->setStateSet(_stateSet);

        return viewRoot;
    }

    void updateCamera(osg::Camera* camera) const
    {
        CameraData* data = CameraData::getOrCreateCameraData(camera);
        data->viewStyle = _styleData;
    }

    void update()
    {
        std::lock_guard<std::mutex> lock(_mutex);

        if (_initPending)
            _init();

        /* Applying attribute updates and clearing the attribute map
           that stores them.
           Attributes are applied when
           AttributeUpdateProxy::onAttributeChangedImpl is called by the
           merge. */
        _attributeUpdateProxy.getAttributes().merge(_toApply);
        _toApply.clear();
    }

    ViewStyleData* getData() { return _styleData.get(); }
    void onAttributeChangedImpl(const AttributeMap& attributes,
                                const std::string& attribute)
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _toSerialize.copy(attributes, attribute);
        }
        /* Emitting SceneDecorator::dirty signal */
        _parent.dirty();
    }

    template <class Archive>
    void load(Archive& archive, const unsigned int)
    {
        AttributeMap updates;
        archive& updates;

        std::lock_guard<std::mutex> lock(_mutex);
        _toApply.merge(updates);
    }

    template <class Archive>
    void save(Archive& archive, const unsigned int) const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        archive& _toSerialize;
        _toSerialize.clear();
    }

private:
    /*--- Private member attributes ---*/
    mutable std::mutex _mutex;

    bool _initPending;

    ViewStyle& _parent;

    std::map<std::string, ColorMapPtr> _colorMaps;

    osg::ref_ptr<osg::Uniform> _probeThreshold;
    osg::ref_ptr<osg::Uniform> _probeColor;
    osg::ref_ptr<osg::Uniform> _spikeTail;

    osg::ref_ptr<osg::Uniform> _useStaticColor;
    osg::ref_ptr<osg::Uniform> _inflationFactor;
    osg::ref_ptr<osg::Uniform> _clodBias;

    osg::ref_ptr<osg::Uniform> _highlightColor;

    osg::ref_ptr<osg::StateSet> _stateSet;

    osg::ref_ptr<ViewStyleData> _styleData;

    mutable AttributeMap _toSerialize;
    AttributeMap _toApply;

    osg::ref_ptr<UpdateCallback> _callback;

    AttributeUpdateProxy _attributeUpdateProxy;

    /*--- Private member functions ---*/

    void _init()
    {
        const AttributeMap& attributes = getAttributes();

        /* View style data */
        _styleData = new ViewStyleData();

        /* Creating the decorating state set */
        _stateSet = new osg::StateSet();

        /* LOD attributes, uniforms, ... */
        _clodBias = new osg::Uniform("clodThreshold", float());
        _applyLODAttributes(attributes);
        _stateSet->addUniform(_clodBias);

        /* Simulation display uniforms */
        const float probeThreshold = double(attributes("probe_threshold"));
        _probeThreshold = new osg::Uniform("threshold", probeThreshold);
        _stateSet->addUniform(_probeThreshold);
        osg::Vec4 color;
        AttributeMapHelpers::getColor(attributes, "probe_color", color);
        _probeColor = new osg::Uniform("aboveThresholdColor", color);
        _stateSet->addUniform(_probeColor);
        const float spikeTail = double(attributes("spike_tail"));
        _spikeTail = new osg::Uniform("spikeTail", spikeTail);
        _stateSet->addUniform(_spikeTail);

        /* Display mode uniforms */
        AttributeMapHelpers::getColor(attributes, "highlight_color", color);
        _highlightColor = new osg::Uniform("highlightColor", color);
        _stateSet->addUniform(_highlightColor);
        _useStaticColor = new osg::Uniform("useStaticColor", true);
        _stateSet->addUniform(_useStaticColor);
        _inflationFactor = new osg::Uniform("inflation", 0.0f);
        _stateSet->addUniform(_inflationFactor);
        /* Fallback uniforms for per model configurable display modes */
        _stateSet->addUniform(new osg::Uniform("useColorMap", false));
        _stateSet->addUniform(new osg::Uniform("highlighted", false));

        /* Colormap uniforms */

        _initPending = false;
    }

    void _applyLODAttributes(const AttributeMap& attributes)
    {
        /* We cannot assume the presence of any defaults in attributes
           because this function is also used during the first udpate
           performed by AttributeUpdateProxy. */
        const float bias =
            _correctedLODBias(double(attributes("lod_bias", DEFAULT_LOD_BIAS)));
        _styleData->_lodBias = bias;
        const double clodThreshold = attributes("clod_threshold");
        _clodBias->set(float(clodThreshold) * bias);
    }

    void _updateColorMaps(const AttributeMap& colorMaps)
    {
        for (AttributeMap::const_iterator i = colorMaps.begin();
             i != colorMaps.end(); ++i)
        {
            const std::string name = i->first;
            ColorMapPtr incoming = i->second;
            ColorMapPtr& colorMap = _colorMaps[name];
            if (!colorMap)
            {
                colorMap.reset(new rtneuron::ColorMap());
                if (name == "compartments")
                    colorMap->getImpl().createUniforms(
                        COMPARTMENT_COLOR_MAP_UNIFORM_PREFIX,
                        COMPARTMENT_COLOR_MAP_TEXTURE_NUMBER);
                else if (name == "spikes")
                    colorMap->getImpl().createUniforms(
                        SPIKE_COLOR_MAP_UNIFORM_PREFIX,
                        SPIKE_COLOR_MAP_TEXTURE_NUMBER);
                else
                    LBWARN << "Unknown colormap '" << name << "'" << std::endl;
                colorMap->getImpl().addStateAttributes(_stateSet);
            }
            *colorMap = *incoming;
        }
    }
};

/*
  ViewStyle
*/
ViewStyle::ViewStyle(const AttributeMap& attributes)
    : _impl(new Impl(*this, attributes))
{
}

ViewStyle::~ViewStyle()
{
    delete _impl;
    _impl = 0;
}

osg::Group* ViewStyle::decorateScenegraph(osg::Node* root)
{
    return _impl->decorateScenegraph(root);
}

void ViewStyle::updateCamera(osg::Camera* camera) const
{
    return _impl->updateCamera(camera);
}

void ViewStyle::update()
{
    return _impl->update();
}

osgEq::SceneDecoratorPtr ViewStyle::clone() const
{
    auto viewStyle = std::make_shared<ViewStyle>();
    viewStyle->getAttributes().merge(getAttributes());
    return viewStyle;
}

AttributeMap& ViewStyle::getAttributes()
{
    return _impl->getAttributes();
}

const AttributeMap& ViewStyle::getAttributes() const
{
    return _impl->getAttributes();
}

void ViewStyle::validateAttributeChange(
    const AttributeMap&, const std::string& name,
    const AttributeMap::AttributeProxy& parameters)
{
    if (name == "highlight_color")
    {
        osg::Vec4 color;
        core::AttributeMapHelpers::getRequiredColor(parameters, color);
    }
    else if (name == "spike_tail")
    {
        (void)(double) parameters;
    }
    else if (name == "probe_threshold")
    {
        (void)(double) parameters;
    }
    else if (name == "probe_color")
    {
        osg::Vec4 color;
        AttributeMapHelpers::getRequiredColor(parameters, color);
    }
    else if (name == "display_simulation")
    {
        (void)(bool) parameters;
    }
    else if (name == "lod_bias")
    {
        (void)(double) parameters;
    }
    else if (name == "clod_threshold")
    {
        (void)(double) parameters;
    }
    else if (name == "inflation_factor")
    {
        double factor = parameters;
        if (factor < 0)
            throw std::runtime_error(
                "Negative values not allowed for inflation_factor");
    }
    else if (name == "colormaps")
    {
        try
        {
            const AttributeMapPtr maps = parameters;
            (void)maps;
        }
        catch (...)
        {
            throw std::runtime_error(
                "The attribute colormaps must be an AttributeMap");
        }
    }
    else if (name.substr(0, 10) == "colormaps.")
    {
        try
        {
            const ColorMapPtr map = parameters;
            (void)map;
        }
        catch (...)
        {
            throw std::runtime_error("Attribute must be a ColorMap");
        }
    }
    else
        throw std::runtime_error("Unknown or immutable attribute: " + name);
}

template <class Archive>
void ViewStyle::serialize(Archive& archive, const unsigned int version)
{
    archive& boost::serialization::base_object<SceneDecorator>(*this);
    boost::serialization::split_member(archive, *this, version);
}

template <class Archive>
void ViewStyle::load(Archive& archive, const unsigned int version)
{
    _impl->load(archive, version);
}

template <class Archive>
void ViewStyle::save(Archive& archive, const unsigned int version) const
{
    _impl->save(archive, version);
}

template void ViewStyle::serialize<net::DataOStreamArchive>(
    net::DataOStreamArchive& archive, const unsigned int version);

template void ViewStyle::serialize<net::DataIStreamArchive>(
    net::DataIStreamArchive& archive, const unsigned int version);

template void ViewStyle::load<net::DataIStreamArchive>(
    net::DataIStreamArchive& archive, const unsigned int version);

template void ViewStyle::save<net::DataOStreamArchive>(
    net::DataOStreamArchive& archive, const unsigned int version) const;

ViewStyleData* ViewStyle::getData()
{
    return _impl->getData();
}
}
}
}
