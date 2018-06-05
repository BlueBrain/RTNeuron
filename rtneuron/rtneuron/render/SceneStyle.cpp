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

#include "SceneStyle.h"

#include "config/constants.h"
#include "data/loaders.h"
#include "render/ColorMapAtlas.h"
#include "render/Noise.h"
#include "render/RenderBinManager.h"
#include "types.h"

#include "detail/Configurable.h"

#include <osgTransparency/BaseRenderBin.h>

#include <osg/CullFace>
#include <osg/Depth>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Program>
#include <osg/StateSet>

#include <osg/Billboard>
#include <osg/BlendEquation>
#include <osg/BlendFunc>
#include <osg/CameraView>
#include <osg/ClipNode>
#include <osg/CoordinateSystemNode>
#include <osg/Depth>
#include <osg/FrontFace>
#include <osg/Image>
#include <osg/LightSource>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/OccluderNode>
#include <osg/OcclusionQueryNode>
#include <osg/PagedLOD>
#include <osg/PositionAttitudeTransform>
#include <osg/Projection>
#include <osg/ProxyNode>
#include <osg/Sequence>
#include <osg/Switch>
#include <osg/TexGenNode>
#include <osg/Texture1D>
#include <osg/Version>

#include <osgDB/ReadFile>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <lunchbox/scopedMutex.h>

/* This define is needed because in the tested drivers we need to enable
   GL_POINT_SPRITE for core GL3 profiles. The glcorearb.h header doesn't
   contain it anymore because it should be always on. */
#ifdef OSG_GL3_AVAILABLE
#define GL_POINT_SPRITE 0x8861
#endif

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Helper functions
*/
namespace
{
void _setNoiseTextureForSampler(osg::StateSet* state)
{
    static osg::ref_ptr<osg::Texture3D> noiseTexture;
    if (!noiseTexture.valid())
        noiseTexture = make3DNoiseTexture(128);

    static osg::ref_ptr<osg::Uniform> noiseUniform;
    if (!noiseUniform.valid())
        noiseUniform = new osg::Uniform("noise", NOISE_TEXTURE_NUMBER);

    state->addUniform(noiseUniform.get());
    state->setTextureAttribute(NOISE_TEXTURE_NUMBER, noiseTexture.get());
}

/*
  This helper class is a node visitor that traverses a scenegraph and
  prepares the geode objects whose render bin is the alpha blended bin to
  work correctly.
*/
class FixTransparentObjects : public osg::NodeVisitor
{
    /* Constructors */
public:
    FixTransparentObjects(osgTransparency::BaseRenderBin* renderBin)
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
        , _renderBin(renderBin)
    {
    }

    /* Member functions */
public:
    virtual void apply(osg::Node& node)
    {
        if (!checkState(node))
            osg::NodeVisitor::apply(node);
    }

    virtual void apply(osg::Group& node)
    {
        if (!checkState(node))
            osg::NodeVisitor::apply(node);
    }

    virtual void apply(osg::Geode& node) { checkState(node); }
    bool checkState(osg::Node& node)
    {
        osg::StateSet* stateSet = node.getStateSet();
        if (stateSet)
        {
            if (stateSet->getBinName() == "alphaBlended")
            {
                osg::Program* program = dynamic_cast<osg::Program*>(
                    stateSet->getAttribute(osg::StateAttribute::PROGRAM));
                if (program)
                {
                    _renderBin->addExtraShadersForState(stateSet, program);
                    stateSet->removeAttribute(program);
                    return true;
                }
            }
        }
        return false;
    }

    /* Member attributes */
protected:
    osgTransparency::BaseRenderBin* _renderBin;
};

/*
  This helper class is a node visitor that traverses a scenegraph and
  to check out whether it has any shader program or not.
*/
class FindPrograms : public osg::NodeVisitor
{
    /* Constructors */
public:
    FindPrograms()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
        , hasProgram(false)
    {
    }

    /* Member functions */
public:
#define APPLY(type)                        \
    virtual void apply(type& node)         \
    {                                      \
        if (!checkState(node))             \
            osg::NodeVisitor::apply(node); \
    }

    virtual void apply(osg::Geode& node) { checkState(node); }
    APPLY(osg::Node)
    APPLY(osg::Group)
    APPLY(osg::Transform)
    APPLY(osg::Billboard)
    APPLY(osg::ProxyNode)
    APPLY(osg::Projection)
    APPLY(osg::CoordinateSystemNode)
    APPLY(osg::ClipNode)
    APPLY(osg::TexGenNode)
    APPLY(osg::LightSource)
    APPLY(osg::Camera)
    APPLY(osg::CameraView)
    APPLY(osg::MatrixTransform)
    APPLY(osg::PositionAttitudeTransform)
    APPLY(osg::Switch)
    APPLY(osg::Sequence)
    APPLY(osg::LOD)
    APPLY(osg::PagedLOD)
    APPLY(osg::ClearNode)
    APPLY(osg::OccluderNode)
    APPLY(osg::OcclusionQueryNode)
#undef APPLY

    bool checkState(osg::Node& node)
    {
        osg::StateSet* stateSet = node.getStateSet();
        if (stateSet)
        {
            osg::Program* program = dynamic_cast<osg::Program*>(
                stateSet->getAttribute(osg::StateAttribute::PROGRAM));
            if (program)
            {
                hasProgram = true;
                return true;
            }
        }
        return false;
    }

    bool hasProgram;
};
}

/*
  SceneStyle::Impl
*/
class SceneStyle::Impl : public rtneuron::detail::Configurable
{
public:
    /*--- Public constructors/destructor ---*/

    Impl(const AttributeMap& attributes)
        : Configurable(attributes)
        , _renderBinManager(attributes)
    {
        _stateSets.resize(int(STATE_TYPE_COUNT));
    }

    ~Impl() {}
    /*--- Public member functions ---*/

    osg::StateSet* getOrCreateStateSet(
        const StateType type, const AttributeMap& extra = AttributeMap())
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

        /* Should we use a single hash (from the merge of the internal
           attributes and the extra ones) to identify the state set?. */

        const Key key(getAttributes().hash(), extra.hash());
        osg::ref_ptr<osg::StateSet>& stateSet = _stateSets[int(type)][key];

        if (!stateSet)
        {
            AttributeMap attributes = getAttributes();

            /* If there are extra attributes, creating a single attribute map
               from the internal attributes and the extra attributes
               provided and using that one. */
            if (extra != AttributeMap())
                attributes.merge(extra);

            switch (type)
            {
            case LINES:
                stateSet = _linesStateSet(attributes);
                break;
            case MESH:
                stateSet = _meshStateSet(attributes);
                break;
            case FLAT_MESH:
                stateSet = _flatMeshStateSet(attributes);
                break;
            case SPHERES:
                stateSet = _raycastSphereStateSet(attributes);
                break;
            case POINTS:
                stateSet = _pointStateSet(attributes);
                break;
            case NEURON_MESH:
                stateSet = _neuronMeshStateSet(attributes);
                break;
            case NEURON_TUBELETS:
                stateSet = _tubeletStateSet(attributes);
                break;
            case NEURON_PSEUDO_CYLINDERS:
                stateSet = _pseudoCylindersStateSet(attributes);
                break;
            case NEURON_SPHERICAL_SOMA:
                stateSet = _sphericalSomaStateSet(attributes);
                break;
            default:
                std::cerr << "Unreachable code" << std::endl;
                abort();
            }
        }

        return stateSet.get();
    }

    void setColorMapAtlas(const ColorMapAtlasPtr& colorMaps)
    {
        StateSetTable& states = _stateSets[int(NEURON_SPHERICAL_SOMA)];

        for (StateSetTable::const_iterator j = states.begin();
             j != states.end(); ++j)
        {
            osg::StateSet* stateSet = j->second.get();
            if (_colorMaps)
                _colorMaps->removeStateAttributes(stateSet);
            colorMaps->addStateAttributes(stateSet);
        }
        _colorMaps = colorMaps;
    }

    RenderBinManager _renderBinManager;

protected:
    /*--- Protected slots ---*/

    void onAttributeChangedImpl(const AttributeMap&, const std::string&)
    {
        LBTHROW(std::runtime_error("Immutable attribute map"));
    }

private:
    /*--- Private member variables ---*/

    /* To protect getOrCreateStateSet in parallel executions. */
    OpenThreads::Mutex _mutex;

    typedef boost::tuple<boost::uint32_t, boost::uint32_t> Key;
    typedef std::map<Key, osg::ref_ptr<osg::StateSet>> StateSetTable;
    typedef std::vector<StateSetTable> StateSets;

    StateSets _stateSets;

    ColorMapAtlasPtr _colorMaps;

    /*--- Private member functions ---*/

    osg::ref_ptr<osg::StateSet> _linesStateSet(const AttributeMap& attributes)
    {
        osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        if (attributes("accurate_headlight", false))
            vars["DEFINES"] += "#define ACCURATE_HEADLIGHT\n";

        std::vector<std::string> shaders;
        shaders.push_back("geom/flat_mesh.vert");
        shaders.push_back("geom/lit_lines.geom");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom");
#endif
        _meshCommonShaders(attributes, shaders);

        osg::Program* program = Loaders::loadProgram(shaders, vars);

        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 2);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Lit lines");

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _meshStateSet(const AttributeMap& attributes)
    {
        osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        if (attributes("accurate_headlight", false))
            vars["DEFINES"] += "#define ACCURATE_HEADLIGHT\n";
        if (attributes("color_uniform", false))
            vars["DEFINES"] += "#define USE_COLOR_UNIFORM\n";
        vars["DEFINES"] += "#define DOUBLE_FACED\n";

        std::vector<std::string> shaders;
        shaders.push_back("geom/mesh.vert");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom[vert]");
#endif
        _meshCommonShaders(attributes, shaders);

        osg::Program* program = Loaders::loadProgram(shaders, vars);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Mesh");

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _flatMeshStateSet(
        const AttributeMap& attributes)
    {
        osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        if (attributes("accurate_headlight", false))
            vars["DEFINES"] += "#define ACCURATE_HEADLIGHT\n";
        if (attributes("color_uniform", false))
            vars["DEFINES"] += "#define USE_COLOR_UNIFORM\n";
        vars["DEFINES"] += "#define DOUBLE_FACED\n";

        std::vector<std::string> shaders;
        shaders.push_back("geom/flat_mesh.vert");
        shaders.push_back("geom/flat_mesh.geom");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom");
#endif
        _meshCommonShaders(attributes, shaders);

        osg::Program* program = Loaders::loadProgram(shaders, vars);

        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Flat mesh");

        return stateSet;
    }

    void _meshCommonShaders(const AttributeMap& attributes,
                            std::vector<std::string>& shaders)
    {
        shaders.push_back("geom/default_color.vert");
        shaders.push_back("shading/default_color.frag");
        shaders.push_back("shading/phong_mesh.frag");
        shaders.push_back("shading/phong.frag");

        if (!attributes("alpha_blending", AttributeMapPtr()))
        {
            shaders.push_back("main.vert");
            shaders.push_back("main.frag");
        }
        shaders.push_back("geom/basic.vert");
    }

    osg::ref_ptr<osg::StateSet> _raycastSphereStateSet(
        const AttributeMap& attributes)
    {
        osg::StateSet* stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        vars["DEFINES"] += "#define OVERRIDE_DEPTH\n";
        if (attributes("accurate_headlight", false))
            vars["DEFINES"] += "#define ACCURATE_HEADLIGHT\n";
        if (attributes("use_radius_uniform", false))
            vars["DEFINES"] += "#define USE_RADIUS_UNIFORM\n";

        std::vector<std::string> shaders;
        shaders.push_back("geom/sphere.vert");
        shaders.push_back("geom/sphere.geom");
        shaders.push_back("geom/default_color.vert");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom");
#endif
        shaders.push_back("shading/phong_sphere.frag");
        shaders.push_back("shading/phong.frag");
        if (!attributes("alpha_blending", AttributeMapPtr()))
        {
            shaders.push_back("main.vert");
            shaders.push_back("main.frag");
        }

        osg::Program* program = Loaders::loadProgram(shaders, vars);
        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Raycast sphere");

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _pointStateSet(const AttributeMap& attributes)
    {
        osg::StateSet* stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        if (attributes("use_point_size_uniform", false))
            vars["DEFINES"] += "#define USE_POINT_SIZE_UNIFORM\n";
        if (attributes("circles", false))
            vars["DEFINES"] += "#define CIRCLES\n";

        std::vector<std::string> shaders;
        shaders.push_back("geom/point.vert");
        shaders.push_back("geom/default_color.vert");
#ifdef OSG_GL3_AVAILABLE
        /* computeClipDistances is used from a vertex shader, not a geometry
           shader. */
        shaders.push_back("geom/compute_clip_distances.geom[vert]");
#endif
        shaders.push_back("shading/point.frag");
        if (!attributes("alpha_blending", AttributeMapPtr()))
        {
            shaders.push_back("main.vert");
            shaders.push_back("main.frag");
        }

        osg::Program* program = Loaders::loadProgram(shaders, vars);
        stateSet->setMode(GL_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
        stateSet->setMode(GL_POINT_SPRITE, osg::StateAttribute::ON);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Point");

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _neuronMeshStateSet(
        const AttributeMap& attributes)
    {
        osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        _detailedNeuronStateDefines(attributes, vars);

        std::vector<std::string> shaders;
        _neuronCommonShaders(attributes, shaders);
        shaders.push_back("geom/basic.vert");
        /* This code now allows mixing EM shading with LODs despite the
           lower LODs do not support this shading. It'll be the user (or
           command line option parser) responsibility to provide a
           consistent style to the scene. */
        bool emShading = attributes("em_shading", false);
        if (emShading)
        {
            shaders.push_back("geom/em_neuron_mesh.vert");
            shaders.push_back("shading/electron_microscopy.frag");
        }
        else
        {
            shaders.push_back("geom/neuron_mesh.vert");
            shaders.push_back("shading/phong.frag");
            shaders.push_back("shading/phong_mesh.frag");
        }
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom[vert]");
#endif
        shaders.push_back("geom/reported_variable.vert");

        osg::Program* program = Loaders::loadProgram(shaders, vars);

        program->addBindAttribLocation("bufferOffsetsAndDelays",
                                       BUFFER_OFFSETS_AND_DELAYS_GLATTRIB);
        program->addBindAttribLocation("colorMapValueAttrib",
                                       STATIC_COLOR_MAP_VALUE_ATTRIB_NUM);

        if (emShading)
            _setNoiseTextureForSampler(stateSet);

        if (attributes("cw_winding", false))
        {
            stateSet->setAttributeAndModes(
                new osg::FrontFace(osg::FrontFace::CLOCKWISE));
        }

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("NeuronMesh");

        stateSet->setAttributeAndModes(new osg::CullFace());

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _tubeletStateSet(const AttributeMap& attributes)

    {
        osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        _detailedNeuronStateDefines(attributes, vars);

        vars["DEFINES"] += "#define OVERRIDE_DEPTH\n";
        std::vector<std::string> shaders;
        _neuronCommonShaders(attributes, shaders);
        shaders.push_back("geom/tubelet.vert");
        shaders.push_back("geom/tubelet.geom");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom");
#endif
        shaders.push_back("geom/reported_variable.vert");
        shaders.push_back("shading/phong_tubelet.frag");
        shaders.push_back("shading/phong.frag");

        osg::Program* program = Loaders::loadProgram(shaders, vars);

        program->addBindAttribLocation("bufferOffsetsAndDelays",
                                       BUFFER_OFFSETS_AND_DELAYS_GLATTRIB);
        program->addBindAttribLocation("colorMapValueAttrib",
                                       STATIC_COLOR_MAP_VALUE_ATTRIB_NUM);

        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 6);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
        program->addBindAttribLocation("radiusAttrib",
                                       TUBELET_POINT_RADIUS_ATTRIB_NUM);
        program->addBindAttribLocation("cutPlaneAttrib",
                                       TUBELET_CUT_PLANE_ATTRIB_NUM);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Tubelet");

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _pseudoCylindersStateSet(
        const AttributeMap& attributes)
    {
        osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        _detailedNeuronStateDefines(attributes, vars);

        std::vector<std::string> shaders;
        _neuronCommonShaders(attributes, shaders);
        shaders.push_back("geom/pseudo_cylinder.vert");
        shaders.push_back("geom/pseudo_cylinder.geom");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom");
#endif
        shaders.push_back("geom/reported_variable.vert");
        shaders.push_back("shading/phong_pseudo_cylinder.frag");
        shaders.push_back("shading/phong.frag");

        osg::Program* program = Loaders::loadProgram(shaders, vars);

        program->addBindAttribLocation("bufferOffsetsAndDelays",
                                       BUFFER_OFFSETS_AND_DELAYS_GLATTRIB);
        program->addBindAttribLocation("colorMapValueAttrib",
                                       STATIC_COLOR_MAP_VALUE_ATTRIB_NUM);

        program->addBindAttribLocation("tangentAndThickness",
                                       TANGENT_AND_THICKNESS_ATTRIB_NUM);
        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 6);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Pseudo cylinder");

        return stateSet;
    }

    osg::ref_ptr<osg::StateSet> _sphericalSomaStateSet(
        const AttributeMap& attributes)
    {
        osg::StateSet* stateSet = new osg::StateSet();

        std::map<std::string, std::string> vars;
        _neuronStateDefines(attributes, vars);

        vars["DEFINES"] += "#define OVERRIDE_DEPTH\n";

        if (attributes("show_soma_spikes", false))
            vars["DEFINES"] += "#define SHOW_SPIKES\n";

        std::vector<std::string> shaders;
        shaders.push_back("geom/sphere.vert");
        shaders.push_back("geom/sphere.geom");
#ifdef OSG_GL3_AVAILABLE
        shaders.push_back("geom/compute_clip_distances.geom");
#endif
        shaders.push_back("geom/reported_variable_at_soma.vert");
        shaders.push_back("geom/simulation_value_to_soma_color.vert");
        shaders.push_back("geom/soma_color.vert");
        shaders.push_back("shading/phong_sphere.frag");
        shaders.push_back("shading/phong.frag");

        /* If no alpha blending render bin is used, the program needs a main
           for the vertex and fragment shaders */
        AttributeMapPtr alphaBlending =
            attributes("alpha_blending", AttributeMapPtr());
        if (!alphaBlending || alphaBlending->empty())
        {
            shaders.push_back("main.vert");
            shaders.push_back("main.frag");
        }

        osg::Program* program = Loaders::loadProgram(shaders, vars);
        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

        program->addBindAttribLocation("cellIndex", CELL_INDEX_GLATTRIB);
        program->addBindAttribLocation("bufferOffsetsAndDelays",
                                       BUFFER_OFFSETS_AND_DELAYS_GLATTRIB);
        program->addBindAttribLocation("highlighted",
                                       HIGHLIGHTED_SOMA_ATTRIB_NUM);
        program->addBindAttribLocation("compartmentColorMapIndex",
                                       COMPARTMENT_COLOR_MAP_INDEX_ATTRIB_NUM);
        program->addBindAttribLocation("spikeColorMapIndex",
                                       SPIKE_COLOR_MAP_INDEX_ATTRIB_NUM);

        _addProgramToStateOrEnableAlphaBlending(stateSet, program);
        stateSet->setName("Soma sphere");

        if (_colorMaps)
            _colorMaps->addStateAttributes(stateSet);

        return stateSet;
    }

    void _neuronStateDefines(const AttributeMap& attributes,
                             std::map<std::string, std::string>& vars)
    {
        if (attributes("inflatable_neurons", false))
            vars["DEFINES"] += "#define INFLATABLE\n";
        if (attributes("accurate_headlight", false))
            vars["DEFINES"] += "#define ACCURATE_HEADLIGHT\n";

        AttributeMapPtr alphaBlending =
            attributes("alpha_blending", AttributeMapPtr());
        if (alphaBlending && !alphaBlending->empty())
            vars["DEFINES"] += "#define USE_ALPHA_BLENDING\n";
    }

    void _detailedNeuronStateDefines(const AttributeMap& attributes,
                                     std::map<std::string, std::string>& vars)
    {
        _neuronStateDefines(attributes, vars);

        if (attributes("smooth_tubelets", false))
            vars["DEFINES"] += "#define SMOOTH_TUBELETS\n";
        if (attributes("faster_pseudocylinder_normals", false))
            vars["DEFINES"] += "#define FASTER_PSEUDOCYLINDER_NORMALS\n";

        /* This attribute depends on the scene attributes and is passed as
           an extra attribute to getOrCreateStateSet from
           CylinderBasedModel::getOrCreateModelStateSet. */
        if (attributes("clod", false))
            vars["DEFINES"] += "#define USE_CONTINUOUS_LOD\n";

        if (attributes("show_spikes", false))
            vars["DEFINES"] += "#define SHOW_SPIKES\n";
    }

    void _neuronCommonShaders(const AttributeMap& attributes,
                              std::vector<std::string>& shaders)
    {
        shaders.push_back("shading/membrane_color.frag");
        shaders.push_back("shading/simulation_value_to_color.frag");
        shaders.push_back("geom/membrane_color.vert");

        /* If no alpha blending render bin is used, the program needs a main
           for the vertex and fragment shaders */
        AttributeMapPtr alphaBlending =
            attributes("alpha_blending", AttributeMapPtr());
        if (!alphaBlending || alphaBlending->empty())
        {
            shaders.push_back("main.vert");
            shaders.push_back("main.frag");
        }
    }

    void _addProgramToStateOrEnableAlphaBlending(osg::StateSet* stateSet,
                                                 osg::Program* program)
    {
        osgTransparency::BaseRenderBin* renderBin =
            _renderBinManager.getAlphaBlendedRenderBin();
        if (renderBin)
        {
            renderBin->addExtraShadersForState(stateSet, program);
            stateSet->setRenderBinDetails(ALPHA_BLENDED_RENDER_ORDER,
                                          renderBin->getName());
        }
        else
            stateSet->setAttributeAndModes(program);
    }
};

/*
  SceneStyle
*/
SceneStyle::SceneStyle(const AttributeMap& attributes)
    : _impl(new Impl(attributes))
{
}

SceneStyle::~SceneStyle()
{
}

const AttributeMap& SceneStyle::getAttributes() const
{
    return _impl->getAttributes();
}

void SceneStyle::processModel(osg::Node* model)
{
    /* Applying a basic program if the model doesn't already have one.
       This is useful to get the lighting correct in head tracked
       projection displays. */
    osg::ref_ptr<FindPrograms> finder(new FindPrograms());
    model->accept(*finder);
    if (!finder->hasProgram)
        model->setStateSet(getStateSet(MESH));

    /* The fixer visitor removes the programs in those state sets
       that target the alpha blended render bin. Watch out and
       don't move this traversal before the previous one */
    osgTransparency::BaseRenderBin* renderBin =
        _impl->_renderBinManager.getAlphaBlendedRenderBin();
    if (renderBin)
    {
        osg::ref_ptr<FixTransparentObjects> fixer(
            new FixTransparentObjects(renderBin));
        model->accept(*fixer);
    }
}

osg::StateSet* SceneStyle::getStateSet(const StateType type) const
{
    return _impl->getOrCreateStateSet(type);
}

osg::StateSet* SceneStyle::getStateSet(
    const StateType type, const rtneuron::AttributeMap& extra) const
{
    return _impl->getOrCreateStateSet(type, extra);
}

void SceneStyle::setColorMapAtlas(const ColorMapAtlasPtr& colorMaps)
{
    return _impl->setColorMapAtlas(colorMaps);
}

RenderBinManager& SceneStyle::getRenderBinManager()
{
    return _impl->_renderBinManager;
}
}
}
}
