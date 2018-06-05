/* Copyright (c) 2006-2018, École Polytechnique Fédérale de Lausanne (EPFL) /
 *                           Blue Brain Project and
 *                          Universidad Politécnica de Madrid (UPM)
 *                          Jafet Villafrance <jafet.villafrancadiaz@epfl.ch>
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
#include "DepthOfField.h"
#include "util/extensions.h"

#include <osg/FrameBufferObject>
#include <osg/Geode>
#include <osg/Geometry>

#include <osgViewer/Renderer>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const char* const blurFragSource = R"(
uniform sampler2D inputTex;
uniform vec2 blurDir;
uniform vec2 resolution;)"
#ifdef OSG_GL3_AVAILABLE
                                   "    in vec2 uv;"
#else
                                   "    varying vec2 uv;"
#endif
                                   R"(
void main(void)
{
    vec4 color = vec4(0.0);
    vec2 off1 = vec2(1) * blurDir / resolution;
    vec2 off2 = vec2(2) * blurDir / resolution;
    vec2 off3 = vec2(3) * blurDir / resolution;
    vec2 off4 = vec2(4) * blurDir / resolution;
    vec2 off5 = vec2(5) * blurDir / resolution;
    color += texture2D(inputTex, uv) * 0.159576912161;
    color += texture2D(inputTex, uv + off1) * 0.147308056121;
    color += texture2D(inputTex, uv - off1) * 0.147308056121;
    color += texture2D(inputTex, uv + off2) * 0.115876621105;
    color += texture2D(inputTex, uv - off2) * 0.115876621105;
    color += texture2D(inputTex, uv + off3) * 0.0776744219933;
    color += texture2D(inputTex, uv - off3) * 0.0776744219933;
    color += texture2D(inputTex, uv + off4) * 0.0443683338718;
    color += texture2D(inputTex, uv - off4) * 0.0443683338718;
    color += texture2D(inputTex, uv + off5) * 0.0215963866053;
    color += texture2D(inputTex, uv - off5) * 0.0215963866053;

    gl_FragColor = color;
})";

#ifdef OSG_GL3_AVAILABLE
const char* const bypassVertSource = R"(
#version 410
in vec4 osg_Vertex;
in vec4 osg_MultiTexCoord0;
out vec2 uv;
void main()
{
    gl_Position = osg_Vertex;
    uv = osg_MultiTexCoord0.st;
}
)";
#else
const char* const bypassVertSource = R"(
varying vec2 uv;
void main()
{
    gl_Position = gl_Vertex;
    uv = gl_MultiTexCoord0.st;
}
)";
#endif

osg::Geode* _createScreenQuad()
{
    osg::Geode* geode = new osg::Geode;
    geode->addDrawable(osg::createTexturedQuadGeometry(osg::Vec3(-1, -1, 0),
                                                       osg::Vec3(2, 0, 0),
                                                       osg::Vec3(0, 2, 0), 0, 0,
                                                       1, 1));
    return geode;
}

osg::ref_ptr<osg::Camera> _createRTTCamera(osg::Camera::BufferComponent buffer,
                                           osg::Texture* tex)
{
    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);

    tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
    tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
    camera->setViewport(0, 0, tex->getTextureWidth(), tex->getTextureHeight());
    camera->attach(buffer, tex);

    camera->addChild(_createScreenQuad());
    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    return camera;
}

RTTPair _createBlurPass(osg::Texture* inputTex, const osg::Vec2& dir)
{
    const auto width = inputTex->getTextureWidth();
    const auto height = inputTex->getTextureHeight();
    osg::ref_ptr<osg::Texture2D> tex2D = new osg::Texture2D;
    tex2D->setTextureSize(width, height);
    tex2D->setInternalFormat(GL_RGBA);
    auto camera = _createRTTCamera(osg::Camera::COLOR_BUFFER, tex2D.get());

    osg::ref_ptr<osg::Program> blurProg = new osg::Program;
    blurProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, blurFragSource));
    blurProg->addShader(new osg::Shader(osg::Shader::VERTEX, bypassVertSource));

    auto* stateSet = camera->getOrCreateStateSet();
    stateSet->setTextureAttribute(0, inputTex);
    stateSet->setAttributeAndModes(blurProg.get(), osg::StateAttribute::ON);
    stateSet->addUniform(new osg::Uniform("inputTex", 0));
    stateSet->addUniform(new osg::Uniform("blurDir", dir));
    stateSet->addUniform(
        new osg::Uniform("resolution", osg::Vec2(width, height)));

    return {camera.release(), tex2D.get()};
}

class PreDrawCallback : public osg::Camera::DrawCallback
{
public:
    PreDrawCallback() { glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &_fbo); }
    virtual void operator()(osg::RenderInfo& renderInfo) const
    {
        osg::State& state = *renderInfo.getState();
        const auto* ext = getFBOExtensions(state.getContextID());
        ext->glBindFramebuffer(GL_FRAMEBUFFER_EXT, _fbo);
    }

    GLint _fbo;
};
}

DepthOfField::DepthOfField(osg::Camera* camera)
    : _width(camera->getViewport()->width())
    , _height(camera->getViewport()->height())
    , _screenQuad(_createScreenQuad())
    , _orthoCamera(new osg::Camera)
    , _originalRenderTargetImpl(camera->getRenderTargetImplementation())
    , _camera(camera)
{
    /* Creating the depth and color textures were the original camera will
       render into. */
    _textures[0] = new osg::Texture2D;
    _textures[0]->setTextureSize(_width, _height);
    _textures[0]->setInternalFormat(GL_DEPTH_COMPONENT24);
    _textures[0]->setSourceFormat(GL_DEPTH_COMPONENT);
    _textures[0]->setSourceType(GL_FLOAT);
    _textures[0]->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
    _textures[0]->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);

    _textures[1] = new osg::Texture2D;
    _textures[1]->setTextureSize(_width, _height);
    _textures[1]->setInternalFormat(GL_RGBA);

    /* Creating the cameras with the render passes with the horizontal and
       vertical blur. */
    _pairs[0] = _createBlurPass(_textures[1].get(), osg::Vec2(1.0f, 0.0f));
    _pairs[1] = _createBlurPass(_pairs[0].second, osg::Vec2(0.0f, 1.0f));
    _textures[2] = _pairs[1].second;

    /* Creating the screen quad for the final render pass. */
    osg::StateSet* stateset = _screenQuad->getOrCreateStateSet();
    stateset->addUniform(new osg::Uniform("focalDistance", 0.f));
    stateset->addUniform(new osg::Uniform("focalRange", 0.f));

    const osg::Matrixf& projection = camera->getProjectionMatrix();
    stateset->addUniform(new osg::Uniform("projectionMatrix", projection));

    stateset->setTextureAttribute(0, _textures[0]);
    stateset->addUniform(new osg::Uniform("depth", 0));
    stateset->setTextureAttribute(1, _textures[1]);
    stateset->addUniform(new osg::Uniform("color", 1));
    stateset->setTextureAttribute(2, _textures[2]);
    stateset->addUniform(new osg::Uniform("blur", 2));

    osg::Program* program =
        core::Loaders::loadProgram({"shading/depth_field.frag"}, {});
    program->addShader(new osg::Shader(osg::Shader::VERTEX, bypassVertSource));
    stateset->setAttributeAndModes(program, osg::StateAttribute::ON);

    /* Set orthographic camera to render the quad */
    _orthoCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    _orthoCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
    _orthoCamera->setRenderOrder(osg::Camera::POST_RENDER);
    _orthoCamera->setInitialDrawCallback(new PreDrawCallback());
    _orthoCamera->addChild(_screenQuad);

    /* Setting up the camera original camera to do render to texture. */
    _camera->attach(osg::Camera::DEPTH_BUFFER, _textures[0].get());
    _camera->attach(osg::Camera::COLOR_BUFFER, _textures[1].get());

    _camera->setViewport(0, 0, _width, _height);
    _camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    _camera->setRenderOrder(osg::Camera::PRE_RENDER);

    /* Adding the depth of field cameras the main camera. */
    _camera->addChild(_pairs[0].first);
    _camera->addChild(_pairs[1].first);
    _camera->addChild(_orthoCamera.get());
}

DepthOfField::~DepthOfField()
{
    _camera->setRenderTargetImplementation(_originalRenderTargetImpl);
    _camera->removeChild(_pairs[0].first);
    _camera->removeChild(_pairs[1].first);
    _camera->removeChild(_orthoCamera.get());
}

void DepthOfField::setFocalDistance(const float focalDistance)
{
    _screenQuad->getStateSet()->getUniform("focalDistance")->set(focalDistance);
}

void DepthOfField::setFocalRange(const float focalRange)
{
    _screenQuad->getStateSet()->getUniform("focalRange")->set(focalRange);
}

void DepthOfField::update()
{
    const int texWidth = _camera->getViewport()->width();
    const int texHeight = _camera->getViewport()->height();

    osgUtil::CullVisitor* cullVisitor =
        static_cast<osgViewer::Renderer*>(_camera->getRenderer())
            ->getSceneView(0)
            ->getCullVisitor();
    auto zNear = cullVisitor->getCalculatedNearPlane();
    auto zFar = cullVisitor->getCalculatedFarPlane();
    cullVisitor->clampProjectionMatrix(_camera->getProjectionMatrix(), zNear,
                                       zFar);

    const osg::Matrixf& projection = _camera->getProjectionMatrix();
    _screenQuad->getStateSet()->getUniform("projectionMatrix")->set(projection);

    if (texWidth == _width && texHeight == _height)
        return;

    _width = texWidth;
    _height = texHeight;

    _textures[0]->setTextureSize(_width, _height);
    _textures[0]->dirtyTextureObject();
    _textures[1]->setTextureSize(_width, _height);
    _textures[1]->dirtyTextureObject();
    _textures[2]->setTextureSize(_width, _height);
    _textures[2]->dirtyTextureObject();

    _pairs[0].first->setViewport(0, 0, _width, _height);
    _pairs[0].second->setTextureSize(_width, _height);
    _pairs[0].second->dirtyTextureObject();
    _pairs[1].first->setViewport(0, 0, _width, _height);
    _pairs[1].second->setTextureSize(_width, _height);
    _pairs[1].second->dirtyTextureObject();
    _camera->setViewport(0, 0, _width, _height);

    // Needed to refresh the textures attached to the FBO after resizing
    osgViewer::Renderer* renderer =
        static_cast<osgViewer::Renderer*>(_camera->getRenderer());
    renderer->getSceneView(0)->getRenderStage()->setCameraRequiresSetUp(true);
    _pairs[0].first->setRenderingCache(0);
    _pairs[1].first->setRenderingCache(0);
}
}
}
}
