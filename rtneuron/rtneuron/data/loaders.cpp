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

#include "data/loaders.h"
#include "config/paths.h"

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>
#include <cstring>
#include <sstream>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace
{
const char* _getShaderPath()
{
    char* path = ::getenv("RTNEURON_SHADER_PATH");
    if (path)
        return path;
    return "";
}
}

const std::string s_shaderPath(_getShaderPath());

/*
  Helper classes
*/
namespace Loaders
{
/*
  Helper functions
*/
osg::Program* loadProgram(const std::vector<std::string>& sourceFiles,
                          const std::map<std::string, std::string>& vars)
{
    osg::Program* program = new osg::Program();
    addShaders(program, sourceFiles, vars);
    return program;
}

void addShaders(osg::Program* program,
                const std::vector<std::string>& sourceFiles,
                const std::map<std::string, std::string>& vars)
{
    std::string programName = program->getName();

    for (std::vector<std::string>::const_iterator source = sourceFiles.begin();
         source != sourceFiles.end(); ++source)
    {
        programName += *source + ", ";

        osg::Shader* shader = new osg::Shader();
        shader->setName(*source);

        std::string filename = *source;

        if (source->rfind("[vert]") == source->size() - 6)
        {
            shader->setType(osg::Shader::VERTEX);
            filename = source->substr(0, source->size() - 6);
        }
        else if (source->rfind("[geom]") == source->size() - 6)
        {
            shader->setType(osg::Shader::GEOMETRY);
            filename = source->substr(0, source->size() - 6);
        }
        else if (source->rfind("[frag]") == source->size() - 6)
        {
            shader->setType(osg::Shader::FRAGMENT);
            filename = source->substr(0, source->size() - 6);
        }
        else if (source->rfind(".vert") == source->size() - 5)
            shader->setType(osg::Shader::VERTEX);
        else if (source->rfind(".geom") == source->size() - 5)
            shader->setType(osg::Shader::GEOMETRY);
        else if (source->rfind(".frag") == source->size() - 5)
            shader->setType(osg::Shader::FRAGMENT);

        const std::string text = readSourceAndReplaceVariables(filename, vars);
        shader->setShaderSource(text);

        program->addShader(shader);
    }

    program->setName(programName);
}

std::string readSourceAndReplaceVariables(
    const boost::filesystem::path& shaderFile,
    const std::map<std::string, std::string>& vars,
    const std::vector<boost::filesystem::path>& extraPaths)
{
#ifdef OSG_GL3_AVAILABLE
    std::string shaders = "shaders/GL3";
#else
    std::string shaders = "shaders/GL2";
#endif

    typedef boost::filesystem::path Path;
    typedef std::vector<Path> Paths;
    Paths paths;
    paths.resize(extraPaths.size());
    std::copy(extraPaths.begin(), extraPaths.end(), paths.begin());
    paths.push_back(Path(g_install_prefix() / shaders));
    paths.push_back(
        Path(g_source_root_path() / "rtneuron/rtneuron/render" / shaders));
    paths.push_back(Path("./" + shaders));

    if (!s_shaderPath.empty())
        paths.push_back(s_shaderPath);

    while (!paths.empty())
    {
        const Path path(paths.back() / shaderFile);
        paths.pop_back();
        try
        {
            if (!boost::filesystem::exists(path))
                continue;

            boost::filesystem::ifstream file(path);
            if (file.fail())
                continue;

            std::stringstream out;
            /* Reading until a $ is found or EOF is reached. */
            do
            {
                int c = file.get();
                if (!file.eof())
                {
                    if (file.fail())
                    {
                        std::cerr << "RTNeuron: error reading shader source '"
                                  << shaderFile << "'" << std::endl;
                        break;
                    }
                    if (c == '$')
                    {
                        /* Getting the variable name. */
                        std::string name;
                        char ch;
                        while (!file.get(ch).fail() &&
                               (isalnum(ch) || ch == '_'))
                            name += ch;
                        if (!file.fail())
                            file.unget();
                        std::map<std::string, std::string>::const_iterator
                            entry = vars.find(name);
                        /* Outputing the variable substitution if the name is
                           found in the table. */
                        if (entry != vars.end())
                            out << entry->second;
                    }
                    else
                    {
                        out << (char)c;
                    }
                }
            } while (!file.eof());

            return out.str();
        }
#if BOOST_FILESYSTEM_VERSION == 3
        catch (const boost::filesystem::filesystem_error&)
#else
        catch (const boost::filesystem::basic_filesystem_error<Path>&)
#endif
        {
            // ignore and try next location
        }
    }

    std::cerr << "RTNeuron: couldn't find shader source '" << shaderFile
              << "' in any of the default locations" << std::endl;
    return "";
}
}
}
}
}
