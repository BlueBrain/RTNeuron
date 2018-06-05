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

#ifndef RTNEURON_LOADERS_H
#define RTNEURON_LOADERS_H

#include <osg/Drawable>

#include <boost/filesystem/fstream.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#include <string>

namespace bbp
{
namespace rtneuron
{
namespace core
{
namespace Loaders
{
/* Namespace functions */
/**
   Creates a GLSL program using the shader sources indicated.
   If the sourceFiles vector is emtpy, then returns an empty program.

   The shader types are inferred from the source file names.
   If the file name has one of the extensions .vert, .frag or .geom, the
   corresponding shader type is created.
   For files with other extensions, the caller can provide the shader type
   appending "[vert]", "[frag]" or "[geom]" at the end of the string. The
   type hint is removed from the name before loading the file.
 */
osg::Program* loadProgram(const std::vector<std::string>& sourceFiles,
                          const std::map<std::string, std::string>& vars =
                              std::map<std::string, std::string>());

/**
   Adds shader sources to a GLSL program using the file names indicated.

   The shader types are inferred from the source file names.
   @sa loadProgram
 */
void addShaders(osg::Program* program,
                const std::vector<std::string>& sourceFiles,
                const std::map<std::string, std::string>& vars =
                    std::map<std::string, std::string>());

/**
   Loads a shader source file and replaces $varname words with the
   contents of the table inside the cell with the same name (but no dollar
   symbol).
   If any error is detected an empty string is returned.
*/
std::string readSourceAndReplaceVariables(
    const boost::filesystem::path& shaderFile,
    const std::map<std::string, std::string>& vars,
    const std::vector<boost::filesystem::path>& extraPaths =
        std::vector<boost::filesystem::path>());
}
}
}
}
#endif
