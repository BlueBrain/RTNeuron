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

#include "triangleClassifiers.h"

namespace bbp
{
namespace rtneuron
{
namespace core
{
TriangleClassifier triangleClassifier = classifyTriangle;

unsigned int classifyTriangle(const uint16_t* sections, const float* positions,
                              const uint32_t triangle[3])
{
    const uint32_t* t = triangle;
    const float p[3] = {positions[t[0]], positions[t[1]], positions[t[2]]};
    const float s[3] = {float(sections[t[0]]), float(sections[t[1]]),
                        float(sections[t[2]])};
    unsigned int corner = 0;
    if (p[0] > p[1] || (p[0] == p[1] && s[0] > s[1]))
        corner = 1;
    if (p[corner] > p[2] || (p[corner] == p[2] && s[corner] > s[2]))
        corner = 2;
    return corner;
}

unsigned int classifyTriangleOld(const uint16_t*, const float* positions,
                                 const uint32_t triangle[3])
{
    unsigned int corner = 0;
    const uint32_t* t = triangle;
    const float p[3] = {positions[t[0]], positions[t[1]], positions[t[2]]};
    if (p[0] > p[1] || (p[0] == p[1] && t[0] > t[1]))
        corner = 1;
    if (p[corner] > p[2] || (p[corner] == p[2] && t[corner] > t[2]))
        corner = 2;
    return corner;
}
}
}
}
