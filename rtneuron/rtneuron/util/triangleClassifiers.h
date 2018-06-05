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

#ifndef RTNEURON_TRIANGLE_CLASSIFIERS_H
#define RTNEURON_TRIANGLE_CLASSIFIERS_H

#include <cstdint>

namespace bbp
{
namespace rtneuron
{
namespace core
{
/* This file provides two versions of the triangle to section mapping, the
   new and the old one.
   The pointer to function is the function that should be used globally. */

/* Return the corner that determines the mapping of a triangle */
unsigned int classifyTriangle(const uint16_t* sections, const float* positions,
                              const uint32_t triangle[3]);
unsigned int classifyTriangleOld(const uint16_t* sections,
                                 const float* positions,
                                 const uint32_t triangle[3]);

typedef unsigned int (*TriangleClassifier)(const uint16_t*, const float*,
                                           const uint32_t[3]);

extern TriangleClassifier triangleClassifier;
}
}
}
#endif
