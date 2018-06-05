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

#ifndef RTNEURON_CONSTANTS_H
#define RTNEURON_CONSTANTS_H

namespace bbp
{
namespace rtneuron
{
namespace core
{
/*
  Render orders
*/
enum RenderOrder
{
    DEFAULT_RENDER_ORDER = 0,
    SILHOUETTES_RENDER_ORDER = 1,
    HIGHLIGHTED_RENDER_ORDER = 2,
    ALPHA_BLENDED_RENDER_ORDER = 4,
    TEXT_LABELS_RENDER_ORDER = 5
};

/*
   Node traversal masks
*/
enum NodeMask
{
    CULL_AND_DRAW_MASK = 1,
    EVENT_MASK = 2,
};

/*
  Shader attribute and texture numbers
*/

/* Pseudo-cylinder shaders */
const int TANGENT_AND_THICKNESS_ATTRIB_NUM = 4;
/* Tubelet shaders */
const int TUBELET_POINT_RADIUS_ATTRIB_NUM = 1;
const int TUBELET_CUT_PLANE_ATTRIB_NUM = 4;
const int FIRST_TUBELET_CORNER_ATTRIB_NUM = 4;
/* Soma shaders */
const int HIGHLIGHTED_SOMA_ATTRIB_NUM = 4;
const int COMPARTMENT_COLOR_MAP_INDEX_ATTRIB_NUM = 5;
const int SPIKE_COLOR_MAP_INDEX_ATTRIB_NUM = 6;
/* Common */
const int STATIC_COLOR_MAP_VALUE_ATTRIB_NUM = 5; // except for somas

const int BUFFER_OFFSETS_AND_DELAYS_GLATTRIB = 8;
const int CELL_INDEX_GLATTRIB = 9; /* Used for soma spike visualization */
const unsigned int MAX_VERTEX_ATTRIBUTE_NUM = 9;

/* Texture units */
const int STATIC_COLOR_MAP_TEXTURE_NUMBER = 0;
const char* const STATIC_COLOR_MAP_UNIFORM_PREFIX = "static";
const int NOISE_TEXTURE_NUMBER = 1;
const int COMPARTMENT_DATA_TEXTURE_NUMBER = 2;
const int SPIKE_DATA_TEXTURE_NUMBER = 3;
const int COMPARTMENT_COLOR_MAP_TEXTURE_NUMBER = 4;
const char* const COMPARTMENT_COLOR_MAP_UNIFORM_PREFIX = "simulation";
const int SPIKE_COLOR_MAP_TEXTURE_NUMBER = 5;
const char* const SPIKE_COLOR_MAP_UNIFORM_PREFIX = "spike";
const int COLOR_MAP_ATLAS_TEXTURE_NUMBER = 6;
const int MAX_TEXTURE_UNITS = 7;

/*
  Misc
*/
const float DEFAULT_SOMA_RADIUS = 9;
const float SOMA_MEAN_TO_RENDER_RADIUS_RATIO = 0.85f;

const float REPORTED_VARIABLE_DEFAULT_MIN_VALUE = -85.0f;
const float REPORTED_VARIABLE_DEFAULT_MAX_VALUE = 20.0f;
const float REPORTED_VARIABLE_DEFAULT_THRESHOLD_VALUE = 40.0f;
const char* const REPORTED_VARIABLE_DEFAULT_NAME = "Voltage";

const float DEFAULT_LOD_BIAS = 0.5f;

const int MAX_SUPPORTED_CONTEXTS = 18;

const char* const DEFAULT_FRAME_FILE_NAME_PREFIX = "frame";
const char* const DEFAULT_FRAME_FILE_FORMAT = "png";
}
}
}
#endif
