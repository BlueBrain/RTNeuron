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

#ifndef RTNEURON_VIEWPORTTILE_H
#define RTNEURON_VIEWPORTTILE_H

namespace bbp
{
namespace rtneuron
{
namespace core
{
/**
   User data class attached to a Viewport used in tiled alpha-blending and
   culling to know the tile being rendered.
*/
struct ViewportTile : public osg::Referenced
{
    ViewportTile(const unsigned row_ = 0, const unsigned column_ = 0,
                 const unsigned rows_ = 1, const unsigned columns_ = 1)
        : row(row_)
        , column(column_)
        , rows(rows_)
        , columns(columns_)
    {
    }

    /** Returns the tile number for this tile.
        Tile 0 is reserved for the whole screen */
    unsigned int index() const { return row * columns + column + 1; }
    const unsigned row;
    const unsigned column;
    const unsigned rows;
    const unsigned columns;
};
}
}
}
#endif
