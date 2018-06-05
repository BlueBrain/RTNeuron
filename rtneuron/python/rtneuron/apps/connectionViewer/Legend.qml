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

import QtQuick 2.4
import QtQuick.Controls 1.3

import "../../gui"

Expandable
{
    id: legend

    function setModel(model)
    {
        list.model = model
        list.itemClicked.connect(model.toggleItem)
        adjustExpandedSize()
        // We don't expand from onVisibleChanged because we want to expand
        // only once the final height is set. Otherwise there's a visual
        // glitch,
        state = "EXPANDED"
    }

    onVisibleChanged:
    {
        if (!visible)
            state = "COLLAPSED"
    }

    onParentChanged:
    {
        parent.onHeightChanged.connect(adjustExpandedSize)
    }

    function adjustExpandedSize()
    {
        if (!list.model || !parent)
           return
        var height = list.model.rows * 20
        var maxScrollHeight = parent.height - header.height - 20
        if (height > maxScrollHeight)
        {
            height = maxScrollHeight
            legend.width = 240
        }
        else
        {
            legend.width = 220
        }
        expandedSize = height + header.height + 20
    }


    // We use transparency in the rectangle color instead of opacity at
    // the item level because we want the rectangle with the color to
    // be opaque.
    color: "#ccffffff"
    header.opacity: 0.64
    width: 220
    headerText: "Legend"
    expandsDown: true

    MouseArea
    {
        // To not propagate wheel events to the parent
        // Declared before scroll view so it doesn't sit on top.
        anchors.top: parent.header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 10
        anchors.topMargin: 5

        onWheel:
        {
            wheel.accepted = true
        }

        ScrollView
        {
            id: scroll

            anchors.fill: parent

            ListView
            {
                id: list

                signal itemClicked(int index)

                clip: true

                delegate: Item
                {
                    height: childrenRect.height
                    width: childrenRect.width
                    Row
                    {
                        Text
                        {
                            opacity: 0.64
                            text: name
                            width: 160
                            height: 20
                        }
                        Item
                        {
                            width: 40
                            height: 20

                            Rectangle
                            {
                                x: selected ? -20 : 0
                                id: rectangle
                                width: parent.width
                                height: parent.height
                                color: value
                                opacity: 1
                            }
                        }
                    }
                    MouseArea
                    {
                        anchors.fill: parent
                        onClicked:
                        {
                            list.itemClicked(index)
                        }
                    }
                }
            }
        }
    }
}
