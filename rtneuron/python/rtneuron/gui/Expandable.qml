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

Rectangle
{
    id: expandable
    radius: 10
    smooth: true
    height: header.height + 10
    clip: true

    property string headerText: qsTr("expand/collapse")
    property string collapsedHeaderText: headerText
    property string expandedHeaderText: headerText
    property real expandedSize: childrenRect.height + 10
    property alias header: header
    property bool expandsDown: true

    function toggle()
    {
        if (state == "EXPANDED")
           state = "COLLAPSED"
        else
           state = "EXPANDED"
    }

    function expand()
    {
        state = "EXPANDED"
    }

    function collapse()
    {
        state = "COLLAPSED"
    }

    states: [
        State
        {
            name: "EXPANDED"
            PropertyChanges
            {
                target: expandable
                height: expandable.expandedSize
            }
            PropertyChanges
            {
                target: header
                // 25BC is Unicode triangle up and 25B2 is Unicode triangle down
                text: (expandsDown ? "\u25B2 " : "\u25BC ") +
                       expandable.expandedHeaderText
            }
        },
        State
        {
            name: "COLLAPSED"
            PropertyChanges
            {
                target: expandable
                height: header.height + 10

            }
            PropertyChanges
            {
                target: header
                text: (expandsDown ? "\u25BC " : "\u25B2 ") +
                       expandable.collapsedHeaderText
            }
        }
    ]
    transitions: [
        Transition
        {
            from: "*"; to: "*"
            NumberAnimation
            {
                properties: "height"
                easing.type: Easing.OutCubic
                duration: 200
            }
        }
    ]

    Text
    {
        id: header
        opacity: 0.8

        MouseArea
        {
            anchors.fill: parent
            onClicked: toggle()
        }

        anchors.margins: 10
        anchors.topMargin: 5
        anchors.bottomMargin: 5
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        text: (expandsDown ? "\u25BC " : "\u25B2 ") + parent.collapsedHeaderText
    }
}
