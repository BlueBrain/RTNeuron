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
import QtQuick 2.0

Rectangle
{
    id: rect
    width: 150
    height: 35
    border.width: 0
    opacity: checked ? baseOpacity : baseOpacity / 2.0
    radius: 6
    smooth: true
    property bool checked: false
    property bool checkable: false
    property real baseOpacity: 0.5

    property alias textColor: label.color
    property alias text: label.text
    property alias fontSize: label.font.pixelSize

    signal clicked
    signal toggled(bool checked)

    Text
    {
        id: label
        x: 0
        y: 0
        width: parent.width
        height: parent.height
        verticalAlignment: Text.AlignVCenter
        anchors.verticalCenter: parent.verticalCenter
        horizontalAlignment: Text.AlignHCenter
        font.pixelSize: 12
        font.bold: checked
        color: "black"
    }

    MouseArea
    {
        id: area
        width: parent.width
        height: parent.height

        onPressed:
        {
            if (!checkable)
               checked = !checked
        }

        onReleased:
        {
            if (!checkable)
               checked = !checked
        }

        onClicked:
        {
            if (checkable)
            {
                checked = !checked
                toggled(checked)
            }
            rect.clicked()
        }
    }
}
