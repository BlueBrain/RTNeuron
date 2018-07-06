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
    opacity: 0.8
    radius: 10
    border.width: 0

    width: 500
    height: 100

    anchors.centerIn: parent? parent : undefined

    property real fontSize: 12

    function step(message, progress)
    {
        progressBar.value = progress
        messageText.text = message
    }

    Column
    {
        anchors.topMargin: 20
        anchors.leftMargin: 15
        anchors.rightMargin: 15
        anchors.fill: parent
        spacing: 5

        Text
        {
           id: messageText
           font.pixelSize: fontSize
           text: ""
           height: 25
        }

        Rectangle
        {
             border.width: 2
             border.color: "#7f7f7f"
             color: "#00000000"
             anchors.left: parent.left
             anchors.right: parent.right
             height: 25

             Rectangle
             {
                 property real value: 0

                 id: progressBar
                 anchors.left: parent.left
                 height: 25
                 width: parent.width * value
                 color: "#b7c7df"
             }
        }
    }
}

