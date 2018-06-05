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
import QtQuick.Layouts 1.1

import "../../gui"

RowLayout
{
    id: picker
    height: 12

    property bool active: false
    property alias label: label.text
    property real fontPixelSize: 12
    property int gid: -1 // undefined

    signal activated()
    signal cellEntered(int gid)

    function setGID(gid_)
    {
        gid = gid_
        if (gid_ == -1)
            input.resetText()
        else
        {
            input.text = gid_.toString()
            input.font.italic = false
        }
    }

    onActiveChanged:
    {
        button.visible = !active
        input.resetText()
        if (active)
            activated()
    }

    Text
    {
        opacity: 0.8
        id: label
        font.pixelSize: picker.fontPixelSize

        Layout.preferredWidth: 110
        text: "Cell GID:"
    }

    TextInput
    {
        id: input

        opacity: 0.8
        Layout.fillWidth: true
        text: picker.active? "enter/pick from scene" : "enter"
        font.italic: true
        font.pixelSize: picker.fontPixelSize
        validator: IntValidator{ bottom: 1 }

        function resetText()
        {
            if (active)
            {
                input.font.italic = true
                input.text = "enter/pick from scene"
            }
            else if (gid == -1)
            {
                input.font.italic = true
                input.text = "enter"
            }
            else
            {
                input.font.italic = false
                input.text = picker.gid.toString()
            }
        }

        onActiveFocusChanged:
        {
            if (activeFocus)
            {
                cursorVisible = true
                // On first selection clear the text
                if (font.italic == true)
                    text = ""
                font.italic = false
            }
            else
                resetText()
        }
        onAccepted:
        {
             picker.gid = parseInt(text)
             cellEntered(gid)
             picker.parent.focus = true
             cursorVisible = false
        }

        Keys.onEscapePressed:
        {
            resetText()
            picker.parent.focus = true
            cursorVisible = false
        }
    }

    Button
    {
        id: button
        height: 12
        Layout.preferredWidth: 100
        checkable: false
        checked: true
        text: "Press to pick"
        onClicked:
        {
            picker.active = true
        }
    }
}
