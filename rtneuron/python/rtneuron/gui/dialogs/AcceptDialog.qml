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

import QtQuick.Controls 1.0

ModalDialog
{

    id: acceptDialog

    signal done(bool accepted)

    function setMessage(text)
    {
        message.text = text
    }

    function setPrefix(text)
    {
        message.text = text
    }

    function setCancelButtonLabel(text)
    {
        cancel.visible = text != ""
        cancel.text = text
    }

    function setAcceptButtonLabel(text)
    {
        accept.visible = text != ""
        accept.text = text
    }

    Rectangle
    {
        opacity: 0.8
        radius: 10
        border.width: 0

        width: childrenRect.width + 30
        height: childrenRect.height + 30

        anchors.centerIn: parent

        property real fontSize: 12

        Column
        {
            // centerIn causes a warning about a binding loop
            x: 15
            y: 15
            spacing: 10

            Text
            {
                id: message
                width: 300
                opacity: 0.8
                wrapMode: Text.WordWrap

            }

            Row
            {
                spacing: 10
                anchors.horizontalCenter: parent.horizontalCenter

                Button
                {
                    id: cancel
                    text: qsTr("Cancel")
                    width: 100
                    height: 35
                    checked: false
                    onClicked: done(false)
                }

                Button
                {
                    id: accept
                    text: qsTr("Accept")
                    width: 100
                    height: 35
                    checked: false
                    onClicked: done(true)
                }
            }
        }
    }
}
