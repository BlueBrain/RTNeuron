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
import QtQuick.Layouts 1.1

import "../gui"
import "connectionViewer"

BaseOverlay
{
    id: overlay

    signal activatePicking(string cell_role) // The string will be pre or post
    signal cellEntered(string cell_role, int gid)
    signal clipNeurons(bool enable)
    signal connectedCellsVisible(bool enable)
    signal coloringChanged(string mode)
    signal expandAnnotationsClicked()
    signal inflationFactorChanged(real factor)
    signal resetClicked()
    signal synapseRadiusChanged(real factor)

    function getLegend()
    {
        return legend
    }

    function setConnectionInfoModel(model)
    {
        connectionInfoList.model = model
    }

    function enableClipButton(enable)
    {
        clipButton.enabled = enable
        clipButton.checked = enable
    }

    function enableShowHideButton(enable)
    {
        showHideButton.enabled = enable
        showHideButton.checked = enable
    }

    function activePicking(cell_role)
    {
        presynapticSelector.active = cell_role == "pre"
        postsynapticSelector.active = cell_role == "post"
    }

    function setCellRoleGID(cell_role, gid)
    {
        if (cell_role == "pre")
            presynapticSelector.setGID(gid)
        else
            postsynapticSelector.setGID(gid)
    }

    function reset()
    {
        presynapticSelector.setGID(-1)
        postsynapticSelector.setGID(-1)
        presynapticSelector.active = false
        postsynapticSelector.active = false
    }

    Rectangle
    {
        id: menu
        color: "#cfcfcf"
        opacity: 1.0
        radius: 10

        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: childrenRect.width + 20
        height: childrenRect.height + 10

        ColumnLayout
        {
            // centerIn causes a warning about a binding loop
            x: 10
            y: 5
            spacing: 5
            width: 330

            CellPicker
            {
                id: presynapticSelector
                opacity: 0.8
                label: "Presynaptic GID:"
                onActivated:
                {
                    postsynapticSelector.active = false
                    activatePicking("pre")
                }
                onCellEntered: overlay.cellEntered("pre", gid)
            }

            CellPicker
            {
                id: postsynapticSelector
                opacity: 0.8
                label: "Postsynaptic GID:"
                active: true
                onActivated:
                {
                    presynapticSelector.active = false
                    activatePicking("post")
                }
                onCellEntered: overlay.cellEntered("post", gid)
            }

            Item { height: 5 } // Separator

            RowLayout
            {
                spacing: 10
                Button
                {
                    id: showHideButton
                    height: 20
                    Layout.preferredWidth: 100
                    checkable: false
                    checked: true
                    property bool state: false
                    text: state ? "Show connected" : "Hide connected"
                    onClicked:
                    {
                        connectedCellsVisible(state)
                        state = !state
                    }
                }
                Slider
                {
                    id: synapseRadius
                    label: "Synapse radius"
                    Layout.fillWidth: true
                    height: 20
                    minimum: 0.5
                    maximum: 10
                    value: 2.5
                    textColor: "black"

                    onValueChanged: synapseRadiusChanged(value)
                }
            }

            RowLayout
            {
                spacing: 10
                Text
                {
                    opacity: 0.8
                    id: colorLabel
                    text: "Coloring"
                    font.pixelSize: 12
                }

                ComboBox
                {
                    opacity: 0.8
                    id: coloring
                    anchors.verticalCenter: colorLabel.verticalCenter
                    model: ["solid", "layer", "mtype", "metype"]
                    width: 50

                    onCurrentIndexChanged:
                    {
                        if (currentIndex == 0)
                           legend.visible = false
                        else
                           legend.visible = true
                        coloringChanged(currentText)
                    }
                }

                Slider
                {
                    id: inflationFactor
                    label: "Inflation"
                    Layout.fillWidth: true
                    height: 20
                    minimum: 0
                    maximum: 10
                    value: 0.0
                    textColor: "black"

                    onValueChanged: inflationFactorChanged(value)
                }
            }


            Row
            {
                spacing: 10

                Layout.alignment: Qt.AlignHCenter

                Button
                {
                    checkable: false
                    checked: true
                    text: "Reset"
                    onClicked: resetClicked()
                }

                Button
                {
                    id: clipButton
                    checkable: false
                    checked: true
                    property bool state: true
                    text: state ? "Clip" : "Unclip"
                    onClicked:
                    {
                        clipNeurons(state)
                        state = !state
                    }
                }
            }

            onActiveFocusChanged:
            {
               // If any of the cell pickers yields us the focus, we return
               // it to the overlay
               if (activeFocus)
                   overlay.focus = true
            }
        }
    }

    Expandable
    {
        id: connectionInfo
        anchors.bottom: menu.top
        anchors.right: parent.right
        opacity: 0.8
        header.opacity: 0.8
        width: 350
        headerText: "Connection info"
        expandsDown: false

        Column
        {
            width: parent.width
            anchors.top: parent.header.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 10
            spacing: 10

            ListView
            {
                id: connectionInfoList

                opacity: 0.8
                width: parent.width
                height: childrenRect.height

                delegate: Text
                {
                    font.pixelSize: 12
                    text: value == "" ? name : name + ": " + value
                }
            }

            Button
            {
                anchors.horizontalCenter: parent.horizontalCenter
                id: expandAnnotationsButton
                checkable: false
                checked: true
                text: "Expand annotations"
                onClicked: expandAnnotationsClicked()
            }
        }
    }

    Legend
    {
        id: legend

        anchors.top: overlay.top
        anchors.left: overlay.left
    }
}
