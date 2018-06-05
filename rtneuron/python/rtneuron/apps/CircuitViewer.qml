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

import RepresentationMode 1.0

// All dialogs are created dynamically because importing QtQuick.Dialogs at
// the top of the file causes a deadlock in PyQt5 when the qml file is loaded.

BaseOverlay
{
    id: overlay

    signal addModel(string modelFile)

    // Clip plane signals
    signal enableSlice(bool enable)
    signal showSlice(bool show)
    signal sliceWidthChanged(real width)

    // Cell dyes signals
    signal addCellDye(string key, real fraction, color primary, color secondary)
    signal clearDyes()

    function showError(message)
    {
        var dialog = Qt.createQmlObject('
            import QtQuick.Dialogs 1.2
            MessageDialog
            {
                title: qsTr("Error")
                onAccepted: close()
            }', overlay, "errorMessage")
        dialog.text = message
        dialog.open()
    }

    function getModelList()
    {
        return modelList
    }

    function getCellDyeList()
    {
        return cellDyeList
    }

    function setSliceWidth(x)
    {
        widthSpin.value = x
    }

    // Slice expandable
    Expandable
    {
        id: slices
        anchors.bottom: overlay.bottom
        anchors.right: parent.right
        opacity: 0.8
        header.opacity: 0.8
        width: 260
        expandedSize: sliceButtons.height + header.height + 20 +
                      (widthEditing.visible ? widthEditing.height + 10: 0)
        headerText: "Slice"
        expandsDown: false

        Row
        {
            id: sliceButtons
            anchors.margins: 10
            anchors.bottomMargin: 0
            anchors.top: parent.header.bottom
            anchors.horizontalCenter: parent.horizontalCenter
            height: childrenRect.height
            width: childrenRect.width
            spacing: 10

            Button
            {
                width: 100
                height: 30

                checked: true
                checkable: false
                property bool added: false
                text: added ? "Clear" : "Add"
                onClicked:
                {
                    added = !added
                    enableSlice(added)
                    showSliceButton.enabled = added
                    showSliceButton.state = !added
                    showSliceButton.checked = added
                }
            }

            Button
            {
                id: showSliceButton

                width: 100
                height: 30

                checkable: false
                enabled: false
                property bool state: false
                text: state ? "Show" : "Hide"
                onClicked:
                {
                    showSlice(state)
                    state = !state
                }
            }
        }

        Row
        {
            id: widthEditing
            visible: showSliceButton.enabled && !showSliceButton.state
            anchors.margins: 10
            anchors.bottomMargin: 0
            anchors.top: sliceButtons.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            width: childrenRect.width
            spacing: 10

            Text
            {
                opacity: 0.8
                anchors.verticalCenter: parent.verticalCenter
                font.pixelSize: 12
                font.bold: true
                Layout.preferredWidth: 110
                text: "Width (\u03BCm):"
            }

            SpinBox
            {
                id: widthSpin

                opacity: 0.8
                font.pixelSize: 12
                minimumValue: 0
                maximumValue: 100000
                stepSize: 1
                anchors.verticalCenter: parent.verticalCenter
                decimals: 2
                onValueChanged: sliceWidthChanged(value)
                onEditingFinished: overlay.focus = true
            }
        }
    }

    // Cell dyes expandable
    Expandable
    {
        id: dyes
        anchors.bottom: slices.top
        anchors.right: parent.right
        opacity: 0.8
        header.opacity: 0.8
        width: 260
        headerText: "Cell dyes"
        expandsDown: false

        Column
        {
            anchors.top: parent.header.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 10
            spacing: 10

            RowLayout
            {
                width: parent.width
                spacing: 5

                TextInput
                {
                    id: targetKey

                    opacity: 0.8
                    Layout.fillWidth: true
                    text: "target or gids"
                    font.italic: true
                    selectByMouse: true

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
                    }
                    onAccepted:
                    {
                         overlay.focus = true
                         cursorVisible = false
                    }

                    Keys.onEscapePressed:
                    {
                        overlay.focus = true
                        cursorVisible = false
                    }
                }

                Item
                {
                    height: childrenRect.height
                    width: childrenRect.width

                    SpinBox
                    {
                        id: fraction
                        opacity: 0.8
                        width: 65
                        font.pixelSize: 12
                        minimumValue: 0
                        maximumValue: 100
                        value: 15
                        stepSize: 1
                        decimals: 3
                        onEditingFinished: overlay.focus = true
                    }

                    Text
                    {
                        text: "% "
                        opacity: 0.8
                        anchors.verticalCenter: fraction.verticalCenter
                        anchors.left: fraction.right
                    }
                }

                ColorPicker
                {
                    width: 17
                    id: primaryColor
                    color: "#ffff3f3f"
                    Image
                    {
                        source: "../gui/icons/soma_dendrites.svg"
                        opacity: 0.5
                        anchors.centerIn: parent
                        sourceSize.width: 15
                        sourceSize.height: 15
                    }
                }
                ColorPicker
                {
                    width: 17
                    id: secondaryColor
                    color: "#ff3f3fff"
                    Image
                    {
                        source: "../gui/icons/axon.svg"
                        opacity: 0.5
                        anchors.centerIn: parent
                        sourceSize.width: 15
                        sourceSize.height: 15
                    }
                }

                Button
                {
                    width: 20
                    height: 20
                    text: "+"
                    checkable: false
                    checked: enabled
                    enabled: (!targetKey.font.italic &&
                              targetKey.text.length != 0)

                    onClicked:
                    {
                        if (!targetKey.font.italic)
                            addCellDye(targetKey.text, fraction.value,
                                       primaryColor.color, secondaryColor.color)
                        // Returning the focus to the background because it's
                        // difficult to realize that the line input won't give
                        // it away when the + button is clicked.
                        overlay.focus = true
                    }
                }
            }

            ListView
            {
                id: cellDyeList

                signal removeItem(int index)
                signal colorsChanged(int index, color primary, color secondary)
                signal toggleMode(int index)
                spacing: 2

                function setModel(model)
                {
                    cellDyeList.model = model
                }

                width: parent.width
                height: childrenRect.height

                delegate: Item
                {
                    height: childrenRect.height
                    width: parent.width
                    RowLayout
                    {
                        width: parent.width
                        spacing: 5

                        Text
                        {
                            opacity: 0.8
                            text: key + "; " + fraction + "%"
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                        Image
                        {
                            height: 20
                            width: 17
                            sourceSize.width: 15
                            sourceSize.height: 15
                            source: "../gui/icons/neuron.svg"
                            property var repr_mode: mode

                            function toggle()
                            {
                                if (repr_mode == RepresentationMode.NO_AXON)
                                {
                                    repr_mode = RepresentationMode.WHOLE_NEURON
                                    source = "../gui/icons/neuron.svg"
                                }
                                else
                                {
                                    repr_mode = RepresentationMode.NO_AXON
                                    source = "../gui/icons/soma_dendrites.svg"
                                }
                                cellDyeList.toggleMode(index)
                            }

                            MouseArea
                            {
                                anchors.fill: parent
                                onClicked:
                                {
                                    parent.toggle()
                                }
                            }
                        }
                        ColorPicker
                        {
                            id: primaryColor
                            width: 17
                            color: primary_color
                            onChanged: cellDyeList.colorsChanged(
                                           index, color, secondaryColor.color)

                            Image
                            {
                                source: "../gui/icons/soma_dendrites.svg"
                                opacity: 0.5
                                anchors.centerIn: parent
                                sourceSize.width: 15
                                sourceSize.height: 15
                            }
                        }
                        ColorPicker
                        {
                            id: secondaryColor
                            width: 17
                            color: secondary_color
                            onChanged: cellDyeList.colorsChanged(
                                           index, primaryColor.color, color)

                            Image
                            {
                                source: "../gui/icons/axon.svg"
                                opacity: 0.5
                                anchors.centerIn: parent
                                sourceSize.width: 15
                                sourceSize.height: 15
                            }
                        }
                        Button
                        {
                            width: 20
                            height: 20
                            text: "-"
                            checkable: false
                            checked: enabled
                            onClicked: cellDyeList.removeItem(index)
                        }
                    }
                }
            }

            Button
            {
                text: "Clear"

                width: 100
                height: 30

                checked: true
                checkable: false
                anchors.horizontalCenter: parent.horizontalCenter

                onClicked: clearDyes()
            }
        }
    }

    // Extra models expandable
    Expandable
    {
        id: models
        anchors.bottom: dyes.top
        anchors.right: parent.right
        opacity: 0.8
        header.opacity: 0.8
        width: 260
        headerText: "Extra models"
        expandsDown: false

        Column
        {
            opacity: 0.8

            anchors.top: parent.header.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 10
            spacing: 10

            ListView
            {
                id: modelList

                signal removeItem(int index)
                signal colorChanged(int index, color color)

                function setModel(model)
                {
                    modelList.model = model
                }

                width: parent.width
                height: childrenRect.height

                delegate: Item
                {
                    height: childrenRect.height
                    width: parent.width

                    RowLayout
                    {
                        width: parent.width
                        Text
                        {
                            text: file_name
                            elide: Text.ElideMiddle
                            Layout.fillWidth: true
                        }
                        ColorPicker
                        {
                            id: color
                            width: 30
                            color: model_color
                            onChanged: modelList.colorChanged(index, color)
                        }
                        Button
                        {
                            width: 20
                            height: 20
                            text: "-"
                            checkable: false
                            checked: enabled
                            onClicked: modelList.removeItem(index)
                        }
                    }
                }
            }

            Button
            {
                text: qsTr("Add model")

                width: 100
                height: 30

                checked: true
                checkable: false
                anchors.horizontalCenter: parent.horizontalCenter

                onClicked:
                {
                    var dialog = Qt.createQmlObject('
                        import QtQuick.Dialogs 1.2
                        FileDialog
                        {
                            title: qsTr("Open 3D model...")
                            onAccepted:
                            {
                                var path = fileUrl.toString()
                                // Stripping the file:/// and deconding the
                                // url into a simple string.
                                path = path.replace(/^(file:\\/{2})/,"");
                                path = decodeURIComponent(path)
                                overlay.addModel(path)
                                close()
                            }
                            onRejected:
                            {
                                close()
                            }
                        }', models, "modelFileBrowser")
                    dialog.open()
                }
            }
        }
    }

    SimulationPlayer
    {
        id: player
        objectName: "player"
        primaryColor: "white"
        secondaryColor: "#dfdfdf"
        shadowColor: "#bfbfbf"
        opacity: 0.8
        anchors
        {
            bottom: parent.bottom
            left: parent.left
            right: models.left
            bottomMargin: 5
        }
    }
}
