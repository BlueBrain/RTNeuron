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

import ".."

Dialog
{
    signal takeSnapshot(string filename, int width, int height)

    width: 300
    height: 130

    function makeSnapshot()
    {
        function validate(input)
        {
            if (input.acceptableInput)
            {
                input.color = "black"
                return true
            }
            input.color = "red"
            input.focus = true
            return false
        }

        if (!validate(heightInput) || !validate(widthInput))
            return

        takeSnapshot(filenameInput.text, widthInput.value, heightInput.value)
    }

    FocusScope
    {
        id: scope
        anchors.fill: parent

        Column
        {
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.topMargin: 10
            anchors.leftMargin: 10
            anchors.rightMargin: 10
            anchors.fill: parent
            spacing: 10

            Row
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    id: filenameLabel
                    text: "Filename:"
                    font.pixelSize: fontSize
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }

                TextInput
                {
                    id: filenameInput
                    text: qsTr("snapshot.png")

                    width: (parent.width - filenameLabel.width -
                            browseButton.width - parent.spacing * 2)
                    clip: true
                    font.pixelSize: fontSize
                    anchors.bottom: filenameLabel.bottom

                    focus: true
                    selectByMouse: true
                    onAccepted:
                    {
                        widthInput.focus = true
                    }
                    Keys.onEscapePressed: { scope.parent.focus = true }
                    KeyNavigation.tab: widthInput
                    KeyNavigation.backtab: heightInput
                }

                Button
                {
                    id: browseButton
                    text: qsTr("Browse")
                    width: 80
                    height: 30
                    checked: true
                    onClicked:
                    {
                        // Creating the dialog dynamically is a workaround for
                        // a deadlock in PyQt5 when QtQuick.Dialogs is imported
                        // at qml load time.
                        var dialog = Qt.createQmlObject('
                            import QtQuick.Dialogs 1.0
                            FileDialog
                            {
                                title: qsTr("Export snapshot image as...")
                                nameFilters: [ "Image files (*.jpg *.png)",
                                               "All files (*)" ]
                                onAccepted:
                                {
                                    var path = fileUrl.toString()
                                    // Stripping the file:/// and deconding the
                                    // url into a simple string.
                                    path = path.replace(/^(file:\\/{2})/,"");
                                    path = decodeURIComponent(path)
                                    filenameInput.text = path
                                    close()
                                }
                                onRejected:
                                {
                                    close()
                                }
                            }', scope, "fileBrowser")
                        dialog.open()
                    }
                }
            }

            Row
            {
                spacing: 10

                Text
                {
                    id: dimensionsLabel
                    text: qsTr("Dimensions:")
                    font.pixelSize: fontSize
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }

                TextInput
                {
                    id: widthInput
                    property int value: 1920

                    width: 40
                    horizontalAlignment: TextEdit.AlignRight
                    text: value
                    font.pixelSize: fontSize
                    anchors.bottom: dimensionsLabel.bottom

                    selectByMouse: true
                    validator: IntValidator { bottom: 1 }
                    onAccepted:
                    {
                        heightInput.focus = true
                    }
                    onEditingFinished:
                    {
                        color = "black"
                        value = parseInt(text)
                    }
                    Keys.onEscapePressed: { scope.parent.focus = true }
                    KeyNavigation.tab: heightInput
                    KeyNavigation.backtab: filenameInput
                }

                Text
                {
                    text: "px   ×" // The x is the UTF8 char c3 97
                    font.pixelSize: fontSize
                    font.bold: false
                    anchors.verticalCenter: parent.verticalCenter
                }

                TextInput
                {
                    id: heightInput
                    property int value: 1080

                    width: 40
                    horizontalAlignment: Text.AlignRight
                    text: value
                    font.pixelSize: fontSize
                    anchors.bottom: dimensionsLabel.bottom

                    selectByMouse: true
                    validator: IntValidator { bottom: 1 }
                    onAccepted:
                    {
                        scope.parent.focus = true
                    }
                    onEditingFinished:
                    {
                        color = "black"
                        value = parseInt(text)
                    }
                    Keys.onEscapePressed: { scope.parent.focus = true }
                    KeyNavigation.tab: filenameInput
                    KeyNavigation.backtab: widthInput
                }

                Text
                {
                    text: "px"
                    font.pixelSize: fontSize
                    font.bold: false
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            Button
            {
                id: ok
                anchors.horizontalCenter: parent.horizontalCenter
                text: qsTr("OK")
                width: 100
                height: 35
                checked: true

                onClicked:
                {
                    makeSnapshot()
                }
            }
        }
    }

    Keys.onReturnPressed: makeSnapshot()
}

