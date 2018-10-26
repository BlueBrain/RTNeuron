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
import QtQuick.Dialogs 1.1
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.1

ModalDialog
{

    id: loadDialog

    signal done(string configFile, string target, string displayMode)

    function invalidSimulation(simulation, message)
    {
        loadError.text = 'Could not load "' + simulation + '": ' + message
        loadError.visible = true
    }

    function invalidTarget(target, message)
    {
        if (target == undefined)
            loadError.text = message
        else
            loadError.text = 'Invalid target "' + target + '": ' + message
        loadError.visible = true
    }

    MessageDialog
    {
        id: loadError
        title: "Error"
        onAccepted: visible = false
    }

    MouseArea
    {
        // Blocking event propagation
        anchors.fill: parent
    }

    Rectangle
    {
        opacity: 0.8
        radius: 10
        border.width: 0

        width: 500
        height: 130

        anchors.centerIn: parent

        property real fontSize: 12

        Column
        {
            anchors.topMargin: 10
            anchors.leftMargin: 10
            anchors.rightMargin: 10
            anchors.fill: parent
            spacing: 10

            Component
            {
                id: textInput

                TextInput
                {
                    Layout.fillWidth: true

                    function getText()
                    {
                        if (_input.font.italic)
                            return "" // Still not modified
                        return _input.text
                    }

                    function setText(text)
                    {
                        _input.text = text
                        _input.font.italic = false
                    }

                    id: _input

                    clip: true
                    font.pixelSize: fontSize
                    font.italic: true

                    text: "default"

                    selectByMouse: true

                    onActiveFocusChanged:
                    {
                        // On first selection clear the text
                        if (activeFocus && font.italic == true)
                            text = ""
                        font.italic = false
                    }
                }
            }

            RowLayout
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    id: configFileLabel
                    text: "Blue or Circuit config:"
                    font.pixelSize: fontSize
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }

                Loader
                {
                    id: configFileInput
                    sourceComponent: textInput
                    Layout.fillWidth: true
                    height: configFileLabel.height
                    anchors.verticalCenter: parent.verticalCenter
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
                                title: qsTr("Open Blue or Circuit config...")
                                nameFilters: [ "Configs (BlueConfig " +
                                               "CircuitConfig *.json ",
                                               "All files (*)" ]
                                onAccepted:
                                {
                                    var path = fileUrl.toString()
                                    // Stripping the file:/// and deconding the
                                    // url into a simple string.
                                    path = path.replace(/^(file:\\/{2})/,"");
                                    path = decodeURIComponent(path)
                                    configFileInput.item.setText(path)
                                    close()
                                }
                                onRejected:
                                {
                                    close()
                                }
                            }', loadDialog, "fileBrowser")
                        dialog.open()
                    }
                }
            }

            RowLayout
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    id: targetLabel
                    text: "Targets:"
                    font.pixelSize: fontSize
                    font.bold: true
                    width: configFileLabel.width
                    anchors.verticalCenter: parent.verticalCenter
                }

                Loader
                {
                    id: targetInput
                    sourceComponent: textInput
                    Layout.fillWidth: true
                    height: targetLabel.height
                    anchors.verticalCenter: parent.verticalCenter
                }

                Text
                {
                    id: modeLabel
                    text: "Display mode:"
                    font.pixelSize: fontSize
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }

                ComboBox
                {
                    id: displayMode
                    width: 120
                    model: ["Soma", "Detailed", "No axon"]
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

                onClicked: done(configFileInput.item.getText(),
                                targetInput.item.getText(),
                                displayMode.currentText)
            }

        }
    }
}
