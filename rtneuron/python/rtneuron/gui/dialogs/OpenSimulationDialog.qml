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

    id: openSimulationDialog

    signal done(int compartmentReportIndex, string spikeReport)
    signal cancelled()

    function invalidReport(name, message)
    {
        openError.text = 'Could not open "' + name + '": ' + message
        openError.visible = true
    }

    function setReportListModel(model)
    {
        reportList.model = model
    }

    MessageDialog
    {
        id: openError
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


            Row
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    text: "Compartment report:"
                    font.pixelSize: fontSize
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }

                ComboBox
                {
                    id: reportList
                    width: 200
                }
            }

            RowLayout
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    text: "Spike file:"
                    font.pixelSize: fontSize
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }

                TextInput
                {
                    id: spikeFile
                    Layout.fillWidth: true

                    font.pixelSize: fontSize
                    font.italic: true
                    text: "click to type"

                    clip: true
                    selectByMouse: true

                    function getText()
                    {
                        if (font.italic)
                            return "" // Still not modified
                        return text
                    }

                    function setText(t)
                    {
                        font.italic = false
                        text = t
                    }

                    onActiveFocusChanged:
                    {
                        // On first selection clear the text
                        if (activeFocus && font.italic == true)
                        text = ""
                        font.italic = false
                    }
                }

                CheckBox
                {
                    id: defaultSpikes
                    text: qsTr("Use default")
                    checked: false
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
                            title: qsTr("Choose spike file...")
                            nameFilters: [ "Spike files (*.dat *.gdf *.spikes)",
                            "All files (*)" ]
                            onAccepted:
                            {
                                var path = fileUrl.toString()
                                // Stripping the file:/// and deconding the
                                // url into a simple string.
                                path = path.replace(/^(file:\\/{2})/,"");
                                path = decodeURIComponent(path)
                                spikeFile.setText(path)
                                close()
                            }
                            onRejected:
                            {
                                close()
                            }
                        }', openSimulationDialog, "fileBrowser")
                        dialog.open()
                    }
                }
            }
            RowLayout
            {
                anchors.horizontalCenter: parent.horizontalCenter

                Button
                {
                    text: qsTr("OK")
                    width: 100
                    height: 35
                    checked: true

                    onClicked:
                    {
                        done(reportList.currentIndex,
                             defaultSpikes.checked ? "/default/" :
                                                     spikeFile.getText())
                    }
                }

                Button
                {
                    text: qsTr("Cancel")
                    width: 100
                    height: 35
                    checked: true

                    onClicked: cancelled()
                }
            }
        }
    }
}
