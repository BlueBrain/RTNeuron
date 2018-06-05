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

BaseOverlay
{
    signal updateScene(string scheme, color primary, color secondary,
                       real attenuation)

    Rectangle
    {
        color: "#cfcfcf"
        opacity: 1.0
        radius: 10

        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: childrenRect.width + 20
        height: childrenRect.height + 20

        ColumnLayout
        {
            // centerIn causes a warning about a binding loop
            x: 10
            y: 10
            width: 350

            spacing: 5

            Row
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    id: schemeLabel
                    text: "Coloring scheme"
                    font.pixelSize: 12
                }

                ComboBox
                {
                    id: colorScheme

                    anchors.verticalCenter: schemeLabel.verticalCenter
                    model: ["solid", "by-width", "by-distance",
                            "by branch"]
                    width: 180

                    onCurrentIndexChanged:
                    {
                        switch (currentIndex)
                        {
                        case 0:
                            primaryColorLabel.text = "Color"
                            break
                        case 1:
                            primaryColorLabel.text = "Zero width color"
                            secondaryColorLabel.text = "Max width color"
                            break
                        case 2:
                            primaryColorLabel.text = "Proximal color"
                            secondaryColorLabel.text = "Distal color"
                            break
                        case 3:
                            primaryColorLabel.text = "Dendrites color"
                            secondaryColorLabel.text = "Axon color"
                            break
                        }
                        secondaryColorLabel.visible = currentIndex != 0
                        secondaryColor.visible = currentIndex != 0
                        attenuationSlider.visible = currentIndex == 1
                    }
                }
            }

            Row
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Text
                {
                    id: primaryColorLabel
                    text: "Primary color"
                    font.pixelSize: 12
                    anchors.verticalCenter: parent.verticalCenter
                }
                ColorPicker
                {
                    id: primaryColor
                }

                Text
                {
                    id: secondaryColorLabel
                    text: "Secondary color"
                    font.pixelSize: 12
                    anchors.verticalCenter: parent.verticalCenter
                }
                ColorPicker
                {
                    id: secondaryColor
                }
            }

            Row
            {
                spacing: 10
                anchors.left: parent.left
                anchors.right: parent.right

                Slider
                {
                    id: attenuationSlider
                    label: qsTr("Attenuation")
                    visible: false
                    height: 20
                    maximum: 20.0
                    value: 2.0
                    textColor: "black"
                }
            }

            Button
            {
                id: updateButton
                Layout.alignment: Qt.AlignHCenter

                text: qsTr("Update")
                width: 100
                height: 30
                checked: true
                onClicked:
                {
                    updateScene(colorScheme.currentIndex,
                                primaryColor.color, secondaryColor.color,
                                attenuationSlider.value)
                }
            }
        }
    }
}
