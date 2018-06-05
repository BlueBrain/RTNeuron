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

import "../gui"

BaseOverlay
{
    signal inflationFactorChanged(real factor)
    signal presynapticOpacityChanged(real factor)
    signal postsynapticOpacityChanged(real factor)
    signal circuitOpacityChanged(real factor)
    signal clearClicked()

    Rectangle
    {
        color: "#cfcfcf"
        opacity: 1.0
        radius: 10

        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: childrenRect.width + 20
        height: childrenRect.height + 15

        ColumnLayout
        {
            x: 10
            y: 10
            spacing: 5
            width: 330

            Slider
            {
                id: circuitOpacitySlider
                label: qsTr("Circuit opacity")
                height: 20
                minimum: 0
                maximum: 1
                value: 0.5
                textColor: "black"

                onValueChanged: circuitOpacityChanged(value)
            }

            Slider
            {
                id: preOpacitySlider
                label: qsTr("Presynaptic opacity")
                height: 20
                minimum: 0
                maximum: 1
                value: 0.5
                textColor: "black"

                onValueChanged: presynapticOpacityChanged(value)
            }

            Slider
            {
                id: postOpacitySlider
                label: qsTr("Postsynaptic opacity")
                height: 20
                minimum: 0
                maximum: 1
                value: 0.5
                textColor: "black"

                onValueChanged: postsynapticOpacityChanged(value)
            }

            Slider
            {
                id: inflationFactor
                label: qsTr("Inflation")
                height: 20
                minimum: 0
                maximum: 10
                value: 0.0
                textColor: "black"

                onValueChanged: inflationFactorChanged(value)
            }

            Button
            {
                Layout.alignment: Qt.AlignHCenter
                checkable: false
                checked: true
                text: "Clear"
                onClicked: clearClicked()
            }
        }
    }
}
