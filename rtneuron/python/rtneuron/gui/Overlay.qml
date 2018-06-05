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

BaseOverlay
{
    property bool guiVisible: true
    property bool playerEnabled: false
    property bool steeringEnabled: false
    property bool steeringAvailable: false

    signal attrChanged(string category, string name, variant value)
    signal enableSteeringClicked()
    signal spikeTailSliderChanged(real value)
    signal spikeTailInputDialogRequested(real value)
    signal takeSnapshotToggled(bool checked)

    signal injectStimuliToggled(bool checked)
    signal changeSimulatorState(bool play)

    signal toggleFullscreen()

    function setSteeringEnabled(value)
    {
        steeringEnabled = true
    }

    function setSteeringAvailable(value)
    {
        steeringAvailable = true
    }

    function onSpikeTailSliderValue(value)
    {
        spikeTailSlider.setValue(value)
    }

    function setPlayerEnabled(enabled)
    {
        playerEnabled = enabled
    }

    function onSnapshotDone()
    {
        takeSnapshotButton.checked = false
    }

    function onInjectStimuliDone()
    {
        injectStimuliButton.checked = false
    }


    id: screen

    Button
    {
        anchors { top: parent.top; left: parent.left }
        text: qsTr("Show GUI")
        textColor: "white"

        onClicked:
        {
            anim.restart()
            guiVisible = !guiVisible
        }
    }

    SimulationPlayer
    {
        id: player
        objectName: "player"
        visible: guiVisible
        textColor: "white"
        anchors
        {
            bottom: parent.bottom
            left: parent.left
            right: parent.right
            bottomMargin: 10
        }
    }

    PropertyAnimation
    {
        id: anim
        targets: [options, visualOptions]
        properties: "x"
        to: guiVisible ? options.x + options.width : options.x - options.width
        easing.type: Easing.InOutElastic
        easing.amplitude: 3.0
        easing.period: 1.0
        duration: 400
    }

    Item
    {
        anchors { top: parent.top; right: parent.right }
        width: 150
        height: parent.height

        Column
        {
            id: options
            spacing: 10
            width: parent.width

            Button
            {
                text: qsTr("Enable steering")
                checked: !steeringEnabled
                visible: (playerEnabled && steeringAvailable &&
                          !steeringEnabled)
                textColor: "white"

                onClicked:
                {
                    enableSteeringClicked()
                }
            }

            Button
            {
                property bool state: true
                checked: true
                text: state ? qsTr("Pause simulator") : qsTr("Resume simulator")
                visible: steeringEnabled
                textColor: "white"
                onClicked:
                {
                    state = !state
                    changeSimulatorState(state)
                }
            }

            Button
            {
                id: injectStimuliButton
                text: qsTr("Inject Stimuli")
                visible: steeringEnabled
                textColor: "white"
                checkable: true
                onToggled: injectStimuliToggled(checked)
            }

            Slider
            {
                id: spikeTailSlider
                label: qsTr("Spike tail")
                visible: playerEnabled

                maximum: 10.0
                textColor: "white"

                onValueChanged:
                {
                    attrChanged("view", "spike_tail", value)
                }
                onRightMouseClicked:
                {
                    spikeTailInputDialogRequested(value)
                }
            }
        }

        Column
        {
            id: visualOptions
            spacing: 10
            anchors.top: options.bottom
            anchors.topMargin: 20

            Button
            {
                text: qsTr("Toggle fullscreen")
                checkable: true
                textColor: "white"

                onClicked:
                {
                    toggleFullscreen()
                }
            }

            Button
            {
                id: takeSnapshotButton
                text: qsTr("Take snapshot")
                checkable: true
                textColor: "white"
                onToggled: takeSnapshotToggled(checked)
            }

            Button
            {
                text: qsTr("EM shading")
                checkable: true
                textColor: "white"

                onClicked:
                {
                    attrChanged("scene", "em_shading", checked)
                }
            }

            Button
            {
                text: qsTr("Stereo")
                checkable: true
                textColor: "white"

                onClicked:
                {
                    attrChanged("view", "stereo", checked)
                }
            }

            Button
            {
                text: qsTr("Transparency")
                checkable: true
                textColor: "white"

                onClicked:
                {
                    attrChanged("scene", "transparency", checked)
                }
            }
        }
    }
}
