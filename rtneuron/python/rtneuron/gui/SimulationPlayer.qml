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
import QtQuick 2.0
import QtQuick.Layouts 1.1

RowLayout
{
    signal speedSliderChanged(real value)
    signal speedInputDialogRequested(real value)
    signal openSimulationClicked()
    signal playPauseClicked()
    signal loopButtonToggled(bool checked)
    signal timeSliderChanged(real value)

    anchors.leftMargin: 5
    anchors.rightMargin: 5

    opacity: 0.5

    property color primaryColor: "white"
    property color secondaryColor: "#9f9f9f"
    property color shadowColor: "#9f9f9f"
    property color textColor: "black"

    function enablePlaybackControls(enable)
    {
        playPauseButton.enabled = enable
        loopButton.enabled = enable
        timeSlider.enabled = enable
        speedSlider.enabled = enable
    }

    function enableOpenButton()
    {
        openButton.visible = true
    }

    function setPlaybackState(state)
    {
        playPauseButton.text = state ? "\u25AE\u25AE" : "\u25B6"
    }

    function onSimulationTimeChanged(timestamp, begin, end)
    {
        var value = (timestamp - begin) / (end - begin)
        // Testing if the value is NaN
        if (value == value)
        {
            timeSlider.setValue(value)
            timeSlider.label = timestamp.toFixed(2) + " ms"
        }
        else
        {
           timeSlider.setValue(0)
           timeSlider.label = ""
        }
    }

    function onPlaybackSpeedChanged(value)
    {
        speedSlider.setValue(value)
    }

    Button
    {
        id: openButton

        // By default this button is invisible until the Python handler
        // has a target scene
        visible: false

        text: "\u23CF"
        textColor: parent.textColor
        fontSize: 15
        color: parent.primaryColor
        baseOpacity: enabled ? 1.0 : 0.2

        width: 25
        height: 25

        checked: true
        checkable: false
        onClicked: openSimulationClicked()
    }

    Button
    {
        id: playPauseButton

        text: "\u25B6"
        textColor: parent.textColor
        fontSize: 15
        color: parent.primaryColor

        width: 25
        height: 25
        baseOpacity: enabled ? 1.0 : 0.2

        checked: true
        checkable: false
        onClicked: playPauseClicked()
        enabled: false
    }

    Button
    {
        id: loopButton

        text: "\u21BB"
        textColor: parent.textColor
        fontSize: 15
        color: parent.primaryColor

        width: 25
        height: 25
        baseOpacity: enabled ? 1.0 : 0.2

        checked: false
        checkable: true
        onToggled: loopButtonToggled(checked)
        enabled: false
    }

    Slider
    {
        id: timeSlider
        Layout.fillWidth: true

        textColor: parent.textColor
        primaryColor: parent.primaryColor
        secondaryColor: parent.secondaryColor
        shadowColor: parent.shadowColor

        height: 16
        opacity: enabled ? 1.0 : 0.2

        onValueChanged:
        {
            if (dragging)
                timeSliderChanged(value)
        }

        enabled: false
    }

    Slider
    {
        id: speedSlider

        label: qsTr("Playback speed")
        textColor: parent.textColor
        primaryColor: parent.primaryColor
        secondaryColor: parent.secondaryColor
        shadowColor: parent.shadowColor
        bidirectional: true

        width: 120
        height: 16
        opacity: enabled ? 1.0 : 0.2

        maximum: 5.0
        minimum: -5.0
        center: 0.0
        value: 0.1

        onValueChanged:
        {
            speedSliderChanged(value)
        }
        onRightMouseClicked:
        {
            speedInputDialogRequested(value)
        }

        enabled: false
    }
}
