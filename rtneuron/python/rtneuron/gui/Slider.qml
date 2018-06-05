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

Rectangle
{
    id: container

    width: parent.width
    height: 35

    opacity: 0.5
    radius: 8
    smooth: true

    property alias label: labelText.text
    property alias textColor: labelText.color
    property color primaryColor: "white"
    property color secondaryColor: "#9f9f9f"
    property color shadowColor: "#9f9f9f"

    border.width: 1
    border.color: shadowColor

    property real minimum: 0.0
    property real maximum: 1.0
    property real value: 0
    property bool dragging: false
    property bool bidirectional: false
    property real center: 0.5 // only used if bidirectional

    signal rightMouseClicked()

    function setValue(val)
    {
        if (dragging)
            // Reject programmatical updates of the slider if the user is
            // dragging
            return
        _setValue(val)
    }

    function _setValue(val)
    {
        if (bidirectional)
        {
            if (val < center)
            {
                var width = container.width *
                            (center - minimum) / (maximum - minimum)
                slider.x = width * (val - minimum) / (center - minimum)
                slider.width = width - slider.x
            }
            else
            {
                slider.x = container.width *
                          (center - minimum) / (maximum - minimum)
                var width = container.width - slider.x
                slider.width = width * (val - center) / (maximum - center)
            }
        }
        else
        {
            var alpha = (val - minimum) / (maximum - minimum)
            slider.width = alpha * container.width
        }
        value = val
    }

    gradient: Gradient {
        GradientStop { position: 0.0; color: container.secondaryColor }
        GradientStop { position: 1.0; color: container.shadowColor }
    }

    MouseArea
    {
        id: mouseArea
        anchors.fill: parent
        acceptedButtons: Qt.RightButton | Qt.LeftButton

        onPressed:
        {
            // Apparently, after opening the QDialog for text input, the mouse
            // event chain gets messed up and this area receives the next
            // click, despite it shouldn't. As a workaround we double check
            // the mouse coordinates and return if they fall outside this area.
            if (mouse.x > width || mouse.x < 0 ||
                mouse.y > height || mouse.y < 0)
                return

            if (mouse.button == Qt.LeftButton)
            {
                dragging = true;
                _update(mouse.x)
            }
            else if (mouse.button == Qt.RightButton)
            {
                var alpha = slider.width / container.width
                value = (maximum - minimum) * alpha + minimum
                rightMouseClicked()
            }
        }
        onReleased:
        {
            if (mouse.button ==  Qt.LeftButton)
                dragging = false;
        }
        onMouseXChanged:
        {
            if (!dragging)
                return;
            _update(mouse.x)
        }

        function _update(x)
        {
            var width = container.width
            x = x < 0 ? 0 : (x > width ? width : x)
            _setValue((maximum - minimum) * (x / width) + minimum)
        }
    }

    Rectangle
    {
        id: slider
        x: 0
        y: 0
        width: parent.width / 2.0
        height: parent.height
        radius: 8
        smooth: true
        opacity: 0.75
        color: primaryColor
    }

    Text
    {
        id: labelText
        anchors.horizontalCenter: container.horizontalCenter
        anchors.verticalCenter: container.verticalCenter
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.pixelSize: 12
    }

    Component.onCompleted: setValue(value)
}
