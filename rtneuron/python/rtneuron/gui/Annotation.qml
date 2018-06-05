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

Expandable
{
    id: annotation
    opacity: 0.8
    width: 200
    expandedSize: content.paintedHeight + header.height + 10

    signal updated()
    signal closeClicked()

    headerText: "Annotation"

    property bool dragging: false
    property real lineWidth: 1.5
    property color highlightColor: "#3FFF3F"

    property alias text: content.text

    onWidthChanged: updated()
    onHeightChanged: updated()
    onColorChanged: updated()

    // Annotation content
    Text
    {
        id: content

        anchors.top: parent.header.bottom
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 10
        anchors.topMargin: 5

        opacity: 0.8
        font.pixelSize: 12
    }

    // Mouse area for dragging events and forwarding the expand/collapse event
    // to the base component.
    MouseArea
    {
        anchors.fill: parent

        hoverEnabled: true
        property point clickPos
        property point annotationPos

        // internal
        property bool _wasDragged: false

        onPressed:
        {
            if (mouse.button == Qt.LeftButton)
            {
                clickPos = Qt.point(mouse.x, mouse.y)
                annotationPos = Qt.point(annotation.x, annotation.y)
                dragging = true
                _wasDragged = false
            }
        }

        onReleased:
        {
            dragging = false;
            // If the mouse is outside the bouding box we have to ensure
            // the annotation is not left with the hovered color
            if (!containsMouse)
            {
                annotation.color = "#FFFFFF"
                annotation.lineWidth = 1.5
            }
        }

        onClicked:
        {
            if (!_wasDragged)
                toggle()
        }

        onEntered:
        {
            annotation.color = highlightColor
            annotation.lineWidth = 2.5
        }
        onExited:
        {
            if (!dragging)
            {
                annotation.color = "#FFFFFF"
                annotation.lineWidth = 1.5
            }
        }

        onPositionChanged:
        {
            if (!dragging)
                return;
            // This is a workaround for mouse.wasHeld always returning
            // false.
            _wasDragged = true

            var shiftX = mouse.x - clickPos.x
            var shiftY = mouse.y - clickPos.y
            var maxX = annotation.parent.width - annotation.width
            annotation.x += shiftX
            if (annotation.x < 0)
                annotation.x = 0
            if (annotation.x > maxX)
                annotation.x = maxX
            var maxY = annotation.parent.height - annotation.height
            annotation.y += shiftY
            if (annotation.y < 0)
                annotation.y = 0
            if (annotation.y > maxY)
                annotation.y = maxY
            updated()
        }
    }

    // Close button.
    // Must be defined after all the rest mouse areas
    Text
    {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.rightMargin: 10
        anchors.topMargin: 5
        height: parent.header.height
        width: parent.header.height
        text: "\u2716" // Heavy cross unicode character
        MouseArea
        {
            anchors.fill:parent
            onClicked: closeClicked()
        }
    }
}
