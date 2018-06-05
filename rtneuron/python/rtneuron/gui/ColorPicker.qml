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
    id: rect

    signal changed(color color)

    height: 20
    width: 40
    color: "#ffffffff"
    MouseArea
    {
        anchors.fill: parent
        onClicked:
        {
            // The color dialog is created dynamically to workaround a deadlock
            // in PyQt5 when QtQuick.Dialogs is imported at qml load time.
            var dialog = Qt.createQmlObject('
                import QtQuick.Dialogs 1.2
                ColorDialog
                {
                    color: rect.color
                    showAlphaChannel: true
                    modality: Qt.ApplicationModal
                    title: "Color chooser"
                    onAccepted:
                    {
                        rect.color = currentColor
                        rect.changed(currentColor)
                    }
                }', rect, 'colorPicker')
            dialog.open()
        }
    }
}
