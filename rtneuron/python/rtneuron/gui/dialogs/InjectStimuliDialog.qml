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
    signal generatorSelected(string generator)
    signal injectStimuli(bool allCells)

    width: generatorListView.width + generatorDetailView.width + 30
    height: 420

    function onGeneratorValid(valid)
    {
        buttonRow.visible = valid
    }

    Component
    {
        id: generatorListDelegate
        Rectangle
        {
            height: 20
            width: generatorListView.width
            radius: 3
            color: generatorListView.currentIndex == index ?
                   "lightblue" : "white"
            Text
            {
                text: generatorName
                font.bold: generatorListView.currentIndex == index
                anchors.horizontalCenter: parent.horizontalCenter
            }

            MouseArea
            {
                anchors.fill: parent
                onClicked:
                {
                    generatorListView.currentIndex = index
                    generatorDetailView.currentIndex = -1
                    generatorSelected(generatorName)
                }
            }
        }
    }

    Component
    {
        id: generatorDetailDelegate
        Rectangle
        {
            id: textRect
            height: 20
            width: generatorDetailView.width
            radius: 3

            Row
            {
                Text
                {
                    width: generatorDetailView.width / 2
                    text: generatorDetailKey + ": "
                    horizontalAlignment: Text.AlignRight
                }
                TextInput
                {
                    id: detailTextInput
                    width: generatorDetailView.width / 2
                    text: generatorDetailValue
                    font.bold: cursorVisible
                    selectByMouse: true
                    cursorVisible: generatorDetailView.currentIndex == index
                    color: generatorDetailValid ? "#000000" : "#FF0000"
                    activeFocusOnTab: true
                    onEditingFinished: generatorDetailModel.updateItem(
                                           index, text)
                    onFocusChanged:
                    {
                        if (!activeFocus)
                            generatorDetailModel.updateItem(index, text)
                    }
                    Keys.onEscapePressed:
                        generatorDetailView.parent.focus = true
                }
            }
        }
    }

    ListView
    {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.topMargin: 10
        anchors.leftMargin: 10
        width: 220
        height: parent.height
        spacing: 2
        id: generatorListView
        currentIndex: -1
        model: generatorListModel
        delegate: generatorListDelegate
    }

    ListView
    {
        anchors.top: parent.top
        anchors.left: generatorListView.right
        anchors.topMargin: 10
        anchors.leftMargin: 10
        width: 220
        height: parent.height
        spacing: 2
        id: generatorDetailView
        currentIndex: -1
        model: generatorDetailModel
        delegate: generatorDetailDelegate
    }

    Row
    {
        id: buttonRow
        visible: false
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 10
        anchors.bottomMargin: 10
        anchors.leftMargin: 10

        Button
        {
            text: qsTr("Inject Stimuli")
            width: 200
            height: 40
            checked: true
            onClicked:
            {
                // Force finish current editing
                generatorDetailView.focus = false
                // And inject if still valid
                if (buttonRow.visible)
                    injectStimuli(false)
            }
        }

        Button
        {
            width: 200
            height: 40
            text: qsTr("Inject Multiple Stimuli")
            checked: true
            onClicked:
            {
                // Force finish current editing
                generatorDetailView.focus = false
                // And inject if still valid
                if (buttonRow.visible)
                    injectStimuli(true)
            }
        }
    }

}
