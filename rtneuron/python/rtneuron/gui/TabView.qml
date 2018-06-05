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

/* Implementation adapted from
   http://developer.nokia.com/community/wiki/How_to_build_a_tabbed_UI_with_QML

   Usage example:

    Item
    {
        // Contains the visual items (tabs) to be used in the view (TabView)
        VisualItemModel
        {
            id: tabsModel
            Tab
            {
                name: "Tab 1"
                icon: "pics/tab0.png"
                color: "yellow"
                Text { text: "This is page 1" }
            }
            // Add more tabs here...
        }

        TabView
        {
            // Height of the panel when expanded
            tabPanelHeight: 200
            tabBarHeight: 30
            // Attribute that uses the index property attached to each item
            // to determine the active tab (starting from 0)
            tabIndex: 0
            // References the VisualItemModel defined above
            tabsModel: tabsModel
        }
    }
*/

Item
{
    // Total panel height (expanded)
    property int tabPanelHeight: 200

    // Height of the tab bar
    property int tabBarHeight: 30

    // Index of the active tab
    property int tabIndex: 0

    // The model used to build the tabs
    property VisualItemModel tabsModel

    property bool expanded: false

    id: tabView

    anchors.fill: parent

    Item
    {
        id: tabBar

        height: tabBarHeight
        width: parent.width

        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top

        Row
        {
            id: tabs

            anchors.fill: parent

            Repeater
            {
                model: tabsModel.count
                delegate: tabBarItem
            }
        }
    }

    // Delegate component for the items on the tab bar using the names an
    // icons of the Tabs from the tabsmodel.
    Component
    {
        id: tabBarItem

        Rectangle
        {
            height: tabBarHeight
            width: tabs.width / tabsModel.count
            opacity: 0.3
            smooth: true

            Image
            {
                source: tabsModel.children[index].icon
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                anchors.topMargin: 4
            }

            Text
            {
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 4
                color: "black"
                font.bold: true
                font.pixelSize: 16
                text: tabsModel.children[index].name
            }

            MouseArea
            {
                anchors.fill: parent
                onClicked: tabClicked(index)
            }
        }
    }

    Component.onCompleted:
    {
        // Hide all the tab views and collapse the panel
        for(var i = 0; i < tabsModel.children.length; i++)
            tabsModel.children[i].visible = false
        parent.height = tabBar.height
    }

    function tabClicked(index)
    {
        // Unselect the currently selected tab
        tabs.children[tabIndex].opacity = 0.3

        // Hide the currently selected tab view
        tabsModel.children[tabIndex].visible = false

        if (expanded && index == tabIndex)
        {
            // Collapse the panel
            parent.height = tabBar.height
            expanded = false
            return
        }

        // Expand the panel
        parent.height = tabPanelHeight
        expanded = true

        // Change the current tab index
        tabIndex = index

        // Show the new tab view
        tabsModel.children[tabIndex].visible = true

        // Highlight the new tab
        tabs.children[tabIndex].opacity = 0.5
    }

    Item
    {
        id: tabViewContainer

        width: parent.width

        anchors.top: tabBar.bottom
        anchors.bottom: parent.bottom

        // Build all the tab views
        Repeater
        {
            model: tabsModel
        }
    }
}
