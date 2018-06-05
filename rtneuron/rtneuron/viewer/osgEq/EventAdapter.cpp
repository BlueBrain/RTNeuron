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

#include "EventAdapter.h"

#include <eq/eventICommand.h>
#include <eq/fabric/keyEvent.h>
#include <eq/fabric/pointerEvent.h>

namespace bbp
{
namespace rtneuron
{
namespace osgEq
{
namespace
{
int _translateKey(uint32_t key)
{
    switch (key)
    {
    case eq::KC_ESCAPE:
        return osgGA::GUIEventAdapter::KEY_Escape;
    case eq::KC_BACKSPACE:
        return osgGA::GUIEventAdapter::KEY_BackSpace;
    case eq::KC_RETURN:
        return osgGA::GUIEventAdapter::KEY_Return;
    case eq::KC_TAB:
        return osgGA::GUIEventAdapter::KEY_Tab;
    case eq::KC_HOME:
        return osgGA::GUIEventAdapter::KEY_Home;
    case eq::KC_LEFT:
        return osgGA::GUIEventAdapter::KEY_Left;
    case eq::KC_UP:
        return osgGA::GUIEventAdapter::KEY_Up;
    case eq::KC_RIGHT:
        return osgGA::GUIEventAdapter::KEY_Right;
    case eq::KC_DOWN:
        return osgGA::GUIEventAdapter::KEY_Down;
    case eq::KC_PAGE_UP:
        return osgGA::GUIEventAdapter::KEY_Page_Up;
    case eq::KC_PAGE_DOWN:
        return osgGA::GUIEventAdapter::KEY_Page_Down;
    case eq::KC_END:
        return osgGA::GUIEventAdapter::KEY_End;
    case eq::KC_F1:
        return osgGA::GUIEventAdapter::KEY_F1;
    case eq::KC_F2:
        return osgGA::GUIEventAdapter::KEY_F2;
    case eq::KC_F3:
        return osgGA::GUIEventAdapter::KEY_F3;
    case eq::KC_F4:
        return osgGA::GUIEventAdapter::KEY_F4;
    case eq::KC_F5:
        return osgGA::GUIEventAdapter::KEY_F5;
    case eq::KC_F6:
        return osgGA::GUIEventAdapter::KEY_F6;
    case eq::KC_F7:
        return osgGA::GUIEventAdapter::KEY_F7;
    case eq::KC_F8:
        return osgGA::GUIEventAdapter::KEY_F8;
    case eq::KC_F9:
        return osgGA::GUIEventAdapter::KEY_F9;
    case eq::KC_F10:
        return osgGA::GUIEventAdapter::KEY_F10;
    case eq::KC_F11:
        return osgGA::GUIEventAdapter::KEY_F11;
    case eq::KC_F12:
        return osgGA::GUIEventAdapter::KEY_F12;
    case eq::KC_F13:
        return osgGA::GUIEventAdapter::KEY_F13;
    case eq::KC_F14:
        return osgGA::GUIEventAdapter::KEY_F14;
    case eq::KC_F15:
        return osgGA::GUIEventAdapter::KEY_F15;
    case eq::KC_F16:
        return osgGA::GUIEventAdapter::KEY_F16;
    case eq::KC_F17:
        return osgGA::GUIEventAdapter::KEY_F17;
    case eq::KC_F18:
        return osgGA::GUIEventAdapter::KEY_F18;
    case eq::KC_F19:
        return osgGA::GUIEventAdapter::KEY_F19;
    case eq::KC_F20:
        return osgGA::GUIEventAdapter::KEY_F20;
    case eq::KC_F21:
        return osgGA::GUIEventAdapter::KEY_F21;
    case eq::KC_F22:
        return osgGA::GUIEventAdapter::KEY_F22;
    case eq::KC_F23:
        return osgGA::GUIEventAdapter::KEY_F23;
    case eq::KC_F24:
        return osgGA::GUIEventAdapter::KEY_F24;
    case eq::KC_SHIFT_L:
        return osgGA::GUIEventAdapter::KEY_Shift_L;
    case eq::KC_SHIFT_R:
        return osgGA::GUIEventAdapter::KEY_Shift_R;
    case eq::KC_CONTROL_L:
        return osgGA::GUIEventAdapter::KEY_Control_L;
    case eq::KC_CONTROL_R:
        return osgGA::GUIEventAdapter::KEY_Control_R;
    case eq::KC_ALT_L:
        return osgGA::GUIEventAdapter::KEY_Alt_L;
    case eq::KC_ALT_R:
        return osgGA::GUIEventAdapter::KEY_Alt_R;
    case eq::KC_VOID:
        return -1;
    default:
        return key;
    }
}

int _getKeyModifiers(const eq::KeyModifier eqModifiers)
{
    int result = 0;
    if ((eqModifiers & eq::KeyModifier::alt) != eq::KeyModifier::none)
        result |= osgGA::GUIEventAdapter::MODKEY_ALT;
    if ((eqModifiers & eq::KeyModifier::control) != eq::KeyModifier::none)
        result |= osgGA::GUIEventAdapter::MODKEY_CTRL;
    if ((eqModifiers & eq::KeyModifier::shift) != eq::KeyModifier::none)
        result |= osgGA::GUIEventAdapter::MODKEY_SHIFT;
    return result;
}
}

/*
  Constructors
*/
EventAdapter::EventAdapter()
{
    _eventType = NONE;
    _key = -1;
    _button = -1;
    _Xmin = -1;
    _Xmax = 1;
    _Ymin = -1;
    _Ymax = 1;
    _mx = 0;
    _my = 0;
    _buttonMask = 0;
    _modKeyMask = 0;
    _mouseYOrientation = Y_INCREASING_UPWARDS;
    _windowX = 0;
    _windowY = 0;
    _windowWidth = -1;
    _windowHeight = -1;
    setScrollingMotionDelta(0.f, 0.f);
}

EventAdapter::EventAdapter(eq::EventICommand command,
                           const eq::RenderContext& lastFocusedContext)
    : EventAdapter()
{
    eq::PointerEvent pointerEvent;
    eq::KeyEvent keyEvent;

    switch (command.getEventType())
    {
    case eq::EVENT_CHANNEL_POINTER_MOTION:
    case eq::EVENT_CHANNEL_POINTER_BUTTON_PRESS:
    case eq::EVENT_CHANNEL_POINTER_BUTTON_RELEASE:
    case eq::EVENT_CHANNEL_POINTER_WHEEL:
    case eq::EVENT_WINDOW_POINTER_WHEEL:
    case eq::EVENT_WINDOW_POINTER_MOTION:
    case eq::EVENT_WINDOW_POINTER_BUTTON_PRESS:
    case eq::EVENT_WINDOW_POINTER_BUTTON_RELEASE:
    {
        pointerEvent = command.read<eq::PointerEvent>();
        setTime(double(pointerEvent.time) / 1000.0);

        _windowX = pointerEvent.context.pvp.x;
        _windowY = pointerEvent.context.pvp.y;
        _windowWidth = pointerEvent.context.pvp.w;
        _windowHeight = pointerEvent.context.pvp.h;
        _context.originator = pointerEvent.originator;
        _context.frameID = pointerEvent.context.frameID;
        _context.offset = pointerEvent.context.offset;
        _context.frustum = pointerEvent.context.frustum;
        _context.ortho = pointerEvent.context.ortho;
        _context.headTransform = pointerEvent.context.headTransform;
        break;
    }

    case eq::EVENT_KEY_PRESS:
    case eq::EVENT_KEY_RELEASE:
        keyEvent = command.read<eq::KeyEvent>();
        setTime(double(keyEvent.time) / 1000.0);
        _key = _translateKey(keyEvent.key);
        _modKeyMask = _getKeyModifiers(keyEvent.modifiers);
        break;

    default:
        break;
        setTime(double(command.read<eq::Event>().time) / 1000.0);
    }

    // there should be no command.read<>() below this line

    const int x = pointerEvent.x - _context.offset.x();
    const int y = pointerEvent.y - _context.offset.y();

    switch (command.getEventType())
    {
    case eq::EVENT_CHANNEL_POINTER_MOTION:
    case eq::EVENT_CHANNEL_POINTER_WHEEL:
        if (pointerEvent.buttons == eq::PTR_BUTTON_NONE)
        {
            _eventType = MOVE;
        }
        else
        {
            _eventType = DRAG;
            _windowX = lastFocusedContext.pvp.x;
            _windowY = lastFocusedContext.pvp.y;
            _windowWidth = lastFocusedContext.pvp.w;
            _windowHeight = lastFocusedContext.pvp.h;

            /* This is a temporary hotfix for the an Equalizer issue in
               multipipe 2D configurations. What happens sometimes is that
               the render context provided refers to the channel viewport
               instead of the destination channel view. */
            _windowWidth /= lastFocusedContext.vp.w;
            _windowHeight /= lastFocusedContext.vp.h;
        }

        if (_windowHeight == -1 || _windowWidth == -1)
            /* Discarding the event (as NONE) because we are outside any
               channel */
            return;

        _mx = (x / (float)_windowWidth) * 2 - 1;
        _my = ((_windowHeight - y) / (float)_windowHeight) * 2 - 1;
        break;

    case eq::EVENT_CHANNEL_POINTER_BUTTON_PRESS:
        _eventType = PUSH;
        _mx = (x / (float)_windowWidth) * 2 - 1;
        _my = ((_windowHeight - y) / (float)_windowHeight) * 2 - 1;
        break;

    case eq::EVENT_CHANNEL_POINTER_BUTTON_RELEASE:
        _eventType = RELEASE;
        _mx = (x / (float)_windowWidth) * 2 - 1;
        _my = ((_windowHeight - y) / (float)_windowHeight) * 2 - 1;
        break;

    case eq::EVENT_KEY_PRESS:
        _eventType = KEYDOWN;
        break;

    case eq::EVENT_KEY_RELEASE:
        _eventType = KEYUP;
        break;

    case eq::EVENT_WINDOW_SHOW:
    case eq::EVENT_WINDOW_EXPOSE:
    case eq::EVENT_WINDOW_RESIZE:
    case eq::EVENT_CHANNEL_RESIZE:
    case eq::EVENT_VIEW_RESIZE:
        _eventType = RESIZE;
        break;

    /* These events don't have an equivalent in OSG, so they are ignored */
    case eq::EVENT_WINDOW_CLOSE:
    case eq::EVENT_WINDOW_HIDE:
    case eq::EVENT_WINDOW_SCREENSAVER:
    case eq::EVENT_STATISTIC:
        _eventType = NONE;
        break;
    }

    switch (command.getEventType())
    {
    case eq::EVENT_CHANNEL_POINTER_MOTION:
    case eq::EVENT_CHANNEL_POINTER_BUTTON_PRESS:
    case eq::EVENT_CHANNEL_POINTER_BUTTON_RELEASE:
        if (pointerEvent.button == eq::PTR_BUTTON1)
            _button = LEFT_MOUSE_BUTTON;
        else if (pointerEvent.button == eq::PTR_BUTTON2)
            _button = MIDDLE_MOUSE_BUTTON;
        else if (pointerEvent.button == eq::PTR_BUTTON3)
            _button = RIGHT_MOUSE_BUTTON;

        if (pointerEvent.buttons & eq::PTR_BUTTON1)
            _buttonMask |= LEFT_MOUSE_BUTTON;
        if (pointerEvent.buttons & eq::PTR_BUTTON2)
            _buttonMask |= MIDDLE_MOUSE_BUTTON;
        if (pointerEvent.buttons & eq::PTR_BUTTON3)
            _buttonMask |= RIGHT_MOUSE_BUTTON;

        _modKeyMask = _getKeyModifiers(pointerEvent.modifiers);
        break;

    case eq::EVENT_CHANNEL_POINTER_WHEEL:
        if (pointerEvent.yAxis == 0)
            break;
        _eventType = SCROLL;
        setScrollingMotionDelta(0.f, 0.01f);
        setScrollingMotion(pointerEvent.yAxis < 0
                               ? osgGA::GUIEventAdapter::SCROLL_UP
                               : osgGA::GUIEventAdapter::SCROLL_DOWN);
        _modKeyMask = _getKeyModifiers(pointerEvent.modifiers);
        break;

    default:
        break;
    }
}
}
}
}
