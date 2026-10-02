"""Global Windows hotkey; errors preserve the previously registered shortcut."""

import ctypes
import os
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

HOTKEY_ID = 0x4A45


def parse_hotkey(text: str) -> tuple[int, int]:
    parts = [part.strip().upper() for part in text.split("+")]
    if len(parts) < 2 or not all(parts):
        raise ValueError("Пример сочетания: Ctrl+Alt+Space")
    modifiers = 0
    for modifier in parts[:-1]:
        values = {"CTRL": 2, "ALT": 1, "SHIFT": 4, "WIN": 8}
        if modifier not in values:
            raise ValueError("Поддерживаются Ctrl, Alt, Shift и Win")
        modifiers |= values[modifier]
    key = parts[-1]
    if key == "SPACE":
        code = 0x20
    elif len(key) == 1 and key.isascii() and key.isalnum():
        code = ord(key)
    elif key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 12:
        code = 0x70 + int(key[1:]) - 1
    else:
        raise ValueError("Основная клавиша: Space, буква, цифра или F1–F12")
    return modifiers | 0x4000, code


class GlobalHotkey(QAbstractNativeEventFilter):
    def __init__(self, application, callback):
        super().__init__()
        self.application = application
        self.callback = callback
        self.current = ""
        self.user = None
        if os.name == "nt":
            self.user = ctypes.WinDLL("user32", use_last_error=True)
            self.user.RegisterHotKey.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                wintypes.UINT,
                wintypes.UINT,
            ]
            self.user.RegisterHotKey.restype = wintypes.BOOL
            self.user.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            application.installNativeEventFilter(self)

    def register(self, shortcut: str):
        modifiers, key = parse_hotkey(shortcut)
        if self.user is None:
            raise RuntimeError("Глобальная горячая клавиша поддерживается только в Windows")
        if shortcut == self.current:
            return
        previous = self.current
        if previous:
            self.user.UnregisterHotKey(None, HOTKEY_ID)
        if not self.user.RegisterHotKey(None, HOTKEY_ID, modifiers, key):
            if previous:
                old_modifiers, old_key = parse_hotkey(previous)
                self.user.RegisterHotKey(None, HOTKEY_ID, old_modifiers, old_key)
            raise RuntimeError(
                "Сочетание занято другой программой. Выберите другую горячую клавишу."
            )
        self.current = shortcut

    def nativeEventFilter(self, event_type, message):
        if event_type in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            native = wintypes.MSG.from_address(int(message))
            if self.current and native.message == 0x0312 and native.wParam == HOTKEY_ID:
                self.callback()
                return True, 0
        return False, 0

    def close(self):
        if self.user and self.current:
            self.user.UnregisterHotKey(None, HOTKEY_ID)
        self.application.removeNativeEventFilter(self)
        self.current = ""
