"""Global Windows hotkey with an explicit registration status."""

import ctypes
import logging
import os
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

HOTKEY_ID = 0x4A45
DEFAULT_HOTKEY = "Ctrl+Shift+Space"


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
    def __init__(self, application, callback, on_change=None):
        super().__init__()
        self.application = application
        self.callback = callback
        self.on_change = on_change
        self.current = ""
        self.last_error = ""
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
            self.user.UnregisterHotKey.restype = wintypes.BOOL
            application.installNativeEventFilter(self)

    @property
    def registered(self) -> bool:
        return bool(self.current)

    @property
    def status_text(self) -> str:
        if self.current:
            return f"Активно: {self.current}"
        return self.last_error or "Горячая клавиша отключена"

    def _changed(self):
        if self.on_change:
            self.on_change()

    def register(self, shortcut: str):
        try:
            modifiers, key = parse_hotkey(shortcut)
        except ValueError as exc:
            self.last_error = str(exc)
            self._changed()
            raise
        if self.user is None:
            self.last_error = "Глобальная горячая клавиша поддерживается только в Windows"
            self._changed()
            raise RuntimeError(self.last_error)
        if self.current and parse_hotkey(self.current) == (modifiers, key):
            self.last_error = ""
            self._changed()
            return
        previous = self.current
        if previous:
            self.user.UnregisterHotKey(None, HOTKEY_ID)
            self.current = ""
        if not self.user.RegisterHotKey(None, HOTKEY_ID, modifiers, key):
            code = ctypes.get_last_error()
            if previous:
                old_modifiers, old_key = parse_hotkey(previous)
                if self.user.RegisterHotKey(None, HOTKEY_ID, old_modifiers, old_key):
                    self.current = previous
            self.last_error = (
                "Сочетание занято другой программой. Выберите другую горячую клавишу."
                if code == 1409
                else f"Не удалось зарегистрировать горячую клавишу (Windows: {code})."
            )
            self._changed()
            logging.warning("Global hotkey registration failed: shortcut=%r error=%s", shortcut, code)
            raise RuntimeError(self.last_error)
        self.current = shortcut
        self.last_error = ""
        self._changed()
        logging.info("Global hotkey registered: %s", self.current)

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
        self._changed()
