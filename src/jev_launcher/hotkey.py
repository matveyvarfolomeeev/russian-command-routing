"""Global Windows hotkey with an explicit registration status."""

import ctypes
import logging
import os
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

HOTKEY_ID = 0x4A45
VOICE_HOTKEY_ID = 0x4A46
DEFAULT_HOTKEY = "Ctrl+Shift+Space"
DEFAULT_VOICE_HOTKEY = "Ctrl+Alt+Space"


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
    def __init__(self, application, callback, on_change=None, hotkey_id=HOTKEY_ID):
        super().__init__()
        self.application = application
        self.callback = callback
        self.on_change = on_change
        self.hotkey_id = hotkey_id
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
            return f"Активно: {self.current}" + (f" · {self.last_error}" if self.last_error else "")
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
            self.user.UnregisterHotKey(None, self.hotkey_id)
            self.current = ""
        if not self.user.RegisterHotKey(None, self.hotkey_id, modifiers, key):
            code = ctypes.get_last_error()
            if previous:
                old_modifiers, old_key = parse_hotkey(previous)
                if self.user.RegisterHotKey(None, self.hotkey_id, old_modifiers, old_key):
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
            if self.current and native.message == 0x0312 and native.wParam == self.hotkey_id:
                logging.info("Global hotkey activated: id=%s shortcut=%s", self.hotkey_id, self.current)
                self.callback()
                return True, 0
        return False, 0

    def unregister(self):
        if self.user and self.current:
            self.user.UnregisterHotKey(None, self.hotkey_id)
        self.current = ""
        self._changed()

    def close(self):
        self.unregister()
        self.application.removeNativeEventFilter(self)


def register_pair(text_hotkey, voice_hotkey, text, voice):
    """Register both shortcuts together, including swaps, and roll back on conflict."""
    if parse_hotkey(text) == parse_hotkey(voice):
        raise ValueError("Для текста и голоса нужны разные сочетания клавиш.")
    if text_hotkey.current == text and voice_hotkey.current == voice:
        text_hotkey.register(text)
        voice_hotkey.register(voice)
        return
    previous = (text_hotkey.current, voice_hotkey.current)
    text_hotkey.unregister()
    voice_hotkey.unregister()
    try:
        text_hotkey.register(text)
        voice_hotkey.register(voice)
    except (ValueError, RuntimeError) as exc:
        text_hotkey.unregister()
        voice_hotkey.unregister()
        restored = True
        for hotkey, shortcut in zip((text_hotkey, voice_hotkey), previous):
            if shortcut:
                try:
                    hotkey.register(shortcut)
                except (ValueError, RuntimeError):
                    restored = False
        message = str(exc)
        if any(previous):
            message += (
                " Прежние сочетания восстановлены." if restored
                else " Не удалось восстановить все прежние сочетания."
            )
        for hotkey in (text_hotkey, voice_hotkey):
            hotkey.last_error = message
            hotkey._changed()
        raise RuntimeError(message) from exc
