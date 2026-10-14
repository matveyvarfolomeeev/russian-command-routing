import ctypes

import pytest
from PySide6.QtTest import QTest

from jev_launcher.hotkey import (
    DEFAULT_HOTKEY,
    DEFAULT_VOICE_HOTKEY,
    HOTKEY_ID,
    VOICE_HOTKEY_ID,
    GlobalHotkey,
    register_pair,
)


def test_register_dispatch_conflict_and_unregister(qapp):
    calls = []
    hotkey = GlobalHotkey(qapp, lambda: calls.append(True))
    other = GlobalHotkey(qapp, lambda: None, hotkey_id=VOICE_HOTKEY_ID)
    try:
        try:
            hotkey.register("Ctrl+Alt+Shift+F12")
        except RuntimeError:
            pytest.skip("Test shortcut is occupied on this desktop")
        # Send the same native thread message emitted by Windows for RegisterHotKey.
        kernel = ctypes.WinDLL("kernel32")
        user = ctypes.WinDLL("user32")
        assert user.PostThreadMessageW(kernel.GetCurrentThreadId(), 0x0312, HOTKEY_ID, 0)
        QTest.qWait(50)
        assert calls == [True]
        with pytest.raises(RuntimeError):
            other.register("Ctrl+Alt+Shift+F12")
    finally:
        other.close()
        hotkey.close()
    assert hotkey.current == ""


class FakeUser:
    def __init__(self):
        self.attempts = []
        self.allow = True

    def RegisterHotKey(self, hwnd, key_id, modifiers, key):
        self.attempts.append((modifiers, key))
        if self.allow and key == 0x20:
            return True
        ctypes.set_last_error(1409)
        return False

    def UnregisterHotKey(self, hwnd, key_id):
        return True


def test_registration_status_and_previous_shortcut_restored(qapp):
    changes = []
    hotkey = GlobalHotkey(qapp, lambda: None, lambda: changes.append(True))
    hotkey.user = FakeUser()
    try:
        assert not hotkey.registered
        hotkey.register(DEFAULT_HOTKEY)
        assert hotkey.registered
        assert hotkey.status_text == f"Активно: {DEFAULT_HOTKEY}"
        with pytest.raises(RuntimeError, match="занято"):
            hotkey.register("Ctrl+Shift+F11")
        assert hotkey.current == DEFAULT_HOTKEY
        assert hotkey.registered
        assert hotkey.last_error
        # Alternate spelling of an already registered combination is a no-op.
        attempts = len(hotkey.user.attempts)
        hotkey.register("ctrl + shift + space")
        assert len(hotkey.user.attempts) == attempts
        assert not hotkey.last_error
    finally:
        hotkey.close()
    assert not hotkey.registered
    assert len(changes) == 4


def test_failed_restore_reports_disabled_instead_of_stale_registration(qapp):
    hotkey = GlobalHotkey(qapp, lambda: None)
    hotkey.user = FakeUser()
    try:
        hotkey.register(DEFAULT_HOTKEY)
        hotkey.user.allow = False
        with pytest.raises(RuntimeError, match="занято"):
            hotkey.register("Ctrl+Shift+F11")
        assert not hotkey.registered
        assert hotkey.current == ""
        assert "занято" in hotkey.status_text
    finally:
        hotkey.close()


def test_two_native_ids_dispatch_separately_and_pair_rolls_back(qapp):
    calls = []
    text = GlobalHotkey(qapp, lambda: calls.append("text"))
    voice = GlobalHotkey(qapp, lambda: calls.append("voice"), hotkey_id=VOICE_HOTKEY_ID)
    blocker = GlobalHotkey(qapp, lambda: None, hotkey_id=VOICE_HOTKEY_ID + 1)
    try:
        try:
            register_pair(text, voice, "Ctrl+Alt+Shift+F10", "Ctrl+Alt+Shift+F11")
            blocker.register("Ctrl+Alt+Shift+F9")
        except RuntimeError:
            pytest.skip("Integration shortcuts are occupied")
        kernel, user = ctypes.WinDLL("kernel32"), ctypes.WinDLL("user32")
        user.PostThreadMessageW(kernel.GetCurrentThreadId(), 0x0312, HOTKEY_ID, 0)
        user.PostThreadMessageW(kernel.GetCurrentThreadId(), 0x0312, VOICE_HOTKEY_ID, 0)
        QTest.qWait(50)
        assert calls == ["text", "voice"]
        with pytest.raises(RuntimeError, match="восстановлены"):
            register_pair(text, voice, "Ctrl+Alt+Shift+F12", "Ctrl+Alt+Shift+F9")
        assert (text.current, voice.current) == ("Ctrl+Alt+Shift+F10", "Ctrl+Alt+Shift+F11")
        register_pair(text, voice, "Ctrl+Alt+Shift+F11", "Ctrl+Alt+Shift+F10")
        assert text.current.endswith("F11") and voice.current.endswith("F10")
        with pytest.raises(ValueError, match="разные"):
            register_pair(text, voice, DEFAULT_HOTKEY, "ctrl + shift + space")
        assert DEFAULT_VOICE_HOTKEY != DEFAULT_HOTKEY
    finally:
        text.close()
        voice.close()
        blocker.close()
