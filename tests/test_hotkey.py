import ctypes

import pytest
from PySide6.QtTest import QTest

from jev_launcher.hotkey import DEFAULT_HOTKEY, HOTKEY_ID, GlobalHotkey


def test_register_dispatch_conflict_and_unregister(qapp):
    calls = []
    hotkey = GlobalHotkey(qapp, lambda: calls.append(True))
    other = GlobalHotkey(qapp, lambda: None)
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
