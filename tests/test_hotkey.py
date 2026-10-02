import ctypes

import pytest
from PySide6.QtTest import QTest

from jev_launcher.hotkey import HOTKEY_ID, GlobalHotkey


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
