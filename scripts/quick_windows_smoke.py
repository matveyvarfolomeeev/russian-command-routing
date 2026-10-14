"""Exercise the actual app/tray/hotkeys and warm Laya, with substituted ASR/opening."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSystemTrayIcon

from jev_launcher import app as app_module
from jev_launcher import config
from jev_launcher import quick as quick_module
from jev_launcher.catalog import Catalog, default_data_dir
from jev_launcher.hotkey import DEFAULT_HOTKEY, DEFAULT_VOICE_HOTKEY, parse_hotkey


class Keyboard(ctypes.Structure):
    _fields_ = [("key", wintypes.WORD), ("scan", wintypes.WORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("extra", ctypes.c_size_t)]


class InputData(ctypes.Union):
    _fields_ = [("keyboard", Keyboard), ("mouse_padding", ctypes.c_byte * 32)]


class Input(ctypes.Structure):
    _fields_ = [("kind", wintypes.DWORD), ("data", InputData)]


def send_hotkey(shortcut):
    modifiers, key = parse_hotkey(shortcut)
    keys = [value for flag, value in ((2, 0x11), (1, 0x12), (4, 0x10), (8, 0x5B))
            if modifiers & flag] + [key]
    events = [Input(1, InputData(keyboard=Keyboard(value, 0, flags, 0, 0)))
              for values, flags in ((keys, 0), (keys[::-1], 2)) for value in values]
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    assert user.SendInput(len(events), (Input * len(events))(*events), ctypes.sizeof(Input)) == len(events)


def wait_until(predicate, timeout=8000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(25)
    if not predicate():
        raise RuntimeError("Smoke check timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/quick-launch-v1/lifecycle"))
    parser.add_argument("--send-input", action="store_true",
                        help="Use physical input emulation instead of native WM_HOTKEY dispatch")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    data = root / "profile"
    data.mkdir(parents=True, exist_ok=True)
    source = default_data_dir() / "catalog.sqlite3"
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(data / "catalog.sqlite3") as copy:
            original.backup(copy)
    catalog = Catalog(data / "catalog.sqlite3")
    catalog.set_setting("search_model", "laya")
    catalog.set_setting("hotkey", DEFAULT_HOTKEY)
    catalog.set_setting("voice_hotkey", DEFAULT_VOICE_HOTKEY)
    catalog.set_setting("groq_key", "")
    os.environ.pop("GROQ_API_KEY", None)
    real_create = app_module.create_window
    report, helpers = {}, []
    started = time.perf_counter()

    def create(application, catalog):
        engine, controller, hub = real_create(application, catalog)
        quick, window = controller.quick, controller.quick_window

        def invoke(hotkey):
            if args.send_input:
                send_hotkey(hotkey.current)
            else:
                kernel, user = ctypes.WinDLL("kernel32"), ctypes.WinDLL("user32")
                assert user.PostThreadMessageW(kernel.GetCurrentThreadId(), 0x0312, hotkey.hotkey_id, 0)

        def drive():
            try:
                assert hub.isVisible() and not window.isVisible()
                report["hub_ready_ms"] = round((time.perf_counter() - started) * 1000)
                assert controller.hotkeyRegistered and controller.voiceHotkeyRegistered
                report["hotkeys_registered"] = [controller.hotkey.current, controller.voice_hotkey.current]
                report["hotkey_dispatch"] = "SendInput" if args.send_input else "WM_HOTKEY"
                trays = application.findChildren(QSystemTrayIcon)
                assert len(trays) == 1 and trays[0].isVisible()
                tray = trays[0]
                hub.close()
                QTest.qWait(100)
                assert not hub.isVisible()
                report["close_keeps_tray"] = True
                invoke(controller.hotkey)
                wait_until(lambda: quick.visible and window.isActive())
                QTest.qWait(200)
                assert window.width() == 680 and window.height() == 218
                user = ctypes.WinDLL("user32")
                user.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
                style = user.GetWindowLongW(int(window.winId()), -20)
                assert style & 0x80  # WS_EX_TOOLWINDOW: no separate taskbar button.
                report["quick_tool_window"] = True
                assert window.grabWindow().save(str(root / "quick-native.png"))
                glaze = shutil.which("glazewm")
                if glaze:
                    snapshot = json.loads(subprocess.check_output([glaze, "query", "windows"],
                                                                encoding="utf-8"))
                    managed = snapshot["data"]["windows"]
                    report["glazewm_quick_managed"] = any(w["title"] == window.title() for w in managed)
                    assert not report["glazewm_quick_managed"]
                invoke(controller.hotkey)
                wait_until(lambda: not quick.visible)
                invoke(controller.voice_hotkey)
                wait_until(lambda: quick.visible and bool(quick.error))
                assert not quick.recording and not quick.submitted
                report["voice_hotkey_missing_key_error"] = quick.error
                invoke(controller.voice_hotkey)
                wait_until(lambda: not quick.visible)
                tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)
                wait_until(hub.isVisible)
                report["tray_activation_opens_hub"] = True
                hub.hide()
                second = subprocess.run([sys.executable, "-m", "jev_launcher", "--data-dir", str(data),
                                         "--no-scan"], timeout=15, check=True,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                wait_until(hub.isVisible)
                report["second_instance_exit"] = second.returncode
                wait_until(lambda: controller.modelReady or bool(controller._model_error), timeout=120000)
                assert controller.modelReady, controller.modelStatus
                report["model_ready_ms"] = round((time.perf_counter() - started) * 1000)
                helper = controller.model_runtime._process
                helpers.append(helper)
                report["helper_pid"] = helper.pid
                report["catalog_objects"] = controller.activeCount
                expected = next(item for item in controller._all
                                if item.name == "Notepad" and item.enabled and item.available)
                calls = []
                with patch.object(quick_module, "launch_by_id",
                                  side_effect=lambda db, value: calls.append(value) or db.get(value)):
                    quick.show("text")
                    quick.setQuery(expected.name)
                    quick.submit()
                    quick.submit()
                    wait_until(lambda: not quick.visible, timeout=30000)
                    assert calls == [expected.object_id], calls
                    report["real_model_text_opens_expected_once"] = True
                    def start(device):
                        quick.recorder.recording = True

                    def stop():
                        quick.recorder.recording = False
                        return SimpleNamespace(wav=lambda: b"mock-wav")

                    quick.recorder = SimpleNamespace(recording=False, start=start, stop=stop,
                        cancel=lambda: setattr(quick.recorder, "recording", False))
                    with patch.object(config, "api_key", return_value="smoke-only"), patch.object(
                            quick_module, "transcribe", return_value=expected.name):
                        quick.show("voice")
                        assert quick.recording
                        quick.submit()
                        quick.submit()
                        wait_until(lambda: not quick.visible, timeout=30000)
                    assert calls == [expected.object_id, expected.object_id]
                    report["real_model_mock_voice_opens_expected_once"] = True
                assert controller.model_runtime._process.pid == helper.pid
                report["worker_reused_across_panels"] = True
                assert hub.grabWindow().save(str(root / "hub-native.png"))
                quit_action = next(action for action in tray.contextMenu().actions()
                                   if action.text() == "Выйти")
                quit_action.trigger()
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["quick_visible_at_error"] = quick.visible
                report["quick_active_at_error"] = window.isActive()
                if controller.model_runtime._process:
                    helpers.append(controller.model_runtime._process)
                application.quit()

        QTimer.singleShot(700, drive)
        return engine, controller, hub

    with patch.object(app_module, "create_window", side_effect=create):
        code = app_module.main(["--data-dir", str(data), "--no-scan"])
    report["exit_code"] = code
    report["helpers_stopped_on_quit"] = bool(helpers) and all(p.poll() is not None for p in helpers)
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if "error" in report or not report["helpers_stopped_on_quit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
