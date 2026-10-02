"""Windows integration check. --open-objects also opens benign test objects."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from jev_launcher.catalog import Catalog
from jev_launcher.discovery import scan_windows
from jev_launcher.launch import launch_by_id

USER = ctypes.WinDLL("user32", use_last_error=True)
CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
USER.EnumWindows.argtypes = [CALLBACK, wintypes.LPARAM]
USER.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
USER.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
USER.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
USER.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]


def process_family(root_pid):
    class Entry(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("usage", wintypes.DWORD),
            ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t),
            ("module", wintypes.DWORD),
            ("threads", wintypes.DWORD),
            ("parent", wintypes.DWORD),
            ("priority", wintypes.LONG),
            ("flags", wintypes.DWORD),
            ("name", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return {root_pid}
    entry = Entry()
    entry.size = ctypes.sizeof(entry)
    parents = {}
    try:
        valid = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while valid:
            parents[entry.pid] = entry.parent
            valid = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    family = {root_pid}
    while True:
        expanded = family | {pid for pid, parent in parents.items() if parent in family}
        if expanded == family:
            return family
        family = expanded


def windows():
    values = []

    def visit(handle, parameter):
        if USER.IsWindowVisible(handle):
            pid = wintypes.DWORD()
            USER.GetWindowThreadProcessId(handle, ctypes.byref(pid))
            title, class_name = ctypes.create_unicode_buffer(512), ctypes.create_unicode_buffer(128)
            USER.GetWindowTextW(handle, title, 512)
            USER.GetClassNameW(handle, class_name, 128)
            values.append(
                {
                    "handle": int(handle),
                    "pid": pid.value,
                    "title": title.value,
                    "class": class_name.value,
                }
            )
        return True

    USER.EnumWindows(CALLBACK(visit), 0)
    return values


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        QTest.qWait(100)
    raise RuntimeError("Integration check timed out")


def packaged_check(executable: Path, root: Path, application):
    data_dir = root / "packaged-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "QT_PLUGIN_PATH",
        "QML_IMPORT_PATH",
        "QML2_IMPORT_PATH",
    ):
        environment.pop(name, None)
    command = [
        str(executable.resolve()),
        "--data-dir",
        str(data_dir.resolve()),
        "--no-hotkey",
        "--no-tray",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=root, env=environment)
    handle = None
    try:

        def find_window():
            if process.poll() is not None:
                raise RuntimeError(f"Packaged application exited: {process.returncode}")
            return next(
                (
                    window
                    for window in windows()
                    if window["pid"] in process_family(process.pid)
                    and window["title"] == "Jev Launcher"
                ),
                None,
            )

        native = wait_for(find_window, 20)
        window_ms = round((time.perf_counter() - started) * 1000)
        handle = native["handle"]
        print("Packaged window is visible", flush=True)

        def scan_completed():
            path = data_dir / "catalog.sqlite3"
            if not path.is_file():
                return None
            catalog = Catalog(path)
            if catalog.setting("last_scan") and catalog.model_catalog():
                return catalog
            return None

        catalog = wait_for(scan_completed, 50)
        assert not catalog.needs_refresh(), "Packaged catalog did not complete all sources"
        catalog_ready_ms = round((time.perf_counter() - started) * 1000)
        image = application.primaryScreen().grabWindow(handle)
        assert not image.isNull() and image.save(str(root / "packaged.png"))
        second = subprocess.run(command, cwd=root, env=environment, timeout=10, check=False)
        assert second.returncode == 0
        assert process.poll() is None
        USER.PostMessageW(handle, 0x0010, 0, 0)
        process.wait(10)
        assert process.returncode == 0
        return {
            "window_visible": True,
            "window_ready_ms": window_ms,
            "catalog_ready_ms": catalog_ready_ms,
            "catalog_count": len(catalog.items()),
            "active_count": len(catalog.model_catalog()),
            "source_errors": catalog.setting("last_scan")["errors"],
            "second_instance_exit": second.returncode,
            "close_exit": process.returncode,
        }
    finally:
        if process.poll() is None:
            if handle:
                USER.PostMessageW(handle, 0x0010, 0, 0)
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.terminate()
            else:
                process.terminate()


def cached_start_check(executable: Path, root: Path):
    """Same packaged process with an existing catalog; no fresh scan is expected."""
    catalog = Catalog(root / "packaged-data" / "catalog.sqlite3")
    assert not catalog.needs_refresh(), "Cached start requires a fully refreshed catalog"
    before = catalog.setting("last_scan")
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(executable.resolve()), "--data-dir", str(catalog.path.parent),
         "--no-hotkey", "--no-tray"], cwd=root,
    )
    handle = None
    try:
        def find_window():
            if process.poll() is not None:
                raise RuntimeError(f"Cached launch exited: {process.returncode}")
            return next((window for window in windows()
                         if window["pid"] in process_family(process.pid)
                         and window["title"] == "Jev Launcher"), None)

        native = wait_for(find_window, 20)
        handle = native["handle"]
        window_ms = round((time.perf_counter() - started) * 1000)
        QTest.qWait(1200)
        assert catalog.setting("last_scan") == before, "Fresh catalog was rescanned"
        assert catalog.model_catalog()
        USER.PostMessageW(handle, 0x0010, 0, 0)
        process.wait(10)
        assert process.returncode == 0
        return {"window_ready_ms": window_ms, "last_scan_unchanged": True,
                "active_count": len(catalog.model_catalog()), "close_exit": process.returncode}
    finally:
        if process.poll() is None:
            if handle:
                USER.PostMessageW(handle, 0x0010, 0, 0)
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.terminate()
            else:
                process.terminate()


def actual_open_check(root: Path):
    baseline = {window["handle"] for window in windows()}
    catalog = Catalog(root / "launch-data" / "catalog.sqlite3")
    catalog.merge_scan(scan_windows())
    folder = (root / "Проверка папки с пробелами").resolve()
    folder.mkdir(exist_ok=True)
    document = folder / "Проверка открытия.txt"
    document.write_text("Проверка открытия файла лаунчером.\n", encoding="utf-8")
    notepad_path = Path(os.environ["SystemRoot"]) / "System32" / "notepad.exe"
    link = folder / "Проверка ярлыка.lnk"
    environment = dict(
        os.environ,
        JEV_SMOKE_LINK=str(link),
        JEV_SMOKE_EXE=str(notepad_path),
        JEV_SMOKE_DOCUMENT=str(document),
    )
    script = (
        "$s = New-Object -ComObject WScript.Shell; "
        "$l = $s.CreateShortcut($env:JEV_SMOKE_LINK); "
        "$l.TargetPath = $env:JEV_SMOKE_EXE; "
        "$l.Arguments = [char]34 + $env:JEV_SMOKE_DOCUMENT + [char]34; $l.Save()"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=True,
    )
    result = {}
    try:
        executable = catalog.add_path(notepad_path)
        launch_by_id(catalog, executable.object_id)
        wait_for(
            lambda: any(
                "notepad" in w["title"].casefold() or "блокнот" in w["title"].casefold()
                for w in windows()
            )
        )
        result["executable_window"] = True
        calculator = next(
            item
            for item in catalog.items()
            if item.launch_type == "app_id" and "windowscalculator" in item.launch_target.casefold()
        )
        launch_by_id(catalog, calculator.object_id)
        wait_for(
            lambda: any(w["title"].casefold() in ("calculator", "калькулятор") for w in windows())
        )
        result["app_id_window"] = True
        item = catalog.add_path(folder)
        launch_by_id(catalog, item.object_id)
        wait_for(
            lambda: any(
                folder.name in w["title"] and w["class"] == "CabinetWClass" for w in windows()
            )
        )
        result["folder_window"] = True
        item = catalog.add_path(document)
        launch_by_id(catalog, item.object_id)
        wait_for(lambda: any(document.stem in w["title"] for w in windows()))
        result["associated_file_window"] = True
        item = catalog.add_path(link)
        launch_by_id(catalog, item.object_id)
        wait_for(lambda: any(document.stem in w["title"] for w in windows()))
        result["shortcut_dispatched_with_document_visible"] = True
        return result
    finally:
        markers = (
            "notepad",
            "блокнот",
            "calculator",
            "калькулятор",
            document.stem.casefold(),
            folder.name.casefold(),
        )
        for window in windows():
            if window["handle"] not in baseline and any(
                marker in window["title"].casefold() for marker in markers
            ):
                USER.PostMessageW(window["handle"], 0x0010, 0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/windows-smoke"))
    parser.add_argument("--open-objects", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.output_dir.resolve()
    application = QApplication([])
    report = {}
    if args.exe:
        report["packaged"] = packaged_check(args.exe, root, application)
        report["cached_start"] = cached_start_check(args.exe, root)
    if args.open_objects:
        report["actual_open"] = actual_open_check(root)
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
