"""Windows discovery runs in one hidden, bounded system helper."""

from __future__ import annotations

import base64
import ctypes
import json
import ntpath
import os
import subprocess
from pathlib import Path

from .objects import DiscoveredObject, ScanResult, is_maintenance, launch_identity

RESOURCES = Path(__file__).parent / "resources"


def split_windows_arguments(arguments: str) -> tuple[str, ...]:
    if not arguments:
        return ()
    argc = ctypes.c_int()
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv = shell.CommandLineToArgvW("placeholder.exe " + arguments, ctypes.byref(argc))
    if not argv:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return tuple(argv[index] for index in range(1, argc.value))
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


def parse_scan(payload: dict) -> ScanResult:
    known = json.loads((RESOURCES / "known_apps.json").read_text(encoding="utf-8"))
    result = ScanResult(
        successful_sources=list(payload.get("successful_sources", [])),
        errors=dict(payload.get("errors", {})),
    )
    entries = payload.get("entries", [])
    if isinstance(entries, dict):
        entries = [entries]
    # An AppUserModel.ID can differ between two shortcuts launching the same desktop app.
    # Match it to the actual command only when shortcut metadata identifies one command.
    app_commands: dict[str, set[str]] = {}
    desktop_commands: dict[str, set[str]] = {}
    for entry in entries:
        resolved = entry.get("resolved_target") or ""
        arguments = split_windows_arguments(entry.get("raw_arguments") or "")
        app_id = entry.get("app_id")
        if entry["launch_type"] == "shortcut" and resolved and not arguments:
            command = launch_identity(
                "executable" if ntpath.splitext(resolved)[1].casefold() == ".exe" else "file",
                resolved,
                (),
                entry.get("working_directory") or "",
            )
            if app_id:
                app_commands.setdefault(app_id.casefold(), set()).add(command)
            if ntpath.splitext(resolved)[1].casefold() == ".exe":
                desktop_commands.setdefault(ntpath.normcase(ntpath.normpath(resolved)), set()).add(
                    command
                )
    for entry in entries:
        target = entry["launch_target"]
        launch_type = entry["launch_type"]
        kind = "app"
        resolved = entry.get("resolved_target") or ""
        arguments = split_windows_arguments(entry.get("raw_arguments") or "")
        cwd = entry.get("working_directory") or ""
        if is_maintenance(entry["name"], resolved or target, arguments):
            continue
        # Get-StartApps also returns absolute executable/document paths for desktop entries.
        if launch_type == "app_id" and Path(target).is_file():
            launch_type = "executable" if Path(target).suffix.lower() == ".exe" else "file"
            kind = "app" if launch_type == "executable" else "file"
            resolved = target
            cwd = str(Path(target).parent) if launch_type == "executable" else ""
        identity = ""
        if launch_type == "shortcut" and resolved:
            identity = launch_identity(
                "executable" if ntpath.splitext(resolved)[1].casefold() == ".exe" else "file",
                resolved,
                arguments,
                cwd,
            )
        elif launch_type == "shortcut" and entry.get("app_id") and not arguments:
            identity = launch_identity("app_id", entry["app_id"])
        elif launch_type == "app_id":
            commands = app_commands.get(target.casefold(), set())
            if len(commands) == 1:
                identity = next(iter(commands))
        elif launch_type == "executable" and entry["launch_type"] == "app_id":
            # Get-StartApps provides no cwd. A single no-argument shortcut is authoritative;
            # multiple explicit cwd/profile variants remain separate candidates.
            commands = desktop_commands.get(ntpath.normcase(ntpath.normpath(target)), set())
            if len(commands) == 1:
                identity = next(iter(commands))
                cwd = json.loads(identity)[3]
        builtin = known.get(ntpath.basename(resolved).casefold(), {})
        if not builtin:
            app_id = target.casefold()
            builtin = next(
                (known[name] for name in ("calculator", "notepad") if name in app_id), {}
            )
        description = builtin.get("description") or entry.get("description") or ""
        result.objects.append(
            DiscoveredObject(
                name=entry["name"],
                launch_target=target,
                launch_type=launch_type,
                kind=kind,
                source=entry["source"],
                source_key=entry["source_key"],
                description=description,
                description_source="builtin" if builtin else ("metadata" if description else ""),
                aliases=tuple(builtin.get("aliases", ())),
                arguments=arguments,
                working_directory=cwd,
                identity=identity,
                icon_target=entry.get("icon_target") or target,
                resolved_target=resolved,
                available=True if launch_type == "app_id" else Path(target).exists(),
            )
        )
    return result


def scan_windows() -> ScanResult:
    if os.name != "nt":
        raise RuntimeError("Сбор приложений поддерживается только в Windows")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / (
        "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    try:
        # Windows PowerShell 5.1 reads UTF-8 .ps1 without a BOM as ANSI.
        # Encoding the fixed script also avoids putting user paths in shell code.
        encoded = base64.b64encode(
            (RESOURCES / "discover.ps1").read_text(encoding="utf-8").encode("utf-16-le")
        ).decode("ascii")
        process = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            encoding="utf-8-sig",
            errors="replace",
            timeout=45,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Сканирование заняло слишком много времени. Повторите обновление."
        ) from exc
    if process.returncode:
        raise RuntimeError("Не удалось собрать приложения Windows")
    try:
        return parse_scan(json.loads(process.stdout))
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Windows вернула неполный результат сканирования") from exc
