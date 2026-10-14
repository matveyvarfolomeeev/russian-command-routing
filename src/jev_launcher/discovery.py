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


def resolve_advertised_shortcut(shortcut: str) -> str:
    """Query installed MSI components; these APIs never install or repair anything.

    https://learn.microsoft.com/windows/win32/api/msi/nf-msi-msigetshortcuttargetw
    """
    if os.name != "nt":
        return ""
    msi = ctypes.WinDLL("msi")
    msi.MsiGetShortcutTargetW.argtypes = [ctypes.c_wchar_p] * 4
    msi.MsiGetShortcutTargetW.restype = ctypes.c_uint
    msi.MsiGetComponentPathW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    msi.MsiGetComponentPathW.restype = ctypes.c_int
    product, feature, component = (ctypes.create_unicode_buffer(256) for _ in range(3))
    if msi.MsiGetShortcutTargetW(shortcut, product, feature, component):
        return ""
    target = ctypes.create_unicode_buffer(32768)
    size = ctypes.c_uint(len(target))
    state = msi.MsiGetComponentPathW(product.value, component.value, target, ctypes.byref(size))
    return target.value if state == 3 and Path(target.value).exists() else ""


def parse_scan(payload: dict) -> ScanResult:
    known = json.loads((RESOURCES / "known_apps.json").read_text(encoding="utf-8"))
    result = ScanResult(
        successful_sources=list(payload.get("successful_sources", [])),
        errors=dict(payload.get("errors", {})),
    )
    entries = payload.get("entries", [])
    if isinstance(entries, dict):
        entries = [entries]
    entries = [dict(entry) for entry in entries]
    for entry in entries:
        # WScript reports installer resources for advertised shortcuts, not the real target.
        if (
            entry["launch_type"] == "shortcut"
            and "\\installer\\" in (entry.get("resolved_target") or "").casefold()
        ):
            actual = resolve_advertised_shortcut(entry["launch_target"])
            if actual:
                entry["resolved_target"] = actual
                entry["icon_target"] = actual
    # Shell registration supplies the executable and argv behind opaque/GUID AppIDs.
    # A registration joins a shortcut only when its command is unambiguous.
    app_commands: dict[str, set[str]] = {}
    desktop_commands: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for entry in entries:
        resolved = entry.get("resolved_target") or ""
        arguments = split_windows_arguments(entry.get("raw_arguments") or "")
        app_id = entry.get("app_id")
        if entry["launch_type"] == "shortcut" and resolved:
            command = command_identity(resolved, arguments, entry.get("working_directory") or "")
            if app_id:
                app_commands.setdefault(app_id.casefold(), set()).add(command)
            desktop_commands.setdefault((path_key(resolved), arguments), set()).add(command)
    for entry in entries:
        target = entry["launch_target"]
        launch_type = entry["launch_type"]
        kind = "folder" if launch_type == "folder" else "app"
        resolved = entry.get("resolved_target") or ""
        arguments = split_windows_arguments(entry.get("raw_arguments") or "")
        cwd = entry.get("working_directory") or ""
        if is_maintenance(entry["name"], resolved or target, arguments):
            continue
        # Get-StartApps may return a directory, document or executable instead of an AppID.
        if launch_type == "app_id" and Path(target).exists():
            launch_type = target_type(target)
            kind = "app" if launch_type == "executable" else launch_type
            resolved = target
            cwd = str(Path(target).parent) if launch_type == "executable" else ""
        identity = ""
        if launch_type == "shortcut" and resolved:
            identity = command_identity(resolved, arguments, cwd)
            kind = (
                "folder"
                if target_type(resolved) == "folder"
                else ("app" if target_type(resolved) == "executable" else "file")
            )
        elif launch_type == "shortcut" and entry.get("app_id"):
            identity = launch_identity("app_id", entry["app_id"])
        if entry["launch_type"] == "app_id":
            commands = app_commands.get(target.casefold(), set())
            if not commands and resolved:
                commands = desktop_commands.get((path_key(resolved), arguments), set())
            if len(commands) == 1:
                identity = next(iter(commands))
                command_type, canonical_target, argv, cwd = json.loads(identity)
                arguments = tuple(argv)
                resolved = resolved or canonical_target
                kind = "app" if command_type in ("executable", "app_id") else command_type
            elif resolved and not resolved.startswith(("shell:", "::{")):
                identity = command_identity(resolved, arguments, cwd)
                kind = "app" if target_type(resolved) == "executable" else target_type(resolved)
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
                aliases=tuple(builtin.get("aliases", entry.get("aliases", ()))),
                arguments=arguments,
                working_directory=cwd,
                identity=identity,
                icon_target=entry.get("icon_target") or resolved or target,
                resolved_target=resolved,
                available=True if launch_type == "app_id" else Path(target).exists(),
            )
        )
    return result


def path_key(target: str) -> str:
    return ntpath.normcase(ntpath.normpath(os.path.expandvars(target)))


def target_type(target: str) -> str:
    if Path(target).is_dir():
        return "folder"
    return "executable" if ntpath.splitext(target)[1].casefold() == ".exe" else "file"


def command_identity(target: str, arguments: tuple[str, ...], cwd: str) -> str:
    kind = target_type(target)
    # Document/folder Shell opens do not use the shortcut's working directory.
    return launch_identity(kind, target, arguments, cwd if kind == "executable" else "")


def scan_windows(folder_roots: list[str] | None = None) -> ScanResult:
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
            env={**os.environ, "JEV_LAUNCHER_FOLDER_ROOTS": json.dumps(folder_roots or [])},
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Сканирование заняло слишком много времени. Повторите обновление."
        ) from exc
    if process.returncode:
        raise RuntimeError("Не удалось собрать приложения Windows")
    try:
        result = parse_scan(json.loads(process.stdout))
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Windows вернула неполный результат сканирования") from exc
    from .app_metadata import enrich_descriptions

    enrich_descriptions(result.objects)
    return result
