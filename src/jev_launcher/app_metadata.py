"""Fill missing app descriptions from Windows resources and exact WinGet matches."""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import ntpath
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

MAX_DESCRIPTION = 400
LOG = logging.getLogger(__name__)


def short_description(value: str) -> str:
    text = " ".join(value.split())
    if len(text) <= MAX_DESCRIPTION:
        return text
    return text[: MAX_DESCRIPTION - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"


def executable_metadata(target: str) -> dict[str, str]:
    """Read version resources, without loading or executing the target program."""
    if os.name != "nt" or ntpath.splitext(target)[1].casefold() != ".exe":
        return {}
    try:
        version = ctypes.WinDLL("version", use_last_error=True)
        version.GetFileVersionInfoSizeW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
        version.GetFileVersionInfoSizeW.restype = ctypes.c_uint32
        version.GetFileVersionInfoW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ]
        version.GetFileVersionInfoW.restype = ctypes.c_int
        version.VerQueryValueW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
        ]
        version.VerQueryValueW.restype = ctypes.c_int
        size = version.GetFileVersionInfoSizeW(target, None)
        if not size or size > 4_000_000:
            return {}
        data = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(target, 0, size, data):
            return {}
        pointer, length = ctypes.c_void_p(), ctypes.c_uint()
        translations = [(0x0409, 0x04B0)]
        if version.VerQueryValueW(
            data, r"\VarFileInfo\Translation", ctypes.byref(pointer), ctypes.byref(length)
        ) and pointer.value and length.value:
            translations = list(struct.iter_unpack(
                "<HH", ctypes.string_at(pointer, length.value - length.value % 4)
            ))
        result = {}
        for field in ("FileDescription", "ProductName", "CompanyName"):
            for language, codepage in translations:
                key = f"\\StringFileInfo\\{language:04x}{codepage:04x}\\{field}"
                if version.VerQueryValueW(
                    data, key, ctypes.byref(pointer), ctypes.byref(length)
                ) and pointer.value and 1 < length.value < 32_768:
                    text = ctypes.wstring_at(pointer, length.value - 1).strip()
                    if text:
                        result[field] = text
                        break
        return result
    except (OSError, ValueError, struct.error):
        return {}


def _name(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_package_description(output: str, expected_name: str) -> dict | None:
    """Read only the description block of one exact English/Russian CLI result."""
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    found = re.findall(
        r"^(?:Found|Найден[ао]?) (.+?) \[([A-Za-z0-9][A-Za-z0-9._+\-]*)\]\s*$",
        output, re.MULTILINE,
    )
    if len(found) != 1 or _name(found[0][0]) != _name(expected_name):
        return None
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(?:Description|Описание):\s*(.*)$", line)
        if not match:
            continue
        parts = [match[1]] if match[1] else []
        for following in lines[index + 1:]:
            if following and not following[0].isspace():
                break
            parts.append(following.strip())
        description = short_description(" ".join(parts))
        if description and _name(description) != _name(expected_name):
            version = re.search(r"^(?:Version|Версия):\s*(.+)$", output, re.MULTILINE)
            return {"package_id": found[0][1], "description": description, "name": found[0][0],
                    "version": version[1].strip() if version else "",
                    "source": "winget", "locale": "en-US"}
    return None


def installed_package_versions(timeout: float) -> dict[str, str] | None:
    winget = shutil.which("winget")
    if not winget:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="jev-winget-") as directory:
            output = Path(directory) / "installed.json"
            process = subprocess.run(
                [winget, "export", "--output", str(output), "--source", "winget",
                 "--include-versions", "--disable-interactivity"],
                capture_output=True, timeout=timeout, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if process.returncode != 0:
                return None
            exported = json.loads(output.read_text(encoding="utf-8-sig"))
            return {package["PackageIdentifier"]: package.get("Version", "")
                    for source in exported["Sources"] for package in source["Packages"]}
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None


def fetch_package_description(
    name: str, timeout: float, installed_versions: dict[str, str]
) -> tuple[dict | None, bool]:
    """No installs, prompts, fuzzy matches, or automatic source-agreement acceptance."""
    winget = shutil.which("winget")
    if not winget:
        return None, False
    deadline = time.monotonic() + timeout

    def query(arguments):
        return subprocess.run(
            [winget, "show", *arguments, "--exact", "--source", "winget",
             "--locale", "en-US", "--disable-interactivity"],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=max(.001, deadline - time.monotonic()),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, check=False,
        )

    try:
        process = query(["--name", name])
        if process.returncode == 0:
            package = parse_package_description(process.stdout, name)
            if not package:
                return None, True
            installed = installed_versions.get(package["package_id"])
            if not installed or installed.casefold() == "unknown":
                return None, True
            if installed != package["version"]:
                process = query(["--id", package["package_id"], "--version", installed])
                if process.returncode != 0:
                    return None, False
                historical = parse_package_description(process.stdout, name)
                if not historical or historical["package_id"] != package["package_id"]:
                    return None, True
                package = historical
            if package["version"] != installed:
                return None, True
            return package, True
    except (OSError, subprocess.TimeoutExpired):
        return None, False
    # Do not cache a network/source error as a permanent missing package.
    absent = any(text in process.stdout for text in (
        "No package found matching input criteria", "Multiple packages found",
        "Не найден пакет", "Найдено несколько пакетов",
    ))
    return None, absent


def _load_cache(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") == 2 and isinstance(value.get("packages"), dict):
            return value["packages"]
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def _save_cache(path: Path, packages: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: rebuildable cache; invalid/interrupted JSON is ignored on the next scan.
        # Windows can reject os.replace in the encrypted profile directory with WinError 17.
        path.write_text(json.dumps({"schema_version": 2, "packages": packages}, ensure_ascii=False),
                        encoding="utf-8")
    except OSError as exc:
        LOG.warning("Не удалось сохранить кэш описаний приложений: %s", exc)


def enrich_descriptions(
    objects, *, cache_path: Path | None = None, package_budget: float = 15,
    installed_versions: dict[str, str] | None = None,
) -> dict:
    """Only fill empty, unedited app descriptions; bound uncached network work per scan."""
    if cache_path is None:
        from .catalog import default_data_dir

        cache_path = default_data_dir() / "app-descriptions.json"
    packages = _load_cache(cache_path)
    changed_cache = False
    metadata = {}
    deadline = time.monotonic() + max(0, package_budget)
    stats = {"windows": 0, "winget": 0, "cache_hits": 0, "lookups": 0, "unavailable": 0}
    eligible = [item for item in objects if item.kind == "app" and not item.description.strip()
                and "description" not in getattr(item, "overrides", ())]
    if not eligible:
        return stats
    snapshot = packages.get("__installed__", {})
    reused_snapshot = False
    if installed_versions is None and isinstance(snapshot, dict) and isinstance(
        snapshot.get("checked_at"), (int, float)
    ) and 0 <= time.time() - snapshot["checked_at"] < 86_400:
        installed_versions = snapshot.get("versions")
        reused_snapshot = True
    remaining = deadline - time.monotonic()
    if installed_versions is None and os.name == "nt" and remaining > 0:
        installed_versions = installed_package_versions(min(30, remaining))
    if isinstance(installed_versions, dict):
        if not reused_snapshot:
            packages["__installed__"] = {"checked_at": time.time(), "versions": installed_versions}
            changed_cache = True
    else:
        installed_versions = {}
    signature = hashlib.sha256(json.dumps(installed_versions, sort_keys=True).encode()).hexdigest()
    for item in objects:
        if (item.kind != "app" or item.description.strip()
                or "description" in getattr(item, "overrides", ())):
            continue
        target = item.resolved_target or (
            item.launch_target if item.launch_type == "executable" else ""
        )
        # Browser-hosted shortcuts describe the website, not their host executable.
        hosted = any(re.match(r"--app(?:-id)?(?:=|$)", arg) for arg in item.arguments)
        if target and not hosted and target not in metadata:
            metadata[target] = executable_metadata(target)
        fields = {} if hosted else metadata.get(target, {})
        queries = [item.name]
        packaged = item.launch_type == "app_id" and "!" in item.launch_target and "_" in item.launch_target
        if packaged:
            # A Store package title cannot identify an unrelated same-name WinGet package.
            queries = []
        product = fields.get("ProductName", "")
        if not packaged and product and _name(product) != _name(item.name) and not re.search(
            r"windows|operating system|операционная система", product, re.IGNORECASE
        ):
            queries.append(product)
        package = None
        for query in dict.fromkeys(queries):
            key = _name(query)
            record = packages.get(key)
            if isinstance(record, dict) and record.get("installed_signature") == signature:
                ttl = 7 * 86_400 if record.get("package") else 86_400
                if isinstance(record.get("checked_at"), (float, int)) and (
                    0 <= time.time() - record["checked_at"] < ttl
                ):
                    stats["cache_hits"] += 1
                    cached = record.get("package")
                    package = cached if isinstance(cached, dict) and all(
                        isinstance(cached.get(field), str) and cached[field]
                        for field in ("description", "package_id", "name")
                    ) and _name(cached["name"]) == key else None
                    if package:
                        break
                    continue
            remaining = deadline - time.monotonic()
            if os.name != "nt" or remaining <= 0 or not installed_versions:
                continue
            stats["lookups"] += 1
            if stats["lookups"] % 10 == 0:
                LOG.info("WinGet: проверено %s точных имён приложений", stats["lookups"])
            package, cacheable = fetch_package_description(query, min(8, remaining), installed_versions)
            if cacheable:
                packages[key] = {"checked_at": time.time(), "package": package,
                                 "installed_signature": signature}
                changed_cache = True
            else:
                stats["unavailable"] += 1
            if package:
                break
        if package:
            item.description = package["description"]
            item.description_source = "winget:" + package["package_id"]
            stats["winget"] += 1
            continue
        description = fields.get("FileDescription", "")
        duplicate = {_name(item.name), _name(product), _name(ntpath.splitext(ntpath.basename(target))[0])}
        if description and _name(description) not in duplicate:
            item.description = short_description(description)
            item.description_source = "windows_metadata"
            stats["windows"] += 1
    if changed_cache:
        _save_cache(cache_path, packages)
    return stats
